#!/usr/bin/env python3
"""Directional capsule setpoint probe for Track B control maintenance candidates."""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, field_validator, model_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.control_capsule_maintenance import (
    CAPSULE_CHANNELS,
    candidate_paths_from_replication_summary,
    components_from_state,
    continue_control_capsule_resume,
    zero_channels,
)
from development.evolution.config import CHANNELS
from development.gate_state_propagation_characterization import (
    CharacterizationConfig,
    MotifSpec,
    clone_state,
    load_json,
    motif_from_spec,
    run_to_pause_and_baseline,
    write_csv,
    write_json,
    write_jsonl,
)

DEFAULT_REPLICATION_SUMMARY = Path("data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json")
DEFAULT_OUTPUT_DIR = Path("data/diagnostics/control_setpoint_probe_20260514")
EPS = 1e-12

DirectionKind = Literal["basis", "gaussian", "endogenous"]
PerturbationSign = Literal["plus", "minus"]
SetpointResumeKind = Literal["control_preserved", "control_zero_at_resume", "control_clamped_zero"]


class FlexibleSchema(BaseModel):
    """Validate expected fields while preserving searchable extras."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class DirectionSpec(FlexibleSchema):
    kind: DirectionKind
    index: NonNegativeInt = 0
    vector: list[float]
    source_norm: float

    @field_validator("vector")
    @classmethod
    def vector_values_are_finite(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("direction vector must not be empty")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("direction vector must contain only finite values")
        return values


class ControlSetpointConfig(FlexibleSchema):
    candidate_paths: list[Path] = Field(default_factory=list)
    replication_summary_path: Path = DEFAULT_REPLICATION_SUMMARY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seeds: list[int] = Field(default_factory=lambda: [96, 97, 98])
    perturb_scales: list[float] = Field(default_factory=lambda: [0.35, 0.7])
    motifs: list[MotifSpec] = Field(
        default_factory=lambda: [MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)]
    )
    direction_motifs: list[MotifSpec] = Field(
        default_factory=lambda: [MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)]
    )
    hidden_size: NonNegativeInt = 32
    rank: NonNegativeInt = 2
    steps: NonNegativeInt = 128
    perturb_step: NonNegativeInt = 64
    capsule_pause_step: NonNegativeInt = 64
    tail_window: NonNegativeInt = 16
    epsilon: PositiveFloat = 0.1
    perturb_channel: str = "fast"
    device: str = "cpu"

    @model_validator(mode="after")
    def steps_are_consistent(self) -> "ControlSetpointConfig":
        if self.perturb_step > self.steps:
            raise ValueError("perturb_step must be <= steps")
        if self.capsule_pause_step > self.steps:
            raise ValueError("capsule_pause_step must be <= steps")
        if self.tail_window < 1:
            raise ValueError("tail_window must be >= 1")
        return self

    def as_characterization_config(self, candidate_path: Path) -> CharacterizationConfig:
        return CharacterizationConfig(
            candidate_path=candidate_path,
            archive_root=candidate_path.parent.parent,
            output_dir=self.output_dir,
            seeds=self.seeds,
            perturb_scales=self.perturb_scales,
            motifs=self.motifs,
            hidden_size=int(self.hidden_size),
            rank=int(self.rank),
            steps=int(self.steps),
            perturb_step=int(self.perturb_step),
            capsule_pause_step=int(self.capsule_pause_step),
            perturb_channel=self.perturb_channel,
            device=self.device,
            write_parameter_signatures=False,
        )


class ControlSetpointRow(FlexibleSchema):
    candidate_id: str
    candidate_path: str
    seed: int
    perturb_scale: float
    motif_family: str
    motif_index: int
    direction_kind: DirectionKind
    direction_index: int
    perturbation_sign: PerturbationSign
    resume_kind: SetpointResumeKind
    epsilon: float
    initial_projection: float
    tail_projection: float
    final_projection: float
    projection_decay: float
    tail_abs_projection: float
    mean_abs_projection: float
    directional_asymmetry: float | None = None
    setpoint_bias: float | None = None
    control_specific_correction: float | None = None

    @field_validator("*")
    @classmethod
    def numeric_values_are_finite(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric control setpoint fields must be finite")
        return value


def tensor_to_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(-1).detach().float().cpu()


def concatenate_capsule_components(components: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor_to_vector(components[channel]) for channel in CAPSULE_CHANNELS])


def normalize_direction(vector: torch.Tensor) -> torch.Tensor:
    direction = tensor_to_vector(vector)
    norm = float(direction.norm())
    if norm <= EPS:
        raise ValueError("direction vector norm must be positive")
    return direction / norm


def capsule_direction_from_signature(signature: torch.Tensor) -> torch.Tensor:
    channel_direction = normalize_direction(signature)
    return normalize_direction(torch.cat([channel_direction for _ in CAPSULE_CHANNELS]))


def endogenous_capsule_direction(
    previous_components: dict[str, torch.Tensor] | None,
    current_components: dict[str, torch.Tensor],
) -> DirectionSpec | None:
    if previous_components is None:
        return None
    delta = concatenate_capsule_components(current_components) - concatenate_capsule_components(previous_components)
    source_norm = float(delta.norm())
    if source_norm <= EPS:
        return None
    direction = normalize_direction(delta)
    return DirectionSpec(kind="endogenous", index=0, vector=direction.tolist(), source_norm=source_norm)


def directional_specs_for_baseline(
    *,
    config: ControlSetpointConfig,
    seed: int,
    previous_components: dict[str, torch.Tensor] | None,
    current_components: dict[str, torch.Tensor],
) -> list[DirectionSpec]:
    specs: list[DirectionSpec] = []
    for motif_spec in config.direction_motifs:
        motif = motif_from_spec(motif_spec, int(config.hidden_size), int(seed))
        direction = capsule_direction_from_signature(motif["vector"])
        kind: DirectionKind = "basis" if motif_spec.family == "basis" else "gaussian"
        specs.append(
            DirectionSpec(
                kind=kind,
                index=int(motif_spec.index),
                vector=direction.tolist(),
                source_norm=float(torch.norm(motif["vector"])),
            )
        )
    endogenous = endogenous_capsule_direction(previous_components, current_components)
    if endogenous is not None:
        specs.append(endogenous)
    return specs


def apply_capsule_perturbation(
    state: tuple[torch.Tensor, ...],
    direction: torch.Tensor,
    *,
    epsilon: float,
    sign: PerturbationSign,
) -> tuple[torch.Tensor, ...]:
    normalized = normalize_direction(direction)
    sign_value = 1.0 if sign == "plus" else -1.0
    width = state[0].numel()
    expected_width = width * len(CAPSULE_CHANNELS)
    if normalized.numel() != expected_width:
        raise ValueError(f"direction width {normalized.numel()} does not match capsule width {expected_width}")
    values = list(clone_state(state))
    offset = 0
    for channel in CAPSULE_CHANNELS:
        idx = list(CHANNELS).index(channel)
        piece = normalized[offset : offset + width].to(device=values[idx].device, dtype=values[idx].dtype).view_as(values[idx])
        values[idx] = values[idx] + float(epsilon) * sign_value * piece
        offset += width
    return tuple(values)


def signed_projection(
    components: dict[str, torch.Tensor],
    target_components: dict[str, torch.Tensor],
    direction: torch.Tensor,
    *,
    epsilon: float,
) -> float:
    normalized = normalize_direction(direction)
    delta = concatenate_capsule_components(components) - concatenate_capsule_components(target_components)
    if delta.numel() != normalized.numel():
        raise ValueError(f"capsule width {delta.numel()} does not match direction width {normalized.numel()}")
    return float(torch.dot(delta, normalized) / float(epsilon))


def projection_decay(initial_projection: float, tail_projection: float) -> float:
    return float(abs(initial_projection) - abs(tail_projection))


def directional_asymmetry(decay_plus: float, decay_minus: float) -> float:
    return float(abs(decay_plus - decay_minus) / (abs(decay_plus) + abs(decay_minus) + EPS))


def setpoint_bias(tail_projection_plus: float, tail_projection_minus: float) -> float:
    return float((tail_projection_plus + tail_projection_minus) / 2.0)


def row_for_directional_resume(
    *,
    candidate_id: str,
    candidate_path: Path,
    seed: int,
    perturb_scale: float,
    motif: dict[str, Any],
    direction_spec: DirectionSpec,
    perturbation_sign: PerturbationSign,
    resume_kind: SetpointResumeKind,
    initial_components: dict[str, torch.Tensor],
    target_initial_components: dict[str, torch.Tensor],
    components: list[dict[str, torch.Tensor]],
    target_components: list[dict[str, torch.Tensor]],
    tail_window: int,
    epsilon: float,
) -> ControlSetpointRow:
    direction = torch.tensor(direction_spec.vector, dtype=torch.float32)
    initial_projection = signed_projection(
        initial_components,
        target_initial_components,
        direction,
        epsilon=epsilon,
    )
    projections = [
        signed_projection(left, right, direction, epsilon=epsilon)
        for left, right in zip(components, target_components)
    ]
    tail_count = max(1, min(int(tail_window), len(projections)))
    tail_projection = float(mean(projections[-tail_count:])) if projections else initial_projection
    final_projection = float(projections[-1]) if projections else initial_projection
    abs_projections = [abs(value) for value in projections]
    return ControlSetpointRow(
        candidate_id=candidate_id,
        candidate_path=candidate_path.as_posix(),
        seed=int(seed),
        perturb_scale=float(perturb_scale),
        motif_family=str(motif["family"]),
        motif_index=int(motif["index"]),
        direction_kind=direction_spec.kind,
        direction_index=int(direction_spec.index),
        perturbation_sign=perturbation_sign,
        resume_kind=resume_kind,
        epsilon=float(epsilon),
        initial_projection=initial_projection,
        tail_projection=tail_projection,
        final_projection=final_projection,
        projection_decay=projection_decay(initial_projection, tail_projection),
        tail_abs_projection=float(mean(abs_projections[-tail_count:])) if abs_projections else abs(initial_projection),
        mean_abs_projection=float(mean(abs_projections)) if abs_projections else abs(initial_projection),
    )


def attach_pair_metrics(rows: list[ControlSetpointRow]) -> list[ControlSetpointRow]:
    by_pair: dict[tuple[str, int, float, str, int, DirectionKind, int, SetpointResumeKind], dict[str, ControlSetpointRow]] = (
        defaultdict(dict)
    )
    for row in rows:
        key = (
            row.candidate_id,
            row.seed,
            row.perturb_scale,
            row.motif_family,
            row.motif_index,
            row.direction_kind,
            row.direction_index,
            row.resume_kind,
        )
        by_pair[key][row.perturbation_sign] = row

    control_decays: dict[tuple[str, int, float, str, int, DirectionKind, int, PerturbationSign], float] = {}
    for row in rows:
        if row.resume_kind == "control_clamped_zero":
            key = (
                row.candidate_id,
                row.seed,
                row.perturb_scale,
                row.motif_family,
                row.motif_index,
                row.direction_kind,
                row.direction_index,
                row.perturbation_sign,
            )
            control_decays[key] = row.projection_decay

    updated: list[ControlSetpointRow] = []
    for row in rows:
        pair_key = (
            row.candidate_id,
            row.seed,
            row.perturb_scale,
            row.motif_family,
            row.motif_index,
            row.direction_kind,
            row.direction_index,
            row.resume_kind,
        )
        plus = by_pair[pair_key].get("plus")
        minus = by_pair[pair_key].get("minus")
        asymmetry = None
        bias = None
        if plus is not None and minus is not None:
            asymmetry = directional_asymmetry(plus.projection_decay, minus.projection_decay)
            bias = setpoint_bias(plus.tail_projection, minus.tail_projection)

        clamped_key = (
            row.candidate_id,
            row.seed,
            row.perturb_scale,
            row.motif_family,
            row.motif_index,
            row.direction_kind,
            row.direction_index,
            row.perturbation_sign,
        )
        control_specific = None
        if row.resume_kind == "control_preserved" and clamped_key in control_decays:
            control_specific = float(row.projection_decay - control_decays[clamped_key])

        updated.append(
            row.model_copy(
                update={
                    "directional_asymmetry": asymmetry,
                    "setpoint_bias": bias,
                    "control_specific_correction": control_specific,
                }
            )
        )
    return updated


def run_candidate_control_setpoint_probe(candidate_path: Path, config: ControlSetpointConfig) -> list[ControlSetpointRow]:
    candidate = load_json(candidate_path)
    candidate_id = str(candidate.get("id", candidate_path.stem))
    genome = candidate["genome"]
    characterization_config = config.as_characterization_config(candidate_path)
    rows: list[ControlSetpointRow] = []
    for seed in config.seeds:
        motifs = [motif_from_spec(spec, int(config.hidden_size), int(seed)) for spec in config.motifs]
        for perturb_scale in config.perturb_scales:
            for motif in motifs:
                baseline = run_to_pause_and_baseline(
                    genome,
                    candidate_id=candidate_id,
                    config=characterization_config,
                    seed=int(seed),
                    perturb_scale=float(perturb_scale),
                    motif=motif,
                )
                pause = int(config.capsule_pause_step)
                paused_model = baseline["paused_model"]
                paused_state = baseline["paused_state"]
                current_components = baseline["components"][pause - 1]
                previous_components = baseline["components"][pause - 2] if pause >= 2 else None
                target_components = baseline["components"][pause:]
                directions = directional_specs_for_baseline(
                    config=config,
                    seed=int(seed),
                    previous_components=previous_components,
                    current_components=current_components,
                )
                for direction_spec in directions:
                    direction = torch.tensor(direction_spec.vector, dtype=torch.float32)
                    for perturbation_sign in ("plus", "minus"):
                        perturbed_state = apply_capsule_perturbation(
                            paused_state,
                            direction,
                            epsilon=float(config.epsilon),
                            sign=perturbation_sign,
                        )
                        for resume_kind in (
                            "control_preserved",
                            "control_zero_at_resume",
                            "control_clamped_zero",
                        ):
                            resume_model = copy.deepcopy(paused_model)
                            resume_state = clone_state(perturbed_state)
                            if resume_kind in {"control_zero_at_resume", "control_clamped_zero"}:
                                resume_state = zero_channels(resume_state, ("control",))
                            initial_components = components_from_state(resume_model, resume_state)
                            clamp_channels = ("control",) if resume_kind == "control_clamped_zero" else ()
                            _, components = continue_control_capsule_resume(
                                resume_model,
                                resume_state,
                                start_step=pause,
                                steps=int(config.steps),
                                clamp_zero_channels=clamp_channels,
                            )
                            rows.append(
                                row_for_directional_resume(
                                    candidate_id=candidate_id,
                                    candidate_path=candidate_path,
                                    seed=int(seed),
                                    perturb_scale=float(perturb_scale),
                                    motif=motif,
                                    direction_spec=direction_spec,
                                    perturbation_sign=perturbation_sign,
                                    resume_kind=resume_kind,
                                    initial_components=initial_components,
                                    target_initial_components=current_components,
                                    components=components,
                                    target_components=target_components,
                                    tail_window=int(config.tail_window),
                                    epsilon=float(config.epsilon),
                                )
                            )
    return attach_pair_metrics(rows)


def mean_optional(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def summarize_rows(rows: list[ControlSetpointRow]) -> dict[str, Any]:
    grouped: dict[tuple[str, DirectionKind, int, SetpointResumeKind], list[ControlSetpointRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.candidate_id, row.direction_kind, row.direction_index, row.resume_kind)].append(row)

    numeric_fields = (
        "initial_projection",
        "tail_projection",
        "final_projection",
        "projection_decay",
        "tail_abs_projection",
        "mean_abs_projection",
        "directional_asymmetry",
        "setpoint_bias",
        "control_specific_correction",
    )
    by_candidate_direction_resume: list[dict[str, Any]] = []
    for (candidate_id, direction_kind, direction_index, resume_kind), items in sorted(grouped.items()):
        payload: dict[str, Any] = {
            "candidate_id": candidate_id,
            "direction_kind": direction_kind,
            "direction_index": direction_index,
            "resume_kind": resume_kind,
            "run_count": len(items),
        }
        for field in numeric_fields:
            values = [
                float(value)
                for value in (getattr(row, field) for row in items)
                if value is not None and math.isfinite(float(value))
            ]
            if values:
                payload[f"{field}_mean"] = float(mean(values))
                payload[f"{field}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        by_candidate_direction_resume.append(payload)

    preserved_rows = [row for row in rows if row.resume_kind == "control_preserved"]
    return {
        "experiment": "control_setpoint_probe",
        "row_count": len(rows),
        "candidate_count": len({row.candidate_id for row in rows}),
        "by_candidate_direction_resume": by_candidate_direction_resume,
        "aggregate": {
            "control_preserved_directional_asymmetry_mean": mean_optional(
                [row.directional_asymmetry for row in preserved_rows]
            ),
            "control_preserved_abs_setpoint_bias_mean": mean_optional(
                [abs(row.setpoint_bias) if row.setpoint_bias is not None else None for row in preserved_rows]
            ),
            "control_specific_correction_mean": mean_optional(
                [row.control_specific_correction for row in preserved_rows]
            ),
        },
        "interpretation_boundary": (
            "This probe tests directional capsule regulation. Promote only control-mediated setpoint-like "
            "capsule regulation if the effect replicates across candidates and beats control-clamped controls."
        ),
    }


def run_control_setpoint_probe(config: ControlSetpointConfig) -> dict[str, Any]:
    candidate_paths = config.candidate_paths or candidate_paths_from_replication_summary(config.replication_summary_path)
    rows: list[ControlSetpointRow] = []
    torch.set_grad_enabled(False)
    for candidate_path in candidate_paths:
        rows.extend(run_candidate_control_setpoint_probe(candidate_path, config))
    payload = summarize_rows(rows)
    payload["config"] = {
        **config.model_dump(mode="json"),
        "candidate_paths": [path.as_posix() for path in candidate_paths],
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.output_dir / "control_setpoint_rows.jsonl", rows)
    write_csv(config.output_dir / "control_setpoint_summary.csv", payload["by_candidate_direction_resume"])
    write_json(config.output_dir / "control_setpoint_summary.json", payload)
    write_readme(config.output_dir, payload)
    return payload


def write_readme(output_dir: Path, payload: dict[str, Any]) -> None:
    text = f"""# Control Setpoint Probe

Directional capsule perturbation probe for Track B control maintenance candidates.

Artifacts:

- `control_setpoint_rows.jsonl`: per-run signed perturbation and resume-arm rows
- `control_setpoint_summary.csv`: per-candidate, per-direction, per-resume aggregate metrics
- `control_setpoint_summary.json`: aggregate interpretation metrics and config

Interpretation boundary: {payload["interpretation_boundary"]}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_motifs(text: str) -> list[MotifSpec]:
    values: list[MotifSpec] = []
    for item in text.split(","):
        if not item.strip():
            continue
        family, _, index = item.partition(":")
        values.append(MotifSpec(family=family.strip(), index=int(index or 0)))
    return values


def parse_paths(text: str | None) -> list[Path]:
    if not text:
        return []
    return [Path(item.strip()) for item in text.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-paths", default=None)
    parser.add_argument("--replication-summary-path", type=Path, default=DEFAULT_REPLICATION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="96,97,98")
    parser.add_argument("--perturb-scales", default="0.35,0.7")
    parser.add_argument("--motifs", default="basis:0,gaussian:0")
    parser.add_argument("--direction-motifs", default="basis:0,gaussian:0")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--perturb-step", type=int, default=64)
    parser.add_argument("--capsule-pause-step", type=int, default=64)
    parser.add_argument("--tail-window", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ControlSetpointConfig(
        candidate_paths=parse_paths(args.candidate_paths),
        replication_summary_path=args.replication_summary_path,
        output_dir=args.output_dir,
        seeds=parse_csv_ints(args.seeds),
        perturb_scales=parse_csv_floats(args.perturb_scales),
        motifs=parse_motifs(args.motifs),
        direction_motifs=parse_motifs(args.direction_motifs),
        hidden_size=int(args.hidden_size),
        rank=int(args.rank),
        steps=int(args.steps),
        perturb_step=int(args.perturb_step),
        capsule_pause_step=int(args.capsule_pause_step),
        tail_window=int(args.tail_window),
        epsilon=float(args.epsilon),
        device=str(args.device),
    )
    payload = run_control_setpoint_probe(config)
    print(f"wrote {config.output_dir / 'control_setpoint_summary.json'}")
    print(f"rows={payload['row_count']}")


if __name__ == "__main__":
    main()
