#!/usr/bin/env python3
"""Control-mediated capsule maintenance probe for Track B candidates."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator, model_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolution.config import CHANNELS
from development.gate_state_propagation_characterization import (
    CharacterizationConfig,
    MotifSpec,
    clone_state,
    load_json,
    motif_from_spec,
    normalized_l2,
    run_to_pause_and_baseline,
    surface_resume_state,
    write_csv,
    write_json,
    write_jsonl,
    zero_like_state,
)

DEFAULT_REPLICATION_SUMMARY = Path("data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json")
DEFAULT_OUTPUT_DIR = Path("data/diagnostics/control_capsule_maintenance_20260514")
CAPSULE_CHANNELS = ("slow", "message", "carrier")

ControlCapsuleResumeKind = Literal[
    "full_internal_state",
    "surface_only",
    "zero_state",
    "control_zero_at_resume",
    "control_clamped_zero",
    "slow_carrier_zero_control_preserved",
    "control_only",
]


class FlexibleSchema(BaseModel):
    """Validate expected fields while preserving searchable extras."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class ControlCapsuleConfig(FlexibleSchema):
    candidate_paths: list[Path] = Field(default_factory=list)
    replication_summary_path: Path = DEFAULT_REPLICATION_SUMMARY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seeds: list[int] = Field(default_factory=lambda: [96, 97, 98])
    perturb_scales: list[float] = Field(default_factory=lambda: [0.35, 0.7])
    motifs: list[MotifSpec] = Field(
        default_factory=lambda: [MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)]
    )
    hidden_size: NonNegativeInt = 32
    rank: NonNegativeInt = 2
    steps: NonNegativeInt = 128
    perturb_step: NonNegativeInt = 64
    capsule_pause_step: NonNegativeInt = 64
    tail_window: NonNegativeInt = 16
    perturb_channel: str = "fast"
    device: str = "cpu"

    @model_validator(mode="after")
    def steps_are_consistent(self) -> "ControlCapsuleConfig":
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


class ControlCapsuleRow(FlexibleSchema):
    candidate_id: str
    candidate_path: str
    seed: int
    perturb_scale: float
    motif_family: str
    motif_index: int
    resume_kind: ControlCapsuleResumeKind
    final_cosine: float
    final_l2_gap: float
    mean_step_gap: float
    initial_capsule_gap: float
    tail_capsule_gap: float
    final_capsule_gap: float
    capsule_mean_step_gap: float
    capsule_pullback: float
    capsule_pullback_fraction: float | None
    fast_gap: float
    slow_gap: float
    control_gap: float
    message_gap: float
    carrier_gap: float
    control_maintenance_index: float | None = None
    continuous_control_maintenance_index: float | None = None

    @field_validator("*")
    @classmethod
    def numeric_values_are_finite(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric control capsule fields must be finite")
        return value


def tensor_to_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(-1).detach().float().cpu()


def components_from_state(model: Any, state: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    return {
        channel: tensor_to_vector(tensor)
        for channel, tensor in model.state_components(state).items()
    }


def concatenate_channels(components: dict[str, torch.Tensor], channels: tuple[str, ...]) -> torch.Tensor:
    return torch.cat([components[channel].view(-1).detach().float().cpu() for channel in channels])


def component_gap(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    channels: tuple[str, ...] = CAPSULE_CHANNELS,
) -> float:
    return normalized_l2(concatenate_channels(left, channels), concatenate_channels(right, channels))


def zero_channels(state: tuple[torch.Tensor, ...], channels: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    values = list(clone_state(state))
    for channel in channels:
        idx = list(CHANNELS).index(channel)
        values[idx] = torch.zeros_like(values[idx])
    return tuple(values)


def only_channels(state: tuple[torch.Tensor, ...], channels: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    values = list(zero_like_state(state))
    for channel in channels:
        idx = list(CHANNELS).index(channel)
        values[idx] = state[idx].detach().clone()
    return tuple(values)


def resume_state_for_kind(
    model: Any,
    paused_state: tuple[torch.Tensor, ...],
    resume_kind: ControlCapsuleResumeKind,
) -> tuple[torch.Tensor, ...]:
    if resume_kind == "full_internal_state":
        return clone_state(paused_state)
    if resume_kind == "surface_only":
        return surface_resume_state(model, paused_state)
    if resume_kind == "zero_state":
        return zero_like_state(paused_state)
    if resume_kind in {"control_zero_at_resume", "control_clamped_zero"}:
        return zero_channels(paused_state, ("control",))
    if resume_kind == "slow_carrier_zero_control_preserved":
        return zero_channels(paused_state, ("slow", "carrier"))
    if resume_kind == "control_only":
        return only_channels(paused_state, ("control",))
    raise ValueError(f"unknown resume kind: {resume_kind}")


def continue_control_capsule_resume(
    model: Any,
    state: tuple[torch.Tensor, ...],
    *,
    start_step: int,
    steps: int,
    clamp_zero_channels: tuple[str, ...] = (),
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    surfaces: list[torch.Tensor] = []
    components: list[dict[str, torch.Tensor]] = []
    current = clone_state(state)
    with torch.no_grad():
        for _ in range(start_step + 1, steps + 1):
            current = model.step(current)
            if clamp_zero_channels:
                current = zero_channels(current, clamp_zero_channels)
            surfaces.append(tensor_to_vector(model.state_vector(current)))
            components.append(components_from_state(model, current))
    return surfaces, components


def final_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = float(left.norm() * right.norm())
    if denom <= 1e-10:
        return 0.0
    return float(torch.dot(left.view(-1), right.view(-1)) / denom)


def row_for_resume_kind(
    *,
    candidate_id: str,
    candidate_path: Path,
    seed: int,
    perturb_scale: float,
    motif: dict[str, Any],
    resume_kind: ControlCapsuleResumeKind,
    surfaces: list[torch.Tensor],
    components: list[dict[str, torch.Tensor]],
    target_surfaces: list[torch.Tensor],
    target_components: list[dict[str, torch.Tensor]],
    initial_components: dict[str, torch.Tensor],
    target_initial_components: dict[str, torch.Tensor],
    tail_window: int,
) -> ControlCapsuleRow:
    final_left = surfaces[-1]
    final_right = target_surfaces[-1]
    step_gaps = [normalized_l2(left, right) for left, right in zip(surfaces, target_surfaces)]
    capsule_gaps = [
        component_gap(left, right)
        for left, right in zip(components, target_components)
    ]
    final_components = components[-1]
    target_final_components = target_components[-1]
    initial_capsule_gap = component_gap(initial_components, target_initial_components)
    tail_count = max(1, min(int(tail_window), len(capsule_gaps)))
    tail_capsule_gap = float(mean(capsule_gaps[-tail_count:])) if capsule_gaps else 0.0
    capsule_pullback = float(initial_capsule_gap - tail_capsule_gap)
    pullback_fraction = None
    if initial_capsule_gap > 1e-12:
        pullback_fraction = float(capsule_pullback / initial_capsule_gap)
    return ControlCapsuleRow(
        candidate_id=candidate_id,
        candidate_path=candidate_path.as_posix(),
        seed=int(seed),
        perturb_scale=float(perturb_scale),
        motif_family=str(motif["family"]),
        motif_index=int(motif["index"]),
        resume_kind=resume_kind,
        final_cosine=final_cosine(final_left, final_right),
        final_l2_gap=normalized_l2(final_left, final_right),
        mean_step_gap=float(mean(step_gaps)) if step_gaps else 0.0,
        initial_capsule_gap=float(initial_capsule_gap),
        tail_capsule_gap=tail_capsule_gap,
        final_capsule_gap=component_gap(final_components, target_final_components),
        capsule_mean_step_gap=float(mean(capsule_gaps)) if capsule_gaps else 0.0,
        capsule_pullback=capsule_pullback,
        capsule_pullback_fraction=pullback_fraction,
        **{
            f"{channel}_gap": normalized_l2(final_components[channel], target_final_components[channel])
            for channel in CHANNELS
        },
    )


def attach_maintenance_indices(rows: list[ControlCapsuleRow]) -> list[ControlCapsuleRow]:
    grouped: dict[tuple[str, int, float, str, int], dict[str, ControlCapsuleRow]] = defaultdict(dict)
    for row in rows:
        key = (row.candidate_id, row.seed, row.perturb_scale, row.motif_family, row.motif_index)
        grouped[key][row.resume_kind] = row
    updated: list[ControlCapsuleRow] = []
    for row in rows:
        key = (row.candidate_id, row.seed, row.perturb_scale, row.motif_family, row.motif_index)
        arms = grouped[key]
        full = arms.get("full_internal_state")
        surface = arms.get("surface_only")
        if full is None or surface is None:
            updated.append(row)
            continue
        denom = surface.mean_step_gap - full.mean_step_gap
        control_index = None
        continuous_index = None
        if denom > 1e-12:
            control = arms.get("control_zero_at_resume")
            continuous = arms.get("control_clamped_zero")
            if control is not None:
                control_index = float((control.mean_step_gap - full.mean_step_gap) / denom)
            if continuous is not None:
                continuous_index = float((continuous.mean_step_gap - full.mean_step_gap) / denom)
        updated.append(
            row.model_copy(
                update={
                    "control_maintenance_index": control_index,
                    "continuous_control_maintenance_index": continuous_index,
                }
            )
        )
    return updated


def run_candidate_control_capsule_probe(candidate_path: Path, config: ControlCapsuleConfig) -> list[ControlCapsuleRow]:
    candidate = load_json(candidate_path)
    candidate_id = str(candidate.get("id", candidate_path.stem))
    genome = candidate["genome"]
    characterization_config = config.as_characterization_config(candidate_path)
    rows: list[ControlCapsuleRow] = []
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
                target_surfaces = baseline["surfaces"][pause:]
                target_components = baseline["components"][pause:]
                target_initial_components = baseline["components"][pause - 1]
                paused_model = baseline["paused_model"]
                paused_state = baseline["paused_state"]
                for resume_kind in (
                    "full_internal_state",
                    "surface_only",
                    "zero_state",
                    "control_zero_at_resume",
                    "control_clamped_zero",
                    "slow_carrier_zero_control_preserved",
                    "control_only",
                ):
                    resume_model = copy.deepcopy(paused_model)
                    resume_state = resume_state_for_kind(resume_model, paused_state, resume_kind)
                    initial_components = components_from_state(resume_model, resume_state)
                    clamp_channels = ("control",) if resume_kind == "control_clamped_zero" else ()
                    surfaces, components = continue_control_capsule_resume(
                        resume_model,
                        resume_state,
                        start_step=pause,
                        steps=int(config.steps),
                        clamp_zero_channels=clamp_channels,
                    )
                    rows.append(
                        row_for_resume_kind(
                            candidate_id=candidate_id,
                            candidate_path=candidate_path,
                            seed=int(seed),
                            perturb_scale=float(perturb_scale),
                            motif=motif,
                            resume_kind=resume_kind,
                            surfaces=surfaces,
                            components=components,
                            target_surfaces=target_surfaces,
                            target_components=target_components,
                            initial_components=initial_components,
                            target_initial_components=target_initial_components,
                            tail_window=int(config.tail_window),
                        )
                    )
    return attach_maintenance_indices(rows)


def summarize_rows(rows: list[ControlCapsuleRow]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[ControlCapsuleRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.candidate_id, row.resume_kind)].append(row)
    by_candidate_resume = []
    numeric_fields = (
        "final_cosine",
        "final_l2_gap",
        "mean_step_gap",
        "initial_capsule_gap",
        "tail_capsule_gap",
        "final_capsule_gap",
        "capsule_mean_step_gap",
        "capsule_pullback",
        "capsule_pullback_fraction",
        "fast_gap",
        "slow_gap",
        "control_gap",
        "message_gap",
        "carrier_gap",
        "control_maintenance_index",
        "continuous_control_maintenance_index",
    )
    for (candidate_id, resume_kind), items in sorted(grouped.items()):
        payload: dict[str, Any] = {
            "candidate_id": candidate_id,
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
        by_candidate_resume.append(payload)

    index_rows = [row for row in rows if row.resume_kind == "control_zero_at_resume"]
    continuous_rows = [row for row in rows if row.resume_kind == "control_clamped_zero"]
    inverse_rows = [
        row
        for row in rows
        if row.resume_kind in {"slow_carrier_zero_control_preserved", "control_only"}
    ]
    return {
        "experiment": "control_capsule_maintenance",
        "row_count": len(rows),
        "candidate_count": len({row.candidate_id for row in rows}),
        "by_candidate_resume": by_candidate_resume,
        "aggregate": {
            "control_zero_index_mean": mean_optional(
                [row.control_maintenance_index for row in index_rows]
            ),
            "control_clamped_zero_index_mean": mean_optional(
                [row.continuous_control_maintenance_index for row in continuous_rows]
            ),
            "inverse_pullback_positive_fraction": positive_fraction(
                [row.capsule_pullback for row in inverse_rows]
            ),
        },
        "interpretation_boundary": (
            "Evidence is limited to control-mediated capsule maintenance; do not promote selfhood claims."
        ),
    }


def mean_optional(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def positive_fraction(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(value > 0.0 for value in finite)) / float(len(finite))


def candidate_paths_from_replication_summary(path: Path) -> list[Path]:
    payload = load_json(path)
    rows = payload.get("rows", [])
    paths = [Path(row["top_candidate_path"]) for row in rows if row.get("top_candidate_path")]
    if not paths:
        raise ValueError(f"no top_candidate_path values found in {path}")
    return paths


def run_control_capsule_characterization(config: ControlCapsuleConfig) -> dict[str, Any]:
    candidate_paths = config.candidate_paths or candidate_paths_from_replication_summary(config.replication_summary_path)
    rows: list[ControlCapsuleRow] = []
    torch.set_grad_enabled(False)
    for candidate_path in candidate_paths:
        rows.extend(run_candidate_control_capsule_probe(candidate_path, config))
    payload = summarize_rows(rows)
    payload["config"] = {
        **config.model_dump(mode="json"),
        "candidate_paths": [path.as_posix() for path in candidate_paths],
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.output_dir / "control_capsule_rows.jsonl", rows)
    write_csv(config.output_dir / "control_capsule_summary.csv", payload["by_candidate_resume"])
    write_json(config.output_dir / "control_capsule_summary.json", payload)
    write_readme(config.output_dir, payload)
    return payload


def write_readme(output_dir: Path, payload: dict[str, Any]) -> None:
    text = f"""# Control Capsule Maintenance

Control-ablation capsule probe for Track B candidates.

Artifacts:

- `control_capsule_rows.jsonl`: per-run resume arm rows
- `control_capsule_summary.csv`: per-candidate, per-resume aggregate metrics
- `control_capsule_summary.json`: aggregate interpretation metrics and config

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
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--perturb-step", type=int, default=64)
    parser.add_argument("--capsule-pause-step", type=int, default=64)
    parser.add_argument("--tail-window", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ControlCapsuleConfig(
        candidate_paths=parse_paths(args.candidate_paths),
        replication_summary_path=args.replication_summary_path,
        output_dir=args.output_dir,
        seeds=parse_csv_ints(args.seeds),
        perturb_scales=parse_csv_floats(args.perturb_scales),
        motifs=parse_motifs(args.motifs),
        hidden_size=int(args.hidden_size),
        rank=int(args.rank),
        steps=int(args.steps),
        perturb_step=int(args.perturb_step),
        capsule_pause_step=int(args.capsule_pause_step),
        tail_window=int(args.tail_window),
        device=str(args.device),
    )
    payload = run_control_capsule_characterization(config)
    print(f"wrote {config.output_dir / 'control_capsule_summary.json'}")
    print(f"rows={payload['row_count']}")


if __name__ == "__main__":
    main()
