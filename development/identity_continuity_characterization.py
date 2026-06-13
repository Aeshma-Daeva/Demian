#!/usr/bin/env python3
"""Identity-continuity, boundary, and self-maintenance probes for Track B."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator, model_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolution.config import CHANNELS
from development.gate_state_propagation_characterization import (
    CharacterizationConfig,
    MotifSpec,
    apply_controlled_perturbation,
    build_model,
    clamp_channel_state,
    load_json,
    motif_from_spec,
    normalized_l2,
    write_csv,
    write_json,
    write_jsonl,
)

IdentityCondition = Literal[
    "identity_preserved",
    "identity_reset",
    "control_disabled",
    "slow_disabled",
    "carrier_disabled",
    "identity_scrambled",
]
BoundarySource = Literal["internal", "external"]
LesionCondition = Literal["no_lesion", "control_zero", "slow_zero", "carrier_zero", "identity_scramble"]

IDENTITY_CHANNELS: tuple[str, ...] = ("slow", "control", "carrier")
DEFAULT_OUT_DIR = Path("data/diagnostics/identity_continuity_20260513")


class FlexibleSchema(BaseModel):
    """Validate expected fields while preserving searchable extras."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class IdentityProbeConfig(FlexibleSchema):
    candidate_path: Path | None = None
    archive_root: Path | None = None
    output_dir: Path = DEFAULT_OUT_DIR
    seeds: list[int] = Field(default_factory=lambda: list(range(94, 103)))
    perturb_scales: list[float] = Field(default_factory=lambda: [0.2, 0.35, 0.7])
    motifs: list[MotifSpec] = Field(
        default_factory=lambda: [MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)]
    )
    hidden_size: NonNegativeInt = 32
    rank: NonNegativeInt = 2
    steps: NonNegativeInt = 128
    burn_in_steps: NonNegativeInt = 32
    perturb_step: NonNegativeInt = 64
    lesion_step: NonNegativeInt = 64
    tail_window: NonNegativeInt = 16
    perturb_channel: str = "fast"
    device: str = "cpu"

    @model_validator(mode="after")
    def step_order_is_valid(self) -> "IdentityProbeConfig":
        if self.burn_in_steps >= self.steps:
            raise ValueError("burn_in_steps must be < steps")
        if self.perturb_step > self.steps:
            raise ValueError("perturb_step must be <= steps")
        if self.lesion_step > self.steps:
            raise ValueError("lesion_step must be <= steps")
        if self.tail_window < 1:
            raise ValueError("tail_window must be >= 1")
        return self

    def as_characterization_config(self) -> CharacterizationConfig:
        return CharacterizationConfig(
            candidate_path=self.candidate_path or CharacterizationConfig().candidate_path,
            archive_root=self.archive_root or CharacterizationConfig().archive_root,
            output_dir=self.output_dir,
            seeds=self.seeds,
            perturb_scales=self.perturb_scales,
            motifs=self.motifs,
            hidden_size=int(self.hidden_size),
            rank=int(self.rank),
            steps=int(self.steps),
            perturb_step=int(self.perturb_step),
            capsule_pause_step=int(self.perturb_step),
            perturb_channel=self.perturb_channel,
            device=self.device,
        )


class IdentitySignature(FlexibleSchema):
    channels: list[str]
    mean_vector: list[float]
    channel_norms: dict[str, float]
    channel_pair_cosines: dict[str, float]

    @field_validator("mean_vector")
    @classmethod
    def vector_is_finite(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("identity signature vector must be non-empty")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("identity signature vector must be finite")
        return values


class IdentityTraceRow(FlexibleSchema):
    probe: str
    candidate_id: str
    condition: str
    seed: int
    perturb_scale: float
    motif_family: str
    motif_index: int
    step: NonNegativeInt
    source: str | None = None
    identity_distance: float
    identity_cosine: float
    fast_state_norm: float
    slow_state_norm: float
    control_state_norm: float
    message_state_norm: float
    carrier_state_norm: float


class ProbeSummaryRow(FlexibleSchema):
    probe: str
    candidate_id: str
    condition: str
    seed: int
    perturb_scale: float
    motif_family: str
    motif_index: int
    source: str | None = None
    tail_identity_distance: float
    tail_identity_cosine: float
    worst_identity_distance: float
    final_identity_distance: float
    final_identity_cosine: float


def tensor_to_vector(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.view(-1).detach().cpu().tolist()]


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel():
        width = min(left.numel(), right.numel())
        left = left.view(-1)[:width]
        right = right.view(-1)[:width]
    denom = float(left.norm() * right.norm())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(left.view(-1), right.view(-1)) / denom)


def mean_tensor(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("cannot average an empty tensor list")
    return torch.stack([value.view(-1).detach().float().cpu() for value in values]).mean(dim=0)


def state_components_cpu(model: Any, state: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    return {
        channel: tensor.view(-1).detach().float().cpu()
        for channel, tensor in model.state_components(state).items()
    }


def identity_vector_from_components(components: dict[str, torch.Tensor], channels: tuple[str, ...] = IDENTITY_CHANNELS) -> torch.Tensor:
    return torch.cat([components[channel].view(-1).detach().float().cpu() for channel in channels])


def fit_vector_width(vector: torch.Tensor, width: int) -> torch.Tensor:
    values = vector.view(-1).detach().float().cpu()
    if values.numel() == width:
        return values
    if values.numel() > width:
        return values[:width]
    return torch.nn.functional.pad(values, (0, width - values.numel()))


def identity_signature_from_window(
    component_window: list[dict[str, torch.Tensor]],
    *,
    channels: tuple[str, ...] = IDENTITY_CHANNELS,
) -> IdentitySignature:
    """Derive an endogenous identity signature from a burn-in component window."""

    if not component_window:
        raise ValueError("component_window must not be empty")
    channel_means = {
        channel: mean_tensor([components[channel] for components in component_window])
        for channel in channels
    }
    mean_vector = torch.cat([channel_means[channel] for channel in channels])
    pair_cosines = {}
    for left_index, left in enumerate(channels):
        for right in channels[left_index + 1:]:
            pair_cosines[f"{left}:{right}"] = cosine_similarity(channel_means[left], channel_means[right])
    return IdentitySignature(
        channels=list(channels),
        mean_vector=tensor_to_vector(mean_vector),
        channel_norms={
            channel: float(channel_means[channel].norm()) / math.sqrt(max(channel_means[channel].numel(), 1))
            for channel in channels
        },
        channel_pair_cosines=pair_cosines,
    )


def identity_metrics(components: dict[str, torch.Tensor], signature: IdentitySignature) -> dict[str, float]:
    current = identity_vector_from_components(components, channels=tuple(signature.channels))
    target = torch.tensor(signature.mean_vector, dtype=torch.float32)
    return {
        "identity_distance": normalized_l2(current, target),
        "identity_cosine": cosine_similarity(current, target),
    }


def channel_norms(components: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        f"{channel}_state_norm": float(components[channel].norm()) / math.sqrt(max(components[channel].numel(), 1))
        for channel in CHANNELS
    }


def trace_row(
    *,
    probe: str,
    candidate_id: str,
    condition: str,
    seed: int,
    perturb_scale: float,
    motif: dict[str, Any],
    step: int,
    components: dict[str, torch.Tensor],
    signature: IdentitySignature,
    source: str | None = None,
) -> IdentityTraceRow:
    return IdentityTraceRow(
        probe=probe,
        candidate_id=candidate_id,
        condition=condition,
        seed=int(seed),
        perturb_scale=float(perturb_scale),
        motif_family=str(motif["family"]),
        motif_index=int(motif["index"]),
        step=int(step),
        source=source,
        **identity_metrics(components, signature),
        **channel_norms(components),
    )


def summarize_rows(rows: list[IdentityTraceRow], *, tail_window: int) -> list[ProbeSummaryRow]:
    grouped: dict[tuple[str, str, str, int, float, str, int, str | None], list[IdentityTraceRow]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.probe,
                row.candidate_id,
                row.condition,
                row.seed,
                row.perturb_scale,
                row.motif_family,
                row.motif_index,
                row.source,
            )
        ].append(row)
    summaries: list[ProbeSummaryRow] = []
    for key, group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: row.step)
        tail = ordered[-max(1, min(int(tail_window), len(ordered))):]
        summaries.append(
            ProbeSummaryRow(
                probe=key[0],
                candidate_id=key[1],
                condition=key[2],
                seed=key[3],
                perturb_scale=key[4],
                motif_family=key[5],
                motif_index=key[6],
                source=key[7],
                tail_identity_distance=float(mean(row.identity_distance for row in tail)),
                tail_identity_cosine=float(mean(row.identity_cosine for row in tail)),
                worst_identity_distance=max(row.identity_distance for row in ordered),
                final_identity_distance=ordered[-1].identity_distance,
                final_identity_cosine=ordered[-1].identity_cosine,
            )
        )
    return summaries


def reset_identity_channels(
    state: tuple[torch.Tensor, ...],
    reset_state: tuple[torch.Tensor, ...],
    *,
    channels: tuple[str, ...] = IDENTITY_CHANNELS,
) -> tuple[torch.Tensor, ...]:
    values = list(state)
    for channel in channels:
        idx = list(CHANNELS).index(channel)
        values[idx] = reset_state[idx].detach().clone()
    return tuple(values)


def scramble_identity_channels(
    state: tuple[torch.Tensor, ...],
    *,
    channels: tuple[str, ...] = IDENTITY_CHANNELS,
) -> tuple[torch.Tensor, ...]:
    values = list(state)
    for channel in channels:
        idx = list(CHANNELS).index(channel)
        values[idx] = -torch.flip(values[idx], dims=(-1,))
    return tuple(values)


def run_identity_lifetime_probe(
    genome: dict[str, Any],
    *,
    candidate_id: str,
    config: IdentityProbeConfig,
    condition: IdentityCondition,
    seed: int,
    perturb_scale: float,
    motif: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    characterization_config = config.as_characterization_config()
    model = build_model(genome, config=characterization_config, condition="original")
    state = model.initial_state(1, torch.device(config.device))
    reset_state = tuple(item.detach().clone() for item in state)
    burn_in_components: list[dict[str, torch.Tensor]] = []
    rows: list[IdentityTraceRow] = []
    signature: IdentitySignature | None = None
    with torch.no_grad():
        for step in range(1, int(config.steps) + 1):
            if step == int(config.perturb_step):
                if condition == "identity_reset":
                    state = reset_identity_channels(state, reset_state)
                if condition == "identity_scrambled":
                    state = scramble_identity_channels(state)
                state = apply_controlled_perturbation(
                    state,
                    motif["vector"],
                    float(perturb_scale),
                    channel=config.perturb_channel,
                )
                model.record_perturbation(step, float(perturb_scale))
            state = model.step(state)
            if condition in {"control_disabled", "slow_disabled", "carrier_disabled"}:
                state = clamp_channel_state(state, condition.removesuffix("_disabled"))
            components = state_components_cpu(model, state)
            if step <= int(config.burn_in_steps):
                burn_in_components.append(components)
                if step == int(config.burn_in_steps):
                    signature = identity_signature_from_window(burn_in_components)
            elif signature is not None:
                rows.append(
                    trace_row(
                        probe="lifetime",
                        candidate_id=candidate_id,
                        condition=condition,
                        seed=seed,
                        perturb_scale=perturb_scale,
                        motif=motif,
                        step=step,
                        components=components,
                        signature=signature,
                    )
                )
    if signature is None:
        raise RuntimeError("identity signature was not captured")
    return {
        "probe": "lifetime",
        "condition": condition,
        "signature": signature,
        "rows": rows,
        "summary": summarize_rows(rows, tail_window=int(config.tail_window)),
    }


def internal_source_vector(components: dict[str, torch.Tensor], hidden_size: int) -> torch.Tensor:
    source = (
        fit_vector_width(components["slow"], hidden_size)
        + fit_vector_width(components["control"], hidden_size)
        + fit_vector_width(components["carrier"], hidden_size)
    )
    norm = float(source.norm())
    if norm <= 1e-12:
        return torch.nn.functional.normalize(torch.ones(hidden_size, dtype=torch.float32), dim=0)
    return source / norm


def run_boundary_source_probe(
    genome: dict[str, Any],
    *,
    candidate_id: str,
    config: IdentityProbeConfig,
    source: BoundarySource,
    seed: int,
    perturb_scale: float,
    motif: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    characterization_config = config.as_characterization_config()
    model = build_model(genome, config=characterization_config, condition="original")
    state = model.initial_state(1, torch.device(config.device))
    burn_in_components: list[dict[str, torch.Tensor]] = []
    rows: list[IdentityTraceRow] = []
    signature: IdentitySignature | None = None
    with torch.no_grad():
        for step in range(1, int(config.steps) + 1):
            components_before = state_components_cpu(model, state)
            if step == int(config.perturb_step):
                perturb_vector = (
                    internal_source_vector(components_before, int(config.hidden_size))
                    if source == "internal"
                    else motif["vector"]
                )
                state = apply_controlled_perturbation(
                    state,
                    perturb_vector,
                    float(perturb_scale),
                    channel=config.perturb_channel,
                )
                model.record_perturbation(step, float(perturb_scale))
            state = model.step(state)
            components = state_components_cpu(model, state)
            if step <= int(config.burn_in_steps):
                burn_in_components.append(components)
                if step == int(config.burn_in_steps):
                    signature = identity_signature_from_window(burn_in_components)
            elif signature is not None:
                rows.append(
                    trace_row(
                        probe="boundary",
                        candidate_id=candidate_id,
                        condition="source_probe",
                        seed=seed,
                        perturb_scale=perturb_scale,
                        motif=motif,
                        step=step,
                        components=components,
                        signature=signature,
                        source=source,
                    )
                )
    if signature is None:
        raise RuntimeError("identity signature was not captured")
    return {
        "probe": "boundary",
        "source": source,
        "signature": signature,
        "rows": rows,
        "summary": summarize_rows(rows, tail_window=int(config.tail_window)),
    }


def apply_lesion(state: tuple[torch.Tensor, ...], condition: LesionCondition) -> tuple[torch.Tensor, ...]:
    if condition == "no_lesion":
        return state
    if condition == "identity_scramble":
        return scramble_identity_channels(state)
    return clamp_channel_state(state, condition.removesuffix("_zero"))


def run_self_maintenance_lesion_probe(
    genome: dict[str, Any],
    *,
    candidate_id: str,
    config: IdentityProbeConfig,
    condition: LesionCondition,
    seed: int,
    perturb_scale: float,
    motif: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    characterization_config = config.as_characterization_config()
    model = build_model(genome, config=characterization_config, condition="original")
    state = model.initial_state(1, torch.device(config.device))
    burn_in_components: list[dict[str, torch.Tensor]] = []
    rows: list[IdentityTraceRow] = []
    signature: IdentitySignature | None = None
    with torch.no_grad():
        for step in range(1, int(config.steps) + 1):
            if step == int(config.lesion_step):
                state = apply_lesion(state, condition)
                state = apply_controlled_perturbation(
                    state,
                    motif["vector"],
                    float(perturb_scale),
                    channel=config.perturb_channel,
                )
                model.record_perturbation(step, float(perturb_scale))
            state = model.step(state)
            components = state_components_cpu(model, state)
            if step <= int(config.burn_in_steps):
                burn_in_components.append(components)
                if step == int(config.burn_in_steps):
                    signature = identity_signature_from_window(burn_in_components)
            elif signature is not None:
                rows.append(
                    trace_row(
                        probe="lesion",
                        candidate_id=candidate_id,
                        condition=condition,
                        seed=seed,
                        perturb_scale=perturb_scale,
                        motif=motif,
                        step=step,
                        components=components,
                        signature=signature,
                    )
                )
    if signature is None:
        raise RuntimeError("identity signature was not captured")
    return {
        "probe": "lesion",
        "condition": condition,
        "signature": signature,
        "rows": rows,
        "summary": summarize_rows(rows, tail_window=int(config.tail_window)),
    }


def summary_dict(rows: list[ProbeSummaryRow]) -> list[dict[str, Any]]:
    return [row.model_dump(mode="json") for row in rows]


def run_identity_characterization(
    genome: dict[str, Any],
    *,
    candidate_id: str,
    config: IdentityProbeConfig,
    probes: tuple[str, ...] = ("lifetime", "boundary", "lesion"),
) -> dict[str, Any]:
    rows: list[IdentityTraceRow] = []
    signatures: list[dict[str, Any]] = []
    for seed in config.seeds:
        motifs = [motif_from_spec(spec, int(config.hidden_size), int(seed)) for spec in config.motifs]
        for perturb_scale in config.perturb_scales:
            for motif in motifs:
                if "lifetime" in probes:
                    for condition in (
                        "identity_preserved",
                        "identity_reset",
                        "control_disabled",
                        "slow_disabled",
                        "carrier_disabled",
                        "identity_scrambled",
                    ):
                        result = run_identity_lifetime_probe(
                            genome,
                            candidate_id=candidate_id,
                            config=config,
                            condition=condition,  # type: ignore[arg-type]
                            seed=seed,
                            perturb_scale=perturb_scale,
                            motif=motif,
                        )
                        rows.extend(result["rows"])
                        signatures.append(result["signature"].model_dump(mode="json"))
                if "boundary" in probes:
                    for source in ("internal", "external"):
                        result = run_boundary_source_probe(
                            genome,
                            candidate_id=candidate_id,
                            config=config,
                            source=source,  # type: ignore[arg-type]
                            seed=seed,
                            perturb_scale=perturb_scale,
                            motif=motif,
                        )
                        rows.extend(result["rows"])
                        signatures.append(result["signature"].model_dump(mode="json"))
                if "lesion" in probes:
                    for condition in ("no_lesion", "control_zero", "slow_zero", "carrier_zero", "identity_scramble"):
                        result = run_self_maintenance_lesion_probe(
                            genome,
                            candidate_id=candidate_id,
                            config=config,
                            condition=condition,  # type: ignore[arg-type]
                            seed=seed,
                            perturb_scale=perturb_scale,
                            motif=motif,
                        )
                        rows.extend(result["rows"])
                        signatures.append(result["signature"].model_dump(mode="json"))
    summaries = summarize_rows(rows, tail_window=int(config.tail_window))
    payload = {
        "experiment": "identity_continuity_characterization",
        "candidate_id": candidate_id,
        "config": config.model_dump(mode="json"),
        "row_count": len(rows),
        "summary": summary_dict(summaries),
        "signature_count": len(signatures),
        "interpretation_boundary": (
            "Instrumentation artifact only. Promote claims only after comparative or seed-sweep review."
        ),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.output_dir / "identity_rows.jsonl", rows)
    write_json(config.output_dir / "identity_summary.json", payload)
    write_json(config.output_dir / "identity_signatures.json", signatures)
    write_csv(config.output_dir / "identity_summary.csv", summary_dict(summaries))
    return payload


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--probes", default="lifetime,boundary,lesion")
    parser.add_argument("--seeds", default="94,95,96,97,98,99,100,101,102")
    parser.add_argument("--perturb-scales", default="0.2,0.35,0.7")
    parser.add_argument("--motifs", default="basis:0,gaussian:0")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--burn-in-steps", type=int, default=32)
    parser.add_argument("--perturb-step", type=int, default=64)
    parser.add_argument("--lesion-step", type=int, default=64)
    parser.add_argument("--tail-window", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    candidate = load_json(args.candidate_path)
    candidate_id = str(candidate.get("id", args.candidate_path.stem))
    config = IdentityProbeConfig(
        candidate_path=args.candidate_path,
        archive_root=args.candidate_path.parent.parent,
        output_dir=args.output_dir,
        seeds=parse_csv_ints(args.seeds),
        perturb_scales=parse_csv_floats(args.perturb_scales),
        motifs=parse_motifs(args.motifs),
        hidden_size=int(args.hidden_size),
        rank=int(args.rank),
        steps=int(args.steps),
        burn_in_steps=int(args.burn_in_steps),
        perturb_step=int(args.perturb_step),
        lesion_step=int(args.lesion_step),
        tail_window=int(args.tail_window),
        device=str(args.device),
    )
    payload = run_identity_characterization(
        candidate["genome"],
        candidate_id=candidate_id,
        config=config,
        probes=tuple(item.strip() for item in str(args.probes).split(",") if item.strip()),
    )
    print(f"wrote {config.output_dir / 'identity_summary.json'}")
    print(f"rows={payload['row_count']}")


if __name__ == "__main__":
    main()
