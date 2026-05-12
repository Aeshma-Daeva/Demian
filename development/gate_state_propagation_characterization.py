#!/usr/bin/env python3
"""Characterize gate-state propagation in a Track B native candidate."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator, model_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legacy.demian_runtime.machine_observables import classify_attractor, compute_observables
from development.evolution.config import CHANNELS
from development.evolution.scoring import NATIVE_EMERGENCE_RANK_MODE, rank_components, scalar_rank
from development.evolve_v9_5ch_release import (
    AdaptiveV9FiveChannel,
    apply_signature,
    component_delta,
    cosine_to_signature,
    gain_zero_diagnostics,
    motif_vector,
    path_metrics,
    release_anchor_steps,
    release_gain_zero_genome,
)

Condition = Literal[
    "original",
    "routes_disabled",
    "gain_zero",
    "message_disabled",
    "carrier_disabled",
    "control_disabled",
    "slow_disabled",
]

ResumeKind = Literal[
    "uninterrupted",
    "full_internal_state",
    "surface_only",
    "fast_only",
    "slow_only",
    "control_only",
    "message_only",
    "carrier_only",
]

DEFAULT_CANDIDATE = Path("data/evolution/demian_v2_track_b_island_1_20260511/candidates/gen012_candidate000.json")
DEFAULT_ARCHIVE_ROOT = Path("data/evolution/demian_v2_track_b_island_1_20260511")
DEFAULT_OUT_DIR = Path("data/diagnostics/gate_state_propagation_characterization_20260511")
PARAMETER_SIGNATURE_CSV = Path("data/diagnostics/gate_state_parameter_signatures.csv")
PARAMETER_SIGNATURE_PARQUET = Path("data/diagnostics/gate_state_parameter_signatures.parquet")
PARAMETER_SIGNATURE_SUMMARY = Path("data/diagnostics/gate_state_parameter_signatures_summary.json")
DEFAULT_CONDITIONS: tuple[Condition, ...] = (
    "original",
    "routes_disabled",
    "gain_zero",
    "message_disabled",
    "carrier_disabled",
    "control_disabled",
    "slow_disabled",
)


class FlexibleSchema(BaseModel):
    """Validate expected fields while preserving searchable extras."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class MotifSpec(FlexibleSchema):
    family: Literal["basis", "gaussian"]
    index: NonNegativeInt = 0


class CharacterizationConfig(FlexibleSchema):
    candidate_path: Path = DEFAULT_CANDIDATE
    archive_root: Path = DEFAULT_ARCHIVE_ROOT
    output_dir: Path = DEFAULT_OUT_DIR
    seeds: list[int] = Field(default_factory=lambda: list(range(94, 103)))
    perturb_scales: list[float] = Field(default_factory=lambda: [0.2, 0.35, 0.7])
    motifs: list[MotifSpec] = Field(
        default_factory=lambda: [MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)]
    )
    steps: NonNegativeInt = 128
    perturb_step: NonNegativeInt = 64
    hidden_size: NonNegativeInt = 32
    rank: NonNegativeInt = 2
    perturb_channel: str = "fast"
    perturb_mode: str = "external"
    conditions: list[Condition] = Field(default_factory=lambda: list(DEFAULT_CONDITIONS))
    device: str = "cpu"
    anchor_offset: NonNegativeInt = 4
    capsule_pause_step: NonNegativeInt = 64
    write_parameter_signatures: bool = True

    @model_validator(mode="after")
    def steps_are_consistent(self) -> "CharacterizationConfig":
        if self.perturb_step > self.steps:
            raise ValueError("perturb_step must be <= steps")
        if self.capsule_pause_step > self.steps:
            raise ValueError("capsule_pause_step must be <= steps")
        return self


class AblationRow(FlexibleSchema):
    candidate_id: str
    condition: Condition
    seed: int
    perturb_scale: float
    motif_family: str
    motif_index: int
    step: int
    residual_norm: float
    residual_delta: float
    temporal_coherence: float
    release_strength_mean: float
    release_open_mean: float
    release_pressure_mean: float
    fast_state_norm: float
    slow_state_norm: float
    control_state_norm: float
    message_state_norm: float
    carrier_state_norm: float

    @field_validator("*")
    @classmethod
    def numeric_values_are_finite(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric ablation row fields must be finite")
        return value


class CapsuleRow(FlexibleSchema):
    candidate_id: str
    seed: int
    perturb_scale: float
    motif_family: str
    motif_index: int
    resume_kind: ResumeKind
    final_cosine: float
    final_l2_gap: float
    mean_step_gap: float
    fast_gap: float
    slow_gap: float
    control_gap: float
    message_gap: float
    carrier_gap: float


class ParameterSignatureRow(FlexibleSchema):
    selection: str
    candidate_id: str
    candidate_path: str
    native_rank: float
    generation: int | None = None
    reproduction_kind: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    ancestor_ids: list[str] = Field(default_factory=list)
    lineage_key: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    scalars: dict[str, float] = Field(default_factory=dict)


class SummaryPayload(FlexibleSchema):
    experiment: str
    candidate_id: str
    config: dict[str, Any]
    run_count: int
    ablation_summary: list[dict[str, Any]]
    capsule_summary: list[dict[str, Any]]
    parameter_signature_summary_path: str
    evidence_gate: dict[str, Any]
    held_out_seeds: list[int] = Field(default_factory=list)
    perturb_steps: list[int] = Field(default_factory=list)
    channel_necessity_order: list[str] = Field(default_factory=list)
    gain_zero_cleanliness: dict[str, Any] = Field(default_factory=dict)
    capsule_result_flags: dict[str, Any] = Field(default_factory=dict)
    probe_summary: dict[str, Any] = Field(default_factory=dict)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean_or_none(values: Iterable[float | None]) -> float | None:
    rows = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(rows)) if rows else None


def normalized_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.norm(left - right)) / math.sqrt(max(left.numel(), 1))


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    scale = 0.5 * (float(torch.norm(left)) + float(torch.norm(right))) / math.sqrt(max(left.numel(), 1))
    return normalized_l2(left, right) / (scale + 1e-8)


def clone_state(state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(item.detach().clone() for item in state)


def zero_like_state(state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros_like(item) for item in state)


def clamp_channel_state(
    state: tuple[torch.Tensor, ...],
    channel: str,
) -> tuple[torch.Tensor, ...]:
    """Return a state with exactly one requested channel zeroed."""

    if channel not in CHANNELS:
        raise ValueError(f"unknown channel: {channel}")
    values = list(state)
    values[list(CHANNELS).index(channel)] = torch.zeros_like(values[list(CHANNELS).index(channel)])
    return tuple(values)


def condition_to_channel(condition: Condition) -> str | None:
    if condition.endswith("_disabled") and condition not in {"routes_disabled"}:
        return condition.removesuffix("_disabled")
    return None


def motif_from_spec(spec: MotifSpec, hidden_size: int, seed: int) -> dict[str, Any]:
    return {
        "family": spec.family,
        "index": int(spec.index),
        "vector": motif_vector(spec.family, hidden_size, seed, int(spec.index)),
    }


def apply_controlled_perturbation(
    state: tuple[torch.Tensor, ...],
    signature: torch.Tensor,
    scale: float,
    *,
    channel: str,
) -> tuple[torch.Tensor, ...]:
    return apply_signature(state, signature, scale, channel=channel, mode="external")


def build_model(
    genome: dict[str, Any],
    *,
    config: CharacterizationConfig,
    condition: Condition = "original",
) -> AdaptiveV9FiveChannel:
    model_genome = release_gain_zero_genome(genome) if condition == "gain_zero" else genome
    model = AdaptiveV9FiveChannel(
        int(config.hidden_size),
        model_genome,
        rank=int(config.rank),
        release_routes_disabled=condition == "routes_disabled",
    )
    return model.to(device=config.device)


def row_from_step(
    *,
    candidate_id: str,
    condition: Condition,
    seed: int,
    perturb_scale: float,
    motif: dict[str, Any],
    step: int,
    surface: torch.Tensor,
    prev_surface: torch.Tensor | None,
    components: dict[str, torch.Tensor],
    prev_components: dict[str, torch.Tensor] | None,
    aux: dict[str, Any],
    signature: torch.Tensor,
) -> tuple[AblationRow, dict[str, Any]]:
    if prev_surface is None:
        residual_delta = 0.0
        temporal_coherence = 1.0
    else:
        residual_delta = normalized_l2(surface, prev_surface)
        temporal_coherence = float(torch.nn.functional.cosine_similarity(surface, prev_surface, dim=0))
    fast = components["fast"]
    slow = components["slow"]
    control = components["control"]
    message = components["message"]
    carrier = components["carrier"]
    route_metrics = dict(aux)
    metrics_row = {
        "step": step,
        "residual_norm": float(surface.norm()) / math.sqrt(max(surface.numel(), 1)),
        "residual_delta": residual_delta,
        "temporal_coherence": temporal_coherence,
        "velocity_align": 1.0,
        "spectral_centroid": 0.0,
        "spectral_concentration": 0.0,
        "layer_work_ratio": 0.5,
        "fast_state_norm": float(fast.norm()) / math.sqrt(max(fast.numel(), 1)),
        "slow_state_norm": float(slow.norm()) / math.sqrt(max(slow.numel(), 1)),
        "control_state_norm": float(control.norm()) / math.sqrt(max(control.numel(), 1)),
        "message_state_norm": float(message.norm()) / math.sqrt(max(message.numel(), 1)),
        "carrier_state_norm": float(carrier.norm()) / math.sqrt(max(carrier.numel(), 1)),
        "message_signature_cosine": cosine_to_signature(message, signature),
        "carrier_signature_cosine": cosine_to_signature(carrier, signature),
        "fast_signature_cosine": cosine_to_signature(fast, signature),
        "route_metrics": route_metrics,
        "_channel_states": {name: tensor.detach().clone().cpu() for name, tensor in components.items()},
    }
    for name in CHANNELS:
        metrics_row[f"{name}_state_delta"] = component_delta(components[name], None if prev_components is None else prev_components[name])
    output_row = AblationRow(
        candidate_id=candidate_id,
        condition=condition,
        seed=seed,
        perturb_scale=float(perturb_scale),
        motif_family=str(motif["family"]),
        motif_index=int(motif["index"]),
        step=step,
        residual_norm=float(metrics_row["residual_norm"]),
        residual_delta=float(metrics_row["residual_delta"]),
        temporal_coherence=float(metrics_row["temporal_coherence"]),
        release_strength_mean=float(route_metrics.get("release_strength_mean", 0.0)),
        release_open_mean=float(route_metrics.get("release_open_mean", 0.0)),
        release_pressure_mean=float(route_metrics.get("release_pressure_mean", 0.0)),
        fast_state_norm=float(metrics_row["fast_state_norm"]),
        slow_state_norm=float(metrics_row["slow_state_norm"]),
        control_state_norm=float(metrics_row["control_state_norm"]),
        message_state_norm=float(metrics_row["message_state_norm"]),
        carrier_state_norm=float(metrics_row["carrier_state_norm"]),
    )
    return output_row, metrics_row


def run_condition(
    genome: dict[str, Any],
    *,
    candidate_id: str,
    config: CharacterizationConfig,
    condition: Condition,
    seed: int,
    perturb_scale: float,
    motif: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(genome, config=config, condition=condition)
    state = model.initial_state(1, torch.device(config.device))
    disabled_channel = condition_to_channel(condition)
    signature = motif["vector"]
    rows: list[AblationRow] = []
    trajectory: list[dict[str, Any]] = []
    surfaces: list[torch.Tensor] = []
    fasts: list[torch.Tensor] = []
    residual_buffer: list[torch.Tensor] = []
    prev_surface: torch.Tensor | None = None
    prev_components: dict[str, torch.Tensor] | None = None
    window: list[dict[str, Any]] = []
    with torch.no_grad():
        for step in range(1, int(config.steps) + 1):
            if step == int(config.perturb_step):
                state = apply_controlled_perturbation(
                    state,
                    signature,
                    float(perturb_scale),
                    channel=config.perturb_channel,
                )
                model.record_perturbation(step, float(perturb_scale))
            state = model.step(state)
            if disabled_channel is not None:
                state = clamp_channel_state(state, disabled_channel)
            surface = model.state_vector(state).view(-1).detach().float().cpu()
            components = {
                name: tensor.view(-1).detach().float().cpu()
                for name, tensor in model.state_components(state).items()
            }
            output_row, metrics_row = row_from_step(
                candidate_id=candidate_id,
                condition=condition,
                seed=seed,
                perturb_scale=perturb_scale,
                motif=motif,
                step=step,
                surface=surface,
                prev_surface=prev_surface,
                components=components,
                prev_components=prev_components,
                aux=model.step_aux(),
                signature=signature,
            )
            rows.append(output_row)
            trajectory.append(metrics_row)
            surfaces.append(surface.clone())
            fasts.append(components["fast"].clone())
            window.append(metrics_row)
            if len(window) > 20:
                window.pop(0)
            residual_buffer.append(surface.clone())
            if len(residual_buffer) > 16:
                residual_buffer.pop(0)
            prev_surface = surface.clone()
            prev_components = {name: tensor.clone() for name, tensor in components.items()}
    obs = compute_observables(window, surfaces[-1], residual_buffer)
    attractor = classify_attractor(obs)
    metrics = path_metrics(trajectory, surfaces, fasts, signature, perturb_step=int(config.perturb_step))
    summary = {
        "attractor_type": attractor.type,
        "regime_class": classify_regime_for_run(attractor.type, trajectory, metrics),
        "covariance_rank": float(obs["covariance_rank"]),
        "flow_dimension": float(obs["flow_dimension"]),
        "mean_norm": float(mean(row.residual_norm for row in rows)),
        "mean_delta": float(mean(row.residual_delta for row in rows)),
    }
    if condition == "gain_zero":
        metrics.update(gain_zero_diagnostics({"trajectory": trajectory}))
    return {
        "condition": condition,
        "seed": seed,
        "perturb_scale": perturb_scale,
        "motif_family": motif["family"],
        "motif_index": motif["index"],
        "summary": summary,
        "metrics": metrics,
        "trajectory": trajectory,
        "rows": rows,
    }


def classify_regime_for_run(attractor_type: str, trajectory: list[dict[str, Any]], metrics: dict[str, float]) -> str:
    max_norm = max(float(row["residual_norm"]) for row in trajectory)
    if not math.isfinite(max_norm) or max_norm > 40.0:
        return "unbounded_expanding"
    if metrics["channel_separation"] < 0.04 and metrics["internal_richness"] < 0.12:
        return "collapsed_incoherent"
    if attractor_type == "FIXED_POINT":
        if (
            metrics["internal_richness"] >= 0.22
            or metrics["phase_transition_score"] >= 0.012
            or metrics["release_geometric_event"] >= 0.002
        ):
            return "surface_fixed_accumulating"
        return "surface_fixed_internal_quiet"
    if attractor_type == "CHAOTIC":
        return "bounded_chaotic_structured"
    if attractor_type == "STRANGE":
        return "bounded_strange"
    return str(attractor_type).lower()


def causal_divergence_against_original(
    original: dict[str, Any],
    condition_run: dict[str, Any],
    *,
    anchor_offset: int,
) -> dict[str, float]:
    original_rows = {int(row["step"]): row for row in original["trajectory"]}
    condition_rows = {int(row["step"]): row for row in condition_run["trajectory"]}
    values_by_channel: dict[str, list[float]] = {channel: [] for channel in CHANNELS}
    for step in release_anchor_steps(original):
        target_step = step + anchor_offset
        left = original_rows.get(target_step)
        right = condition_rows.get(target_step)
        if left is None or right is None:
            continue
        for channel in CHANNELS:
            left_tensor = left["_channel_states"][channel]
            right_tensor = right["_channel_states"][channel]
            values_by_channel[channel].append(relative_l2(left_tensor, right_tensor))
    return {
        "causal_anchor_count": float(max((len(values) for values in values_by_channel.values()), default=0)),
        "causal_divergence": float(mean([mean(v) for v in values_by_channel.values() if v])) if any(values_by_channel.values()) else 0.0,
        **{
            f"causal_divergence_{channel}": float(mean(values)) if values else 0.0
            for channel, values in values_by_channel.items()
        },
    }


def summarize_ablation_runs(runs: list[dict[str, Any]], *, anchor_offset: int) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, float, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        key = (int(run["seed"]), float(run["perturb_scale"]), str(run["motif_family"]), int(run["motif_index"]))
        by_key[key][str(run["condition"])] = run
    per_run_rows = []
    for key, conditions in by_key.items():
        original = conditions.get("original")
        if original is None:
            continue
        for condition, run in conditions.items():
            metrics = dict(run["metrics"])
            if condition != "original":
                metrics.update(causal_divergence_against_original(original, run, anchor_offset=anchor_offset))
            per_run_rows.append(
                {
                    "seed": key[0],
                    "perturb_scale": key[1],
                    "motif_family": key[2],
                    "motif_index": key[3],
                    "condition": condition,
                    "regime": run["summary"]["regime_class"],
                    **{name: value for name, value in metrics.items() if isinstance(value, int | float | bool)},
                }
            )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_run_rows:
        grouped[str(row["condition"])].append(row)
    summary_rows = []
    for condition, rows in sorted(grouped.items()):
        numeric_keys = sorted(
            key
            for key in {item for row in rows for item in row}
            if all(isinstance(row.get(key), int | float | bool) for row in rows)
        )
        summary = {
            "condition": condition,
            "run_count": len(rows),
            "dominant_regime": dominant(row.get("regime", "unknown") for row in rows),
        }
        for key in numeric_keys:
            if key in {"seed", "motif_index"}:
                continue
            values = [float(row[key]) for row in rows if key in row]
            if values:
                summary[f"{key}_mean"] = float(mean(values))
                summary[f"{key}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    original_summary = next((row for row in summary_rows if row["condition"] == "original"), {})
    for row in summary_rows:
        if row["condition"] == "original":
            row["necessity_drop"] = 0.0
            row["richness_drop"] = 0.0
            continue
        row["necessity_drop"] = float(original_summary.get("causal_divergence_mean", 0.0)) - float(
            row.get("causal_divergence_mean", 0.0)
        )
        row["richness_drop"] = float(original_summary.get("internal_richness_mean", 0.0)) - float(
            row.get("internal_richness_mean", 0.0)
        )
        channel_values = [
            float(row.get(f"causal_divergence_{channel}_mean", 0.0))
            for channel in CHANNELS
        ]
        row["propagation_bias"] = max(channel_values) - min(channel_values) if channel_values else 0.0
    return summary_rows


def dominant(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return max(counts, key=counts.get) if counts else "unknown"


def run_to_pause_and_baseline(
    genome: dict[str, Any],
    *,
    candidate_id: str,
    config: CharacterizationConfig,
    seed: int,
    perturb_scale: float,
    motif: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(genome, config=config, condition="original")
    state = model.initial_state(1, torch.device(config.device))
    signature = motif["vector"]
    surfaces: list[torch.Tensor] = []
    components: list[dict[str, torch.Tensor]] = []
    paused_model: AdaptiveV9FiveChannel | None = None
    paused_state: tuple[torch.Tensor, ...] | None = None
    with torch.no_grad():
        for step in range(1, int(config.steps) + 1):
            if step == int(config.perturb_step):
                state = apply_controlled_perturbation(
                    state,
                    signature,
                    float(perturb_scale),
                    channel=config.perturb_channel,
                )
                model.record_perturbation(step, float(perturb_scale))
            state = model.step(state)
            surface = model.state_vector(state).view(-1).detach().float().cpu()
            comp = {
                name: tensor.view(-1).detach().float().cpu()
                for name, tensor in model.state_components(state).items()
            }
            surfaces.append(surface)
            components.append(comp)
            if step == int(config.capsule_pause_step):
                paused_model = copy.deepcopy(model)
                paused_state = clone_state(state)
    if paused_model is None or paused_state is None:
        raise RuntimeError("pause state was not captured")
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "perturb_scale": perturb_scale,
        "motif_family": motif["family"],
        "motif_index": motif["index"],
        "surfaces": surfaces,
        "components": components,
        "paused_model": paused_model,
        "paused_state": paused_state,
    }


def surface_resume_state(model: AdaptiveV9FiveChannel, state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    zeros = zero_like_state(state)
    surface = model.state_vector(state).detach()
    return surface.view_as(zeros[0]), zeros[1], zeros[2], zeros[3], zeros[4]


def partial_resume_state(state: tuple[torch.Tensor, ...], channel: str) -> tuple[torch.Tensor, ...]:
    values = list(zero_like_state(state))
    idx = list(CHANNELS).index(channel)
    values[idx] = state[idx].detach().clone()
    return tuple(values)


def continue_resume(
    model: AdaptiveV9FiveChannel,
    state: tuple[torch.Tensor, ...],
    *,
    start_step: int,
    steps: int,
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    surfaces: list[torch.Tensor] = []
    components: list[dict[str, torch.Tensor]] = []
    current = clone_state(state)
    with torch.no_grad():
        for _ in range(start_step + 1, steps + 1):
            current = model.step(current)
            surfaces.append(model.state_vector(current).view(-1).detach().float().cpu())
            components.append(
                {
                    name: tensor.view(-1).detach().float().cpu()
                    for name, tensor in model.state_components(current).items()
                }
            )
    return surfaces, components


def capsule_rows_for_condition(
    genome: dict[str, Any],
    *,
    candidate_id: str,
    config: CharacterizationConfig,
    seed: int,
    perturb_scale: float,
    motif: dict[str, Any],
) -> list[CapsuleRow]:
    baseline = run_to_pause_and_baseline(
        genome,
        candidate_id=candidate_id,
        config=config,
        seed=seed,
        perturb_scale=perturb_scale,
        motif=motif,
    )
    pause = int(config.capsule_pause_step)
    target_surfaces = baseline["surfaces"][pause:]
    target_components = baseline["components"][-1]
    paused_model = baseline["paused_model"]
    paused_state = baseline["paused_state"]
    variants: dict[ResumeKind, tuple[AdaptiveV9FiveChannel, tuple[torch.Tensor, ...] | None]] = {
        "uninterrupted": (paused_model, None),
        "full_internal_state": (copy.deepcopy(paused_model), paused_state),
        "surface_only": (copy.deepcopy(paused_model), surface_resume_state(paused_model, paused_state)),
        "fast_only": (copy.deepcopy(paused_model), partial_resume_state(paused_state, "fast")),
        "slow_only": (copy.deepcopy(paused_model), partial_resume_state(paused_state, "slow")),
        "control_only": (copy.deepcopy(paused_model), partial_resume_state(paused_state, "control")),
        "message_only": (copy.deepcopy(paused_model), partial_resume_state(paused_state, "message")),
        "carrier_only": (copy.deepcopy(paused_model), partial_resume_state(paused_state, "carrier")),
    }
    rows = []
    for kind, (model, resume_state) in variants.items():
        if kind == "uninterrupted":
            surfaces = target_surfaces
            final_components = target_components
        else:
            assert resume_state is not None
            surfaces, components = continue_resume(model, resume_state, start_step=pause, steps=int(config.steps))
            final_components = components[-1]
        final_left = surfaces[-1]
        final_right = target_surfaces[-1]
        denom = float(final_left.norm() * final_right.norm())
        final_cosine = float(torch.dot(final_left, final_right) / denom) if denom > 1e-10 else 0.0
        step_gaps = [normalized_l2(left, right) for left, right in zip(surfaces, target_surfaces)]
        rows.append(
            CapsuleRow(
                candidate_id=candidate_id,
                seed=seed,
                perturb_scale=float(perturb_scale),
                motif_family=str(motif["family"]),
                motif_index=int(motif["index"]),
                resume_kind=kind,
                final_cosine=final_cosine,
                final_l2_gap=normalized_l2(final_left, final_right),
                mean_step_gap=float(mean(step_gaps)) if step_gaps else 0.0,
                **{
                    f"{channel}_gap": normalized_l2(final_components[channel], target_components[channel])
                    for channel in CHANNELS
                },
            )
        )
    return rows


def summarize_capsule_rows(rows: list[CapsuleRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CapsuleRow]] = defaultdict(list)
    for row in rows:
        grouped[row.resume_kind].append(row)
    summary = []
    for kind, items in sorted(grouped.items()):
        payload = {"resume_kind": kind, "run_count": len(items)}
        for field in (
            "final_cosine",
            "final_l2_gap",
            "mean_step_gap",
            "fast_gap",
            "slow_gap",
            "control_gap",
            "message_gap",
            "carrier_gap",
        ):
            values = [float(getattr(row, field)) for row in items]
            payload[f"{field}_mean"] = float(mean(values))
            payload[f"{field}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        summary.append(payload)
    return summary


def numeric_dict(values: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items() if isinstance(value, int | float) and not isinstance(value, bool)}


def load_archive_candidates(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "candidates").glob("*.json"))
    candidates = []
    for path in paths:
        candidate = load_json(path)
        candidate["_candidate_path"] = str(path)
        candidate.setdefault("rank_mode", NATIVE_EMERGENCE_RANK_MODE)
        candidate["_native_rank"] = scalar_rank(candidate)
        candidates.append(candidate)
    return candidates


def lineage_key(candidate: dict[str, Any]) -> str:
    ancestors = candidate.get("ancestor_ids") or []
    if ancestors:
        return str(ancestors[0])
    parents = candidate.get("parent_ids") or []
    if parents:
        return str(parents[0])
    return str(candidate.get("id", "unknown"))


def parameter_signature_rows(root: Path, *, top_n: int = 5) -> tuple[list[ParameterSignatureRow], dict[str, Any]]:
    candidates = load_archive_candidates(root)
    ranked = sorted(candidates, key=lambda row: float(row["_native_rank"]), reverse=True)
    selected: list[tuple[str, dict[str, Any]]] = [("top_native_rank", candidate) for candidate in ranked[:top_n]]
    seen = set()
    for candidate in ranked:
        key = lineage_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        selected.append(("top_distinct_lineage", candidate))
        if len(seen) >= top_n:
            break
    rows = []
    for selection, candidate in selected:
        row = ParameterSignatureRow(
            selection=selection,
            candidate_id=str(candidate.get("id", "")),
            candidate_path=str(candidate.get("_candidate_path", "")),
            native_rank=float(candidate["_native_rank"]),
            generation=candidate.get("generation"),
            reproduction_kind=candidate.get("reproduction_kind"),
            parent_ids=list(candidate.get("parent_ids", [])),
            ancestor_ids=list(candidate.get("ancestor_ids", [])),
            lineage_key=lineage_key(candidate),
            metrics=numeric_dict(candidate.get("metrics", {})),
            scalars=numeric_dict(candidate.get("genome", {}).get("scalars", {})),
        )
        rows.append(row)
    baseline_scalars: dict[str, list[float]] = defaultdict(list)
    selected_scalars: dict[str, list[float]] = defaultdict(list)
    for candidate in candidates:
        for key, value in numeric_dict(candidate.get("genome", {}).get("scalars", {})).items():
            baseline_scalars[key].append(value)
    for row in rows:
        for key, value in row.scalars.items():
            selected_scalars[key].append(value)
    scalar_summary = {}
    for key in sorted(baseline_scalars):
        base = baseline_scalars[key]
        sel = selected_scalars.get(key, [])
        base_mean = mean(base)
        selected_mean = mean(sel) if sel else None
        scalar_summary[key] = {
            "baseline_mean": float(base_mean),
            "baseline_std": float(pstdev(base)) if len(base) > 1 else 0.0,
            "baseline_min": float(min(base)),
            "baseline_max": float(max(base)),
            "baseline_cv": float((pstdev(base) if len(base) > 1 else 0.0) / (abs(base_mean) + 1e-10)),
            "selected_mean": float(selected_mean) if selected_mean is not None else None,
            "selected_std": float(pstdev(sel)) if len(sel) > 1 else 0.0,
            "selected_min": float(min(sel)) if sel else None,
            "selected_max": float(max(sel)) if sel else None,
            "selected_cv": float((pstdev(sel) if len(sel) > 1 else 0.0) / (abs(mean(sel)) + 1e-10)) if sel else None,
            "selected_minus_baseline_mean": float(selected_mean - base_mean) if selected_mean is not None else None,
        }
    summary = {
        "archive_root": str(root),
        "candidate_count": len(candidates),
        "selected_count": len(rows),
        "scalar_summary": scalar_summary,
    }
    return rows, summary


def flatten_parameter_rows(rows: list[ParameterSignatureRow]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        payload = row.model_dump(mode="json")
        metrics = payload.pop("metrics")
        scalars = payload.pop("scalars")
        payload["parent_ids"] = json.dumps(payload["parent_ids"], sort_keys=True)
        payload["ancestor_ids"] = json.dumps(payload["ancestor_ids"], sort_keys=True)
        for key, value in metrics.items():
            payload[f"metric_{key}"] = value
        for key, value in scalars.items():
            payload[f"scalar_{key}"] = value
        flat.append(payload)
    return flat


def flatten_ablation_rows(rows: list[AblationRow]) -> list[dict[str, Any]]:
    return [row.model_dump(mode="json") for row in rows]


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, *, alpha: float = 1e-3) -> np.ndarray:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    eval_aug = np.concatenate([x_eval, np.ones((x_eval.shape[0], 1))], axis=1)
    reg = alpha * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_train)
    return eval_aug @ weights


def score_regression(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if total <= 1e-12:
        return 1.0 if float(np.mean(np.abs(y_true - y_pred))) <= 1e-8 else 0.0
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / total


def score_classification(y_true: list[str], y_pred: np.ndarray, classes: list[str]) -> float:
    if not y_true:
        return 0.0
    predicted = [classes[int(np.argmax(row))] for row in y_pred]
    return float(sum(left == right for left, right in zip(y_true, predicted))) / float(len(y_true))


def channel_probe_artifacts(runs: list[dict[str, Any]], *, perturb_step: int, perturb_channel: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run small linear probes over message, carrier, and slow state snapshots."""
    rows = []
    target_steps = {
        "at_perturb": int(perturb_step),
        "post_perturb": int(perturb_step) + 4,
        "final": None,
    }
    for run in runs:
        if run.get("condition") != "original":
            continue
        trajectory = run.get("trajectory", [])
        if not trajectory:
            continue
        by_step = {int(row["step"]): row for row in trajectory}
        for step_label, step_value in target_steps.items():
            source = trajectory[-1] if step_value is None else by_step.get(step_value)
            if source is None:
                continue
            for channel in ("message", "carrier", "slow"):
                tensor = source.get("_channel_states", {}).get(channel)
                if tensor is None:
                    continue
                values = tensor.view(-1).detach().cpu().numpy().astype(float)
                rows.append(
                    {
                        "channel": channel,
                        "step_label": step_label,
                        "seed": int(run["seed"]),
                        "perturb_scale": float(run["perturb_scale"]),
                        "perturb_step": int(perturb_step),
                        "perturb_channel": perturb_channel,
                        "motif_family": str(run["motif_family"]),
                        "motif_index": int(run["motif_index"]),
                        "features": values,
                    }
                )
    flat_rows = []
    summary: dict[str, Any] = {"row_count": len(rows), "probes": []}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["channel"], row["step_label"])].append(row)
    for (channel, step_label), items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        x = np.vstack([item["features"] for item in items])
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
        labels = {
            "scale": np.asarray([float(item["perturb_scale"]) for item in items], dtype=np.float64),
            "step": np.asarray([float(item["perturb_step"]) for item in items], dtype=np.float64),
        }
        for target, y in labels.items():
            pred = ridge_predict(x, y[:, None], x)[:, 0]
            score = score_regression(y, pred)
            summary["probes"].append(
                {"channel": channel, "step_label": step_label, "target": target, "metric": "r2", "score": score}
            )
            flat_rows.append(
                {"channel": channel, "step_label": step_label, "target": target, "metric": "r2", "score": score}
            )
        categorical = {
            "family": [str(item["motif_family"]) for item in items],
            "channel": [str(item["perturb_channel"]) for item in items],
        }
        for target, values in categorical.items():
            classes = sorted(set(values))
            y = np.zeros((len(values), len(classes)), dtype=np.float64)
            for idx, value in enumerate(values):
                y[idx, classes.index(value)] = 1.0
            pred = ridge_predict(x, y, x)
            score = score_classification(values, pred, classes)
            summary["probes"].append(
                {
                    "channel": channel,
                    "step_label": step_label,
                    "target": target,
                    "metric": "accuracy",
                    "score": score,
                    "classes": classes,
                }
            )
            flat_rows.append(
                {
                    "channel": channel,
                    "step_label": step_label,
                    "target": target,
                    "metric": "accuracy",
                    "score": score,
                    "classes": json.dumps(classes),
                }
            )
    return flat_rows, summary


def channel_necessity_order(ablation_summary: list[dict[str, Any]]) -> list[str]:
    candidates = []
    for row in ablation_summary:
        condition = str(row.get("condition", ""))
        if not condition.endswith("_disabled") or condition == "routes_disabled":
            continue
        channel = condition.removesuffix("_disabled")
        score = float(row.get("richness_drop", 0.0)) + float(row.get("causal_divergence_mean", 0.0))
        candidates.append((score, channel))
    return [channel for _, channel in sorted(candidates, reverse=True)]


def evidence_gate(ablation_summary: list[dict[str, Any]], capsule_summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = {str(row["condition"]): row for row in ablation_summary}
    by_resume = {str(row["resume_kind"]): row for row in capsule_summary}
    gain_zero = by_condition.get("gain_zero", {})
    routes = by_condition.get("routes_disabled", {})
    surface = by_resume.get("surface_only", {})
    full = by_resume.get("full_internal_state", {})
    return {
        "status": "candidate_evidence_not_promoted",
        "gain_zero_clean": float(gain_zero.get("gain_zero_clean_mean", 0.0)) >= 1.0,
        "routes_disabled_divergence_positive": float(routes.get("causal_divergence_mean", 0.0)) > 0.0,
        "gain_zero_divergence_positive": float(gain_zero.get("causal_divergence_mean", 0.0)) > 0.0,
        "full_resume_exact": float(full.get("final_l2_gap_mean", 1.0)) <= 1e-7,
        "surface_resume_gap_positive": float(surface.get("final_l2_gap_mean", 0.0)) > 0.0,
        "promotion_note": "Replication is required before updating promoted claims.",
    }


def capsule_flags(capsule_summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_resume = {str(row["resume_kind"]): row for row in capsule_summary}
    full = by_resume.get("full_internal_state", {})
    surface = by_resume.get("surface_only", {})
    single_channel_gaps = {
        channel: float(by_resume.get(f"{channel}_only", {}).get("mean_step_gap_mean", 1.0))
        for channel in CHANNELS
    }
    best_single = min(single_channel_gaps, key=single_channel_gaps.get) if single_channel_gaps else None
    return {
        "full_state_resume_outperforms_surface": float(full.get("mean_step_gap_mean", 1.0)) < float(
            surface.get("mean_step_gap_mean", 1.0)
        ),
        "best_single_channel_resume": best_single,
        "single_channel_mean_step_gaps": single_channel_gaps,
    }


def run_characterization(config: CharacterizationConfig) -> SummaryPayload:
    candidate = load_json(config.candidate_path)
    candidate_id = str(candidate.get("id", config.candidate_path.stem))
    genome = candidate["genome"]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "config.json", config.model_dump(mode="json"))

    all_runs: list[dict[str, Any]] = []
    ablation_rows: list[AblationRow] = []
    capsule_rows: list[CapsuleRow] = []
    for seed in config.seeds:
        motifs = [motif_from_spec(spec, int(config.hidden_size), int(seed)) for spec in config.motifs]
        for scale in config.perturb_scales:
            for motif in motifs:
                for condition in config.conditions:
                    run = run_condition(
                        genome,
                        candidate_id=candidate_id,
                        config=config,
                        condition=condition,
                        seed=int(seed),
                        perturb_scale=float(scale),
                        motif=motif,
                    )
                    all_runs.append(run)
                    ablation_rows.extend(run["rows"])
                capsule_rows.extend(
                    capsule_rows_for_condition(
                        genome,
                        candidate_id=candidate_id,
                        config=config,
                        seed=int(seed),
                        perturb_scale=float(scale),
                        motif=motif,
                    )
                )

    ablation_summary = summarize_ablation_runs(all_runs, anchor_offset=int(config.anchor_offset))
    capsule_summary = summarize_capsule_rows(capsule_rows)
    probe_rows, probe_summary = channel_probe_artifacts(
        all_runs,
        perturb_step=int(config.perturb_step),
        perturb_channel=str(config.perturb_channel),
    )
    parameter_summary_path = ""
    if config.write_parameter_signatures:
        parameter_rows, parameter_summary = parameter_signature_rows(config.archive_root)
        parameter_flat = flatten_parameter_rows(parameter_rows)
        write_csv(PARAMETER_SIGNATURE_CSV, parameter_flat)
        write_parquet(PARAMETER_SIGNATURE_PARQUET, parameter_flat)
        write_json(PARAMETER_SIGNATURE_SUMMARY, parameter_summary)
        parameter_summary_path = str(PARAMETER_SIGNATURE_SUMMARY)

    ablation_flat = flatten_ablation_rows(ablation_rows)
    capsule_flat = [row.model_dump(mode="json") for row in capsule_rows]
    write_jsonl(config.output_dir / "ablation_rows.jsonl", ablation_rows)
    write_csv(config.output_dir / "ablation_summary.csv", ablation_summary)
    write_parquet(config.output_dir / "ablation_results.parquet", ablation_flat)
    write_jsonl(config.output_dir / "capsule_rows.jsonl", capsule_rows)
    write_csv(config.output_dir / "capsule_summary.csv", capsule_summary)
    write_parquet(config.output_dir / "capsule_results.parquet", capsule_flat)
    write_csv(config.output_dir / "channel_probe_summary.csv", probe_rows)
    if probe_rows:
        write_parquet(config.output_dir / "channel_probe_results.parquet", probe_rows)
    write_json(config.output_dir / "channel_probe_summary.json", probe_summary)

    gain_zero_summary = next((row for row in ablation_summary if row["condition"] == "gain_zero"), {})
    capsule_result_flags = capsule_flags(capsule_summary)

    summary = SummaryPayload(
        experiment="gate_state_propagation_characterization_20260511",
        candidate_id=candidate_id,
        config=config.model_dump(mode="json"),
        run_count=len(all_runs),
        ablation_summary=ablation_summary,
        capsule_summary=capsule_summary,
        parameter_signature_summary_path=parameter_summary_path,
        evidence_gate=evidence_gate(ablation_summary, capsule_summary),
        held_out_seeds=list(config.seeds),
        perturb_steps=[int(config.perturb_step)],
        channel_necessity_order=channel_necessity_order(ablation_summary),
        gain_zero_cleanliness={
            key: value
            for key, value in gain_zero_summary.items()
            if key.startswith("gain_zero_") or key == "condition"
        },
        capsule_result_flags=capsule_result_flags,
        probe_summary=probe_summary,
    )
    write_json(config.output_dir / "summary.json", summary.model_dump(mode="json"))
    write_readme(config.output_dir, summary)
    return summary


def write_readme(out_dir: Path, summary: SummaryPayload) -> None:
    text = f"""# Gate-State Propagation Characterization

Candidate evidence for `{summary.candidate_id}`. This is not a promoted mechanism claim.

Artifacts:

- `config.json`: exact characterization configuration
- `summary.json`: aggregate metrics and evidence gate
- `ablation_rows.jsonl`: per-step rows for original, release interventions, and channel-disabled interventions
- `ablation_summary.csv`: per-condition summary metrics
- `ablation_results.parquet`: searchable per-step ablation table
- `capsule_rows.jsonl`: capsule-continuation probe rows
- `capsule_summary.csv`: per-resume aggregate capsule metrics
- `capsule_results.parquet`: searchable capsule table
- `channel_probe_summary.json`: simple linear probe summary for message/carrier/slow state
- `channel_probe_results.parquet`: searchable probe-score table when enough rows exist

Promotion remains blocked until replicated evidence satisfies the criteria in `docs/NATIVE_MECHANISMS.md`.
"""
    out_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_motifs(text: str) -> list[MotifSpec]:
    motifs = []
    for item in text.split(","):
        if not item.strip():
            continue
        family, _, index = item.partition(":")
        motifs.append(MotifSpec(family=family.strip(), index=int(index or 0)))
    return motifs


def parse_conditions(text: str) -> list[Condition]:
    return [item.strip() for item in text.split(",") if item.strip()]  # type: ignore[return-value]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", default="94,95,96,97,98,99,100,101,102")
    parser.add_argument("--perturb-scales", default="0.2,0.35,0.7")
    parser.add_argument("--motifs", default="basis:0,gaussian:0")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--perturb-step", type=int, default=64)
    parser.add_argument("--capsule-pause-step", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one seed, one scale, one motif, and original/gain-zero only.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = parse_csv_ints(args.seeds)
    scales = parse_csv_floats(args.perturb_scales)
    motifs = parse_motifs(args.motifs)
    conditions = parse_conditions(args.conditions)
    if args.smoke:
        seeds = seeds[:1]
        scales = scales[:1]
        motifs = motifs[:1]
        conditions = ["original", "gain_zero"]  # type: ignore[list-item]
    config = CharacterizationConfig(
        candidate_path=args.candidate_path,
        archive_root=args.archive_root,
        output_dir=args.output_dir,
        seeds=seeds,
        perturb_scales=scales,
        motifs=motifs,
        conditions=conditions,
        steps=args.steps,
        perturb_step=args.perturb_step,
        capsule_pause_step=args.capsule_pause_step,
        hidden_size=args.hidden_size,
        rank=args.rank,
        device=args.device,
        write_parameter_signatures=not args.smoke,
    )
    summary = run_characterization(config)
    print(f"wrote {config.output_dir / 'summary.json'}")
    print(json.dumps(summary.evidence_gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
