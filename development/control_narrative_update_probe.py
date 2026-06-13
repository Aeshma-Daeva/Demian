#!/usr/bin/env python3
"""Probe structured perturbation-history updates in the control channel."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator, model_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.control_capsule_maintenance import (
    CAPSULE_CHANNELS,
    candidate_paths_from_replication_summary,
    zero_channels,
)
from development.gate_state_propagation_characterization import (
    CharacterizationConfig,
    MotifSpec,
    apply_controlled_perturbation,
    build_model,
    clone_state,
    load_json,
    motif_from_spec,
    normalized_l2,
    write_csv,
    write_json,
    write_jsonl,
)

DEFAULT_REPLICATION_SUMMARY = Path("data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json")
DEFAULT_OUTPUT_DIR = Path("data/diagnostics/control_narrative_update_probe_20260514")
EPS = 1e-12

HistoryName = Literal["none", "A", "B", "A_then_B", "B_then_A", "A_then_A", "B_then_B"]
NarrativeCondition = Literal[
    "control_preserved",
    "control_reset_at_event",
    "control_clamped_zero",
    "plastic_reset_at_event",
]
HISTORY_NAMES: tuple[HistoryName, ...] = ("none", "A", "B", "A_then_B", "B_then_A", "A_then_A", "B_then_B")
PRIMARY_CONDITIONS: tuple[NarrativeCondition, ...] = (
    "control_preserved",
    "control_reset_at_event",
    "control_clamped_zero",
)
ALL_CONDITIONS: tuple[NarrativeCondition, ...] = (*PRIMARY_CONDITIONS, "plastic_reset_at_event")


class FlexibleSchema(BaseModel):
    """Validate expected fields while preserving searchable extras."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class HistoryEvent(FlexibleSchema):
    step: NonNegativeInt
    motif_key: Literal["A", "B"]


class ControlNarrativeConfig(FlexibleSchema):
    candidate_paths: list[Path] = Field(default_factory=list)
    replication_summary_path: Path = DEFAULT_REPLICATION_SUMMARY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seeds: list[int] = Field(default_factory=lambda: [96, 97, 98])
    perturb_scales: list[float] = Field(default_factory=lambda: [0.35, 0.7])
    motif_a: MotifSpec = Field(default_factory=lambda: MotifSpec(family="basis", index=0))
    motif_b: MotifSpec = Field(default_factory=lambda: MotifSpec(family="gaussian", index=0))
    histories: list[HistoryName] = Field(default_factory=lambda: list(HISTORY_NAMES))
    conditions: list[NarrativeCondition] = Field(default_factory=lambda: list(ALL_CONDITIONS))
    hidden_size: NonNegativeInt = 32
    rank: NonNegativeInt = 2
    steps: NonNegativeInt = 160
    event_steps: tuple[NonNegativeInt, NonNegativeInt] = (64, 96)
    tail_window: NonNegativeInt = 16
    perturb_channel: str = "fast"
    device: str = "cpu"
    write_trace_vectors: bool = False

    @model_validator(mode="after")
    def steps_are_consistent(self) -> "ControlNarrativeConfig":
        first, second = int(self.event_steps[0]), int(self.event_steps[1])
        if first <= 1:
            raise ValueError("first event step must be > 1 so a pre-event snapshot exists")
        if first >= second:
            raise ValueError("event_steps must be increasing")
        if second > self.steps:
            raise ValueError("event_steps must be <= steps")
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
            motifs=[self.motif_a, self.motif_b],
            hidden_size=int(self.hidden_size),
            rank=int(self.rank),
            steps=int(self.steps),
            perturb_step=int(self.event_steps[0]),
            capsule_pause_step=int(self.event_steps[0]),
            perturb_channel=self.perturb_channel,
            device=self.device,
            write_parameter_signatures=False,
        )


class ControlNarrativeRow(FlexibleSchema):
    candidate_id: str
    candidate_path: str
    condition: NarrativeCondition
    history_name: HistoryName
    event_sequence: str
    seed: int
    perturb_scale: float
    steps: int
    first_event_step: int
    second_event_step: int
    control_update_norm: float
    control_shock_norm: float
    control_return_fraction: float
    control_update_cosine_to_capsule_delta: float
    plastic_update_norm: float
    capsule_update_norm: float
    surface_update_norm: float
    final_control_norm: float
    within_history_update_cosine: float | None = None
    cross_history_update_cosine: float | None = None
    history_separation_margin: float | None = None
    history_classification_accuracy: float | None = None
    order_sensitivity: float | None = None
    repeat_consistency: float | None = None
    trace_control_update_vector: list[float] | None = None
    trace_plastic_update_vector: list[float] | None = None

    @field_validator("*")
    @classmethod
    def numeric_values_are_finite(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("numeric control narrative fields must be finite")
        return value


@dataclass(frozen=True)
class NarrativeRun:
    row: ControlNarrativeRow
    control_update: torch.Tensor
    plastic_update: torch.Tensor


def tensor_to_vector(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(-1).detach().float().cpu()


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_vec = tensor_to_vector(left)
    right_vec = tensor_to_vector(right)
    denom = float(left_vec.norm() * right_vec.norm())
    if denom <= EPS:
        return 0.0
    return float(torch.dot(left_vec, right_vec) / denom)


def mean_vector(vectors: list[torch.Tensor]) -> torch.Tensor | None:
    if not vectors:
        return None
    return torch.stack([tensor_to_vector(vector) for vector in vectors], dim=0).mean(dim=0)


def vector_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return normalized_l2(tensor_to_vector(left), tensor_to_vector(right))


def fit_vector_width(vector: torch.Tensor, width: int) -> torch.Tensor:
    flat = tensor_to_vector(vector)
    if flat.numel() == width:
        return flat
    if flat.numel() > width:
        return flat[:width]
    return torch.nn.functional.pad(flat, (0, width - flat.numel()))


def build_history_events(
    history_name: HistoryName,
    event_steps: tuple[int, int],
) -> list[HistoryEvent]:
    first, second = int(event_steps[0]), int(event_steps[1])
    mapping: dict[HistoryName, list[tuple[int, Literal["A", "B"]]]] = {
        "none": [],
        "A": [(first, "A")],
        "B": [(first, "B")],
        "A_then_B": [(first, "A"), (second, "B")],
        "B_then_A": [(first, "B"), (second, "A")],
        "A_then_A": [(first, "A"), (second, "A")],
        "B_then_B": [(first, "B"), (second, "B")],
    }
    return [HistoryEvent(step=step, motif_key=motif_key) for step, motif_key in mapping[history_name]]


def event_sequence(events: list[HistoryEvent]) -> str:
    return "none" if not events else ",".join(f"{event.step}:{event.motif_key}" for event in events)


def capsule_vector(components: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor_to_vector(components[channel]) for channel in CAPSULE_CHANNELS])


def plastic_state_vector(model: Any) -> torch.Tensor:
    values = getattr(model, "plastic_state", {})
    if not values:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat([tensor_to_vector(values[name]) for name in sorted(values)])


def reset_plastic_state(model: Any) -> None:
    for name, value in getattr(model, "plastic_state", {}).items():
        model.plastic_state[name] = torch.zeros_like(value)


def state_snapshot(model: Any, state: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    components = {
        channel: tensor_to_vector(tensor)
        for channel, tensor in model.state_components(state).items()
    }
    return {
        "control": components["control"],
        "capsule": capsule_vector(components),
        "surface": tensor_to_vector(model.state_vector(state)),
        "plastic": plastic_state_vector(model),
    }


def tail_mean_snapshot(snapshots: list[dict[str, torch.Tensor]], tail_window: int) -> dict[str, torch.Tensor]:
    tail_count = max(1, min(int(tail_window), len(snapshots)))
    tail = snapshots[-tail_count:]
    return {
        key: torch.stack([snapshot[key] for snapshot in tail], dim=0).mean(dim=0)
        for key in ("control", "capsule", "surface", "plastic")
    }


def history_structure_metrics(vectors_by_history: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    histories = sorted(history for history, vectors in vectors_by_history.items() if vectors)
    within_values: list[float] = []
    cross_values: list[float] = []
    centroids: dict[str, torch.Tensor] = {}
    for history in histories:
        vectors = vectors_by_history[history]
        centroid = mean_vector(vectors)
        if centroid is not None:
            centroids[history] = centroid
        for i, left in enumerate(vectors):
            for right in vectors[i + 1 :]:
                within_values.append(cosine_similarity(left, right))
    for i, left_history in enumerate(histories):
        for right_history in histories[i + 1 :]:
            for left in vectors_by_history[left_history]:
                for right in vectors_by_history[right_history]:
                    cross_values.append(cosine_similarity(left, right))

    correct = 0
    total = 0
    margins: list[float] = []
    for history in histories:
        history_vectors = vectors_by_history[history]
        for vector_index, vector in enumerate(history_vectors):
            distances = {
                centroid_history: vector_distance(vector, centroid)
                for centroid_history, centroid in centroids.items()
            }
            if not distances:
                continue
            if len(history_vectors) > 1:
                own_vectors = [
                    item
                    for item_index, item in enumerate(history_vectors)
                    if item_index != vector_index
                ]
                own_centroid = mean_vector(own_vectors)
                if own_centroid is not None:
                    distances[history] = vector_distance(vector, own_centroid)
            ordered = sorted(distances.items(), key=lambda item: item[1])
            if ordered[0][0] == history:
                correct += 1
            total += 1
            own = distances[history]
            other = min((distance for name, distance in distances.items() if name != history), default=own)
            margins.append(float(other - own))

    return {
        "within_history_update_cosine": float(mean(within_values)) if within_values else 0.0,
        "cross_history_update_cosine": float(mean(cross_values)) if cross_values else 0.0,
        "history_separation_margin": float(mean(margins)) if margins else 0.0,
        "history_classification_accuracy": float(correct) / float(total) if total else 0.0,
    }


def order_sensitivity_metrics(vectors_by_history: dict[str, list[torch.Tensor]]) -> dict[str, float | None]:
    a_then_b = mean_vector(vectors_by_history.get("A_then_B", []))
    b_then_a = mean_vector(vectors_by_history.get("B_then_A", []))
    repeat_distances: list[float] = []
    for history in ("A_then_A", "B_then_B"):
        vectors = vectors_by_history.get(history, [])
        for i, left in enumerate(vectors):
            for right in vectors[i + 1 :]:
                repeat_distances.append(vector_distance(left, right))
    return {
        "order_sensitivity": vector_distance(a_then_b, b_then_a) if a_then_b is not None and b_then_a is not None else None,
        "repeat_consistency": float(mean(repeat_distances)) if repeat_distances else None,
    }


def run_history(
    genome: dict[str, Any],
    *,
    candidate_id: str,
    candidate_path: Path,
    config: ControlNarrativeConfig,
    characterization_config: CharacterizationConfig,
    condition: NarrativeCondition,
    history_name: HistoryName,
    seed: int,
    perturb_scale: float,
) -> NarrativeRun:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(genome, config=characterization_config, condition="original")
    state = model.initial_state(1, torch.device(config.device))
    reset_control_state = state[2].detach().clone()
    motifs = {
        "A": motif_from_spec(config.motif_a, int(config.hidden_size), int(seed)),
        "B": motif_from_spec(config.motif_b, int(config.hidden_size), int(seed)),
    }
    events = build_history_events(history_name, (int(config.event_steps[0]), int(config.event_steps[1])))
    events_by_step = {int(event.step): event for event in events}
    first_event_step = int(config.event_steps[0])
    pre_snapshot: dict[str, torch.Tensor] | None = None
    shock_snapshots: list[dict[str, torch.Tensor]] = []
    snapshots: list[dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for step in range(1, int(config.steps) + 1):
            if step == first_event_step - 1:
                pre_snapshot = state_snapshot(model, state)
            event = events_by_step.get(step)
            if event is not None:
                if condition == "control_reset_at_event":
                    values = list(state)
                    values[2] = reset_control_state.detach().clone()
                    state = tuple(values)
                if condition == "plastic_reset_at_event":
                    reset_plastic_state(model)
                state = apply_controlled_perturbation(
                    state,
                    motifs[event.motif_key]["vector"],
                    float(perturb_scale),
                    channel=config.perturb_channel,
                )
                model.record_perturbation(step, float(perturb_scale))
            state = model.step(state)
            if condition == "control_clamped_zero":
                state = zero_channels(state, ("control",))
            snapshot = state_snapshot(model, state)
            snapshots.append(snapshot)
            if event is not None:
                shock_snapshots.append(snapshot)

    if pre_snapshot is None:
        raise RuntimeError("pre-event snapshot was not captured")
    tail_snapshot = tail_mean_snapshot(snapshots, int(config.tail_window))
    shock_snapshot = shock_snapshots[0] if shock_snapshots else pre_snapshot
    control_update = tail_snapshot["control"] - pre_snapshot["control"]
    shock_update = shock_snapshot["control"] - pre_snapshot["control"]
    capsule_delta = fit_vector_width(tail_snapshot["capsule"] - pre_snapshot["capsule"], control_update.numel())
    plastic_update = tail_snapshot["plastic"] - pre_snapshot["plastic"]
    control_update_norm = normalized_l2(tail_snapshot["control"], pre_snapshot["control"])
    control_shock_norm = normalized_l2(shock_snapshot["control"], pre_snapshot["control"])
    control_return_fraction = control_update_norm / control_shock_norm if control_shock_norm > EPS else 0.0
    trace_control = control_update.tolist() if config.write_trace_vectors else None
    trace_plastic = plastic_update.tolist() if config.write_trace_vectors else None
    row = ControlNarrativeRow(
        candidate_id=candidate_id,
        candidate_path=candidate_path.as_posix(),
        condition=condition,
        history_name=history_name,
        event_sequence=event_sequence(events),
        seed=int(seed),
        perturb_scale=float(perturb_scale),
        steps=int(config.steps),
        first_event_step=int(config.event_steps[0]),
        second_event_step=int(config.event_steps[1]),
        control_update_norm=float(control_update_norm),
        control_shock_norm=float(control_shock_norm),
        control_return_fraction=float(control_return_fraction),
        control_update_cosine_to_capsule_delta=cosine_similarity(control_update, capsule_delta),
        plastic_update_norm=normalized_l2(tail_snapshot["plastic"], pre_snapshot["plastic"]) if tail_snapshot["plastic"].numel() else 0.0,
        capsule_update_norm=normalized_l2(tail_snapshot["capsule"], pre_snapshot["capsule"]),
        surface_update_norm=normalized_l2(tail_snapshot["surface"], pre_snapshot["surface"]),
        final_control_norm=float(snapshots[-1]["control"].norm()) / math.sqrt(max(snapshots[-1]["control"].numel(), 1)),
        trace_control_update_vector=trace_control,
        trace_plastic_update_vector=trace_plastic,
    )
    return NarrativeRun(row=row, control_update=control_update, plastic_update=plastic_update)


def attach_structure_metrics(runs: list[NarrativeRun]) -> list[ControlNarrativeRow]:
    grouped: dict[tuple[str, NarrativeCondition], list[NarrativeRun]] = defaultdict(list)
    for run in runs:
        grouped[(run.row.candidate_id, run.row.condition)].append(run)

    metrics_by_group: dict[tuple[str, NarrativeCondition], dict[str, float | None]] = {}
    for key, group_runs in grouped.items():
        vectors_by_history: dict[str, list[torch.Tensor]] = defaultdict(list)
        for run in group_runs:
            if run.row.history_name != "none":
                vectors_by_history[run.row.history_name].append(run.control_update)
        metrics = history_structure_metrics(vectors_by_history)
        metrics.update(order_sensitivity_metrics(vectors_by_history))
        metrics_by_group[key] = metrics

    rows: list[ControlNarrativeRow] = []
    for run in runs:
        metrics = metrics_by_group[(run.row.candidate_id, run.row.condition)]
        rows.append(
            run.row.model_copy(
                update={
                    "within_history_update_cosine": metrics["within_history_update_cosine"],
                    "cross_history_update_cosine": metrics["cross_history_update_cosine"],
                    "history_separation_margin": metrics["history_separation_margin"],
                    "history_classification_accuracy": metrics["history_classification_accuracy"],
                    "order_sensitivity": metrics["order_sensitivity"],
                    "repeat_consistency": metrics["repeat_consistency"],
                }
            )
        )
    return rows


def run_candidate_control_narrative_probe(candidate_path: Path, config: ControlNarrativeConfig) -> list[NarrativeRun]:
    candidate = load_json(candidate_path)
    candidate_id = str(candidate.get("id", candidate_path.stem))
    genome = candidate["genome"]
    characterization_config = config.as_characterization_config(candidate_path)
    runs: list[NarrativeRun] = []
    for seed in config.seeds:
        for perturb_scale in config.perturb_scales:
            for condition in config.conditions:
                for history_name in config.histories:
                    runs.append(
                        run_history(
                            genome,
                            candidate_id=candidate_id,
                            candidate_path=candidate_path,
                            config=config,
                            characterization_config=characterization_config,
                            condition=condition,
                            history_name=history_name,
                            seed=int(seed),
                            perturb_scale=float(perturb_scale),
                        )
                    )
    return runs


def mean_optional(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def summarize_rows(rows: list[ControlNarrativeRow]) -> dict[str, Any]:
    grouped: dict[tuple[str, NarrativeCondition], list[ControlNarrativeRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.candidate_id, row.condition)].append(row)

    numeric_fields = (
        "control_update_norm",
        "control_shock_norm",
        "control_return_fraction",
        "control_update_cosine_to_capsule_delta",
        "plastic_update_norm",
        "capsule_update_norm",
        "surface_update_norm",
        "final_control_norm",
        "within_history_update_cosine",
        "cross_history_update_cosine",
        "history_separation_margin",
        "history_classification_accuracy",
        "order_sensitivity",
        "repeat_consistency",
    )
    by_candidate_condition: list[dict[str, Any]] = []
    for (candidate_id, condition), items in sorted(grouped.items()):
        payload: dict[str, Any] = {
            "candidate_id": candidate_id,
            "condition": condition,
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
        by_candidate_condition.append(payload)

    preserved = [row for row in rows if row.condition == "control_preserved"]
    return {
        "experiment": "control_narrative_update_probe",
        "row_count": len(rows),
        "candidate_count": len({row.candidate_id for row in rows}),
        "by_candidate_condition": by_candidate_condition,
        "aggregate": {
            "control_preserved_classification_accuracy_mean": mean_optional(
                [row.history_classification_accuracy for row in preserved]
            ),
            "control_preserved_separation_margin_mean": mean_optional(
                [row.history_separation_margin for row in preserved]
            ),
            "control_preserved_order_sensitivity_mean": mean_optional(
                [row.order_sensitivity for row in preserved]
            ),
            "control_preserved_repeat_consistency_mean": mean_optional(
                [row.repeat_consistency for row in preserved]
            ),
        },
        "interpretation_boundary": (
            "This probe tests whether control carries structured perturbation-history updates. "
            "Do not promote narrative self claims unless preserved-control structure replicates across candidates "
            "and beats control-reset and control-clamped controls."
        ),
    }


def run_control_narrative_update_probe(config: ControlNarrativeConfig) -> dict[str, Any]:
    candidate_paths = config.candidate_paths or candidate_paths_from_replication_summary(config.replication_summary_path)
    torch.set_grad_enabled(False)
    runs: list[NarrativeRun] = []
    for candidate_path in candidate_paths:
        runs.extend(run_candidate_control_narrative_probe(candidate_path, config))
    rows = attach_structure_metrics(runs)
    payload = summarize_rows(rows)
    payload["config"] = {
        **config.model_dump(mode="json"),
        "candidate_paths": [path.as_posix() for path in candidate_paths],
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.output_dir / "control_narrative_rows.jsonl", rows)
    write_csv(config.output_dir / "control_narrative_summary.csv", payload["by_candidate_condition"])
    write_json(config.output_dir / "control_narrative_summary.json", payload)
    write_readme(config.output_dir, payload)
    return payload


def write_readme(output_dir: Path, payload: dict[str, Any]) -> None:
    text = f"""# Control Narrative Update Probe

Structured perturbation-history update probe for the Track B control channel.

Artifacts:

- `control_narrative_rows.jsonl`: per-history rows with control, capsule, surface, and plastic update metrics
- `control_narrative_summary.csv`: per-candidate, per-condition aggregate metrics
- `control_narrative_summary.json`: aggregate interpretation metrics and config

Interpretation boundary: {payload["interpretation_boundary"]}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_paths(text: str | None) -> list[Path]:
    if not text:
        return []
    return [Path(item.strip()) for item in text.split(",") if item.strip()]


def parse_motif(text: str) -> MotifSpec:
    family, _, index = text.partition(":")
    return MotifSpec(family=family.strip(), index=int(index or 0))


def parse_histories(text: str) -> list[HistoryName]:
    values: list[HistoryName] = []
    valid = set(HISTORY_NAMES)
    for item in text.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in valid:
            raise ValueError(f"unknown history: {name}")
        values.append(name)  # type: ignore[arg-type]
    return values


def parse_conditions(text: str) -> list[NarrativeCondition]:
    values: list[NarrativeCondition] = []
    valid = set(ALL_CONDITIONS)
    for item in text.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in valid:
            raise ValueError(f"unknown condition: {name}")
        values.append(name)  # type: ignore[arg-type]
    return values


def parse_event_steps(text: str) -> tuple[int, int]:
    values = parse_csv_ints(text)
    if len(values) != 2:
        raise ValueError("exactly two event steps are required")
    return values[0], values[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-paths", default=None)
    parser.add_argument("--replication-summary-path", type=Path, default=DEFAULT_REPLICATION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="96,97,98")
    parser.add_argument("--perturb-scales", default="0.35,0.7")
    parser.add_argument("--motif-a", default="basis:0")
    parser.add_argument("--motif-b", default="gaussian:0")
    parser.add_argument("--histories", default=",".join(HISTORY_NAMES))
    parser.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--event-steps", default="64,96")
    parser.add_argument("--tail-window", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-trace-vectors", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ControlNarrativeConfig(
        candidate_paths=parse_paths(args.candidate_paths),
        replication_summary_path=args.replication_summary_path,
        output_dir=args.output_dir,
        seeds=parse_csv_ints(args.seeds),
        perturb_scales=parse_csv_floats(args.perturb_scales),
        motif_a=parse_motif(args.motif_a),
        motif_b=parse_motif(args.motif_b),
        histories=parse_histories(args.histories),
        conditions=parse_conditions(args.conditions),
        hidden_size=int(args.hidden_size),
        rank=int(args.rank),
        steps=int(args.steps),
        event_steps=parse_event_steps(args.event_steps),
        tail_window=int(args.tail_window),
        device=str(args.device),
        write_trace_vectors=bool(args.write_trace_vectors),
    )
    payload = run_control_narrative_update_probe(config)
    print(f"wrote {config.output_dir / 'control_narrative_summary.json'}")
    print(f"rows={payload['row_count']}")


if __name__ == "__main__":
    main()
