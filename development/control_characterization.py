#!/usr/bin/env python3
"""Control-channel characterization and persistence probes."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Literal

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
    clone_state,
    load_json,
    motif_from_spec,
    normalized_l2,
    write_csv,
    write_json,
    write_jsonl,
)

ControlPersistenceCondition = Literal["control_preserved", "control_reset", "control_disabled"]


class FlexibleSchema(BaseModel):
    """Validate expected fields while preserving searchable extras."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class ControlTraceSummary(FlexibleSchema):
    candidate_id: str
    condition: str
    seed: int
    perturb_scale: float
    motif_family: str
    motif_index: int
    perturb_step: int
    pre_control_norm_mean: float
    shock_control_norm_mean: float
    tail_control_norm_mean: float
    tail_minus_pre_control_norm: float
    step_mean_std: float
    vector_supported: bool
    tail_to_pre_control_distance: float | None = None
    shock_to_pre_control_distance: float | None = None
    tail_to_pre_control_cosine: float | None = None


class VectorControlTraceRow(FlexibleSchema):
    candidate_id: str
    condition: str
    seed: int
    perturb_scale: float
    motif_family: str
    motif_index: int
    step: NonNegativeInt
    fast_state_vector: list[float]
    slow_state_vector: list[float]
    control_state_vector: list[float]
    message_state_vector: list[float]
    carrier_state_vector: list[float]
    channel_state_vectors: dict[str, list[float]] = Field(default_factory=dict)

    @field_validator(
        "fast_state_vector",
        "slow_state_vector",
        "control_state_vector",
        "message_state_vector",
        "carrier_state_vector",
    )
    @classmethod
    def vectors_are_finite(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("state vectors must contain only finite values")
        return values

    @model_validator(mode="after")
    def populate_channel_vectors(self) -> "VectorControlTraceRow":
        if not self.channel_state_vectors:
            self.channel_state_vectors = {
                "fast": self.fast_state_vector,
                "slow": self.slow_state_vector,
                "control": self.control_state_vector,
                "message": self.message_state_vector,
                "carrier": self.carrier_state_vector,
            }
        return self


def tensor_to_vector(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.view(-1).detach().cpu().tolist()]


def vector_row_from_step(
    *,
    candidate_id: str,
    condition: str,
    seed: int,
    perturb_scale: float,
    motif_family: str,
    motif_index: int,
    step: int,
    components: dict[str, torch.Tensor],
) -> VectorControlTraceRow:
    vectors = {channel: tensor_to_vector(components[channel]) for channel in CHANNELS}
    return VectorControlTraceRow(
        candidate_id=candidate_id,
        condition=condition,
        seed=int(seed),
        perturb_scale=float(perturb_scale),
        motif_family=motif_family,
        motif_index=int(motif_index),
        step=int(step),
        fast_state_vector=vectors["fast"],
        slow_state_vector=vectors["slow"],
        control_state_vector=vectors["control"],
        message_state_vector=vectors["message"],
        carrier_state_vector=vectors["carrier"],
        channel_state_vectors=vectors,
    )


def load_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"expected JSON object row in {path}")
                rows.append(payload)
    return rows


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(mean(finite)) if finite else 0.0


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(left_array, right_array) / denom)


def vector_distance(left: list[float], right: list[float]) -> float:
    left_tensor = torch.tensor(left, dtype=torch.float32)
    right_tensor = torch.tensor(right, dtype=torch.float32)
    return normalized_l2(left_tensor, right_tensor)


def mean_vector(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        return None
    return [float(value) for value in np.asarray(vectors, dtype=np.float64).mean(axis=0).tolist()]


def window_values(rows: list[dict[str, object]], start: int, end: int, key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        step = int(row.get("step", 0))
        if start <= step <= end and key in row:
            values.append(float(row[key]))
    return values


def window_vectors(rows: list[dict[str, object]], start: int, end: int) -> list[list[float]]:
    values: list[list[float]] = []
    for row in rows:
        step = int(row.get("step", 0))
        vector = row.get("control_state_vector")
        if start <= step <= end and isinstance(vector, list):
            values.append([float(item) for item in vector])
    return values


def summarize_control_trace_rows(
    path: Path,
    *,
    perturb_steps: list[int],
    steps: int,
    pre_window: int = 32,
    shock_window: int = 8,
    tail_window: int = 32,
) -> list[ControlTraceSummary]:
    rows = load_jsonl_rows(path)
    grouped: dict[tuple[str, str, int, float, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("candidate_id", "")),
                str(row.get("condition", "")),
                int(row.get("seed", 0)),
                float(row.get("perturb_scale", 0.0)),
                str(row.get("motif_family", "")),
                int(row.get("motif_index", 0)),
            )
        ].append(row)

    summaries: list[ControlTraceSummary] = []
    for key, group_rows in sorted(grouped.items()):
        for perturb_step in perturb_steps:
            pre_start = max(1, int(perturb_step) - int(pre_window))
            pre_end = max(1, int(perturb_step) - 1)
            shock_start = int(perturb_step)
            shock_end = min(int(steps), int(perturb_step) + int(shock_window))
            tail_start = max(1, int(steps) - int(tail_window) + 1)
            tail_end = int(steps)
            pre_norm = finite_mean(window_values(group_rows, pre_start, pre_end, "control_state_norm"))
            shock_norm = finite_mean(window_values(group_rows, shock_start, shock_end, "control_state_norm"))
            tail_norm = finite_mean(window_values(group_rows, tail_start, tail_end, "control_state_norm"))
            step_means = [
                finite_mean(window_values(group_rows, step, step, "control_state_norm"))
                for step in range(1, int(steps) + 1)
                if window_values(group_rows, step, step, "control_state_norm")
            ]
            pre_vector = mean_vector(window_vectors(group_rows, pre_start, pre_end))
            shock_vector = mean_vector(window_vectors(group_rows, shock_start, shock_end))
            tail_vector = mean_vector(window_vectors(group_rows, tail_start, tail_end))
            vector_supported = pre_vector is not None and shock_vector is not None and tail_vector is not None
            summaries.append(
                ControlTraceSummary(
                    candidate_id=key[0],
                    condition=key[1],
                    seed=key[2],
                    perturb_scale=key[3],
                    motif_family=key[4],
                    motif_index=key[5],
                    perturb_step=int(perturb_step),
                    pre_control_norm_mean=pre_norm,
                    shock_control_norm_mean=shock_norm,
                    tail_control_norm_mean=tail_norm,
                    tail_minus_pre_control_norm=float(tail_norm - pre_norm),
                    step_mean_std=float(pstdev(step_means)) if len(step_means) > 1 else 0.0,
                    vector_supported=vector_supported,
                    tail_to_pre_control_distance=vector_distance(tail_vector, pre_vector)
                    if vector_supported and tail_vector is not None and pre_vector is not None
                    else None,
                    shock_to_pre_control_distance=vector_distance(shock_vector, pre_vector)
                    if vector_supported and shock_vector is not None and pre_vector is not None
                    else None,
                    tail_to_pre_control_cosine=cosine_similarity(tail_vector, pre_vector)
                    if vector_supported and tail_vector is not None and pre_vector is not None
                    else None,
                )
            )
    return summaries


def flatten_models(rows: list[BaseModel]) -> list[dict[str, object]]:
    return [row.model_dump(mode="json") for row in rows]


def run_control_cycle(
    genome: dict[str, object],
    *,
    candidate_id: str,
    config: CharacterizationConfig,
    condition: ControlPersistenceCondition,
    seed: int,
    perturb_scale: float,
    perturb_steps: tuple[int, int],
    motif_specs: tuple[MotifSpec, MotifSpec],
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(genome, config=config, condition="original")
    state = model.initial_state(1, torch.device(config.device))
    reset_control_state = state[2].detach().clone()
    motifs = [motif_from_spec(spec, int(config.hidden_size), int(seed)) for spec in motif_specs]
    rows: list[VectorControlTraceRow] = []
    surfaces: list[torch.Tensor] = []
    control_at_perturb: list[torch.Tensor] = []
    control_after_recovery: list[torch.Tensor] = []
    perturb_map = {int(perturb_steps[0]): motifs[0], int(perturb_steps[1]): motifs[1]}
    recovery_lag = max(1, min(16, int(config.steps) // 8))
    with torch.no_grad():
        for step in range(1, int(config.steps) + 1):
            if step in perturb_map:
                if condition == "control_reset":
                    values = list(state)
                    values[2] = reset_control_state.detach().clone()
                    state = tuple(values)
                state = apply_controlled_perturbation(
                    state,
                    perturb_map[step]["vector"],
                    float(perturb_scale),
                    channel=config.perturb_channel,
                )
                model.record_perturbation(step, float(perturb_scale))
                control_at_perturb.append(state[2].view(-1).detach().float().cpu())
            state = model.step(state)
            if condition == "control_disabled":
                state = clamp_channel_state(state, "control")
            components = {
                channel: tensor.view(-1).detach().float().cpu()
                for channel, tensor in model.state_components(state).items()
            }
            surfaces.append(model.state_vector(state).view(-1).detach().float().cpu())
            rows.append(
                vector_row_from_step(
                    candidate_id=candidate_id,
                    condition=condition,
                    seed=seed,
                    perturb_scale=perturb_scale,
                    motif_family="cycle",
                    motif_index=0,
                    step=step,
                    components=components,
                )
            )
            if step in {int(perturb_steps[0]) + recovery_lag, int(perturb_steps[1]) + recovery_lag}:
                control_after_recovery.append(components["control"].clone())
    recovery_distance = 0.0
    recovery_cosine = 0.0
    if len(control_after_recovery) >= 2:
        recovery_distance = normalized_l2(control_after_recovery[0], control_after_recovery[1])
        denom = float(control_after_recovery[0].norm() * control_after_recovery[1].norm())
        recovery_cosine = (
            float(torch.dot(control_after_recovery[0], control_after_recovery[1]) / denom) if denom > 1e-12 else 0.0
        )
    return {
        "condition": condition,
        "candidate_id": candidate_id,
        "seed": int(seed),
        "perturb_scale": float(perturb_scale),
        "rows": rows,
        "summary": {
            "condition": condition,
            "final_surface_norm": float(surfaces[-1].norm()) / math.sqrt(max(surfaces[-1].numel(), 1)),
            "control_recovery_distance": float(recovery_distance),
            "control_recovery_cosine": float(recovery_cosine),
            "recorded_perturbations": len(control_at_perturb),
        },
    }


def run_control_persistence_probe(
    genome: dict[str, object],
    *,
    candidate_id: str,
    config: CharacterizationConfig,
    seed: int,
    perturb_scale: float,
    perturb_steps: tuple[int, int] = (32, 64),
    motif_specs: tuple[MotifSpec, MotifSpec] = (MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)),
) -> dict[str, object]:
    runs = [
        run_control_cycle(
            genome,
            candidate_id=candidate_id,
            config=config,
            condition=condition,
            seed=seed,
            perturb_scale=perturb_scale,
            perturb_steps=perturb_steps,
            motif_specs=motif_specs,
        )
        for condition in ("control_preserved", "control_reset", "control_disabled")
    ]
    summary = [dict(run["summary"]) for run in runs]
    rows: list[VectorControlTraceRow] = []
    for run in runs:
        rows.extend(run["rows"])  # type: ignore[arg-type]
    return {
        "candidate_id": candidate_id,
        "run_count": len(runs),
        "summary": summary,
        "rows": rows,
    }


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_motifs(text: str) -> tuple[MotifSpec, MotifSpec]:
    values: list[MotifSpec] = []
    for item in text.split(","):
        if not item.strip():
            continue
        family, _, index = item.partition(":")
        values.append(MotifSpec(family=family.strip(), index=int(index or 0)))
    if len(values) != 2:
        raise ValueError("exactly two motifs are required for the persistence probe")
    return values[0], values[1]


def write_control_trace_summary(input_path: Path, output_dir: Path, perturb_steps: list[int], steps: int) -> None:
    summaries = summarize_control_trace_rows(input_path, perturb_steps=perturb_steps, steps=steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "control_trace_summary.json", [row.model_dump(mode="json") for row in summaries])
    write_csv(output_dir / "control_trace_summary.csv", flatten_models(summaries))


def write_persistence_probe(
    candidate_path: Path,
    output_dir: Path,
    *,
    seed: int,
    perturb_scale: float,
    perturb_steps: tuple[int, int],
    motif_specs: tuple[MotifSpec, MotifSpec],
    hidden_size: int,
    rank: int,
    steps: int,
    device: str,
) -> None:
    candidate = load_json(candidate_path)
    candidate_id = str(candidate.get("id", candidate_path.stem))
    config = CharacterizationConfig(
        candidate_path=candidate_path,
        archive_root=candidate_path.parent.parent,
        output_dir=output_dir,
        seeds=[seed],
        perturb_scales=[perturb_scale],
        motifs=list(motif_specs),
        hidden_size=hidden_size,
        rank=rank,
        steps=steps,
        perturb_step=perturb_steps[0],
        capsule_pause_step=perturb_steps[0],
        device=device,
    )
    result = run_control_persistence_probe(
        candidate["genome"],
        candidate_id=candidate_id,
        config=config,
        seed=seed,
        perturb_scale=perturb_scale,
        perturb_steps=perturb_steps,
        motif_specs=motif_specs,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    assert isinstance(rows, list)
    write_jsonl(output_dir / "control_persistence_rows.jsonl", rows)
    write_json(output_dir / "control_persistence_summary.json", {k: v for k, v in result.items() if k != "rows"})
    write_csv(output_dir / "control_persistence_summary.csv", result["summary"])  # type: ignore[arg-type]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize-traces")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument("--perturb-steps", default="64")
    summarize.add_argument("--steps", type=int, default=128)

    persistence = subparsers.add_parser("persistence-probe")
    persistence.add_argument("--candidate-path", type=Path, required=True)
    persistence.add_argument("--output-dir", type=Path, required=True)
    persistence.add_argument("--seed", type=int, default=94)
    persistence.add_argument("--perturb-scale", type=float, default=0.35)
    persistence.add_argument("--perturb-steps", default="32,64")
    persistence.add_argument("--motifs", default="basis:0,gaussian:0")
    persistence.add_argument("--hidden-size", type=int, default=32)
    persistence.add_argument("--rank", type=int, default=2)
    persistence.add_argument("--steps", type=int, default=128)
    persistence.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "summarize-traces":
        write_control_trace_summary(
            args.input,
            args.output_dir,
            parse_csv_ints(args.perturb_steps),
            int(args.steps),
        )
        print(f"wrote {args.output_dir / 'control_trace_summary.json'}")
        return
    if args.command == "persistence-probe":
        perturb_steps = parse_csv_ints(args.perturb_steps)
        if len(perturb_steps) != 2:
            raise ValueError("exactly two perturbation steps are required")
        write_persistence_probe(
            args.candidate_path,
            args.output_dir,
            seed=int(args.seed),
            perturb_scale=float(args.perturb_scale),
            perturb_steps=(perturb_steps[0], perturb_steps[1]),
            motif_specs=parse_motifs(args.motifs),
            hidden_size=int(args.hidden_size),
            rank=int(args.rank),
            steps=int(args.steps),
            device=str(args.device),
        )
        print(f"wrote {args.output_dir / 'control_persistence_summary.json'}")
        return
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
