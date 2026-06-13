#!/usr/bin/env python3
"""Matched control/null comparison for the gate-state mechanism gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolve_v9_5ch_release import default_genome, random_genome
from development.gate_state_propagation_characterization import (
    DEFAULT_CONDITIONS,
    CharacterizationConfig,
    MotifSpec,
    parse_csv_floats,
    parse_csv_ints,
    parse_motifs,
    run_characterization,
    write_csv,
    write_json,
)

DEFAULT_REPLICATION_SUMMARY = Path("data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json")
DEFAULT_OUTPUT_DIR = Path("data/diagnostics/gate_state_control_nulls_20260515")

ControlFamily = Literal["track_b_positive", "random_unselected", "default_unselected", "default_jittered", "baseline"]


class FlexibleSchema(BaseModel):
    """Validate expected fields while preserving metadata extras."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class ControlNullConfig(FlexibleSchema):
    replication_summary_path: Path = DEFAULT_REPLICATION_SUMMARY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seeds: list[int] = Field(default_factory=lambda: [96, 97, 98])
    perturb_scales: list[float] = Field(default_factory=lambda: [0.35])
    motifs: list[MotifSpec] = Field(default_factory=lambda: [MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)])
    steps: NonNegativeInt = 128
    perturb_step: NonNegativeInt = 64
    capsule_pause_step: NonNegativeInt = 64
    hidden_size: NonNegativeInt = 32
    rank: NonNegativeInt = 2
    random_control_count: NonNegativeInt = 8
    random_seed: int = 5152026
    include_default_control: bool = True
    device: str = "cpu"
    conditions: list[str] = Field(default_factory=lambda: list(DEFAULT_CONDITIONS))


class CandidateSpec(FlexibleSchema):
    candidate_id: str
    candidate_path: Path
    family: ControlFamily


class CandidateGateRow(FlexibleSchema):
    candidate_id: str
    candidate_path: str
    family: ControlFamily
    gain_zero_clean: bool
    routes_disabled_divergence_positive: bool
    gain_zero_divergence_positive: bool
    full_resume_exact: bool
    surface_resume_gap_positive: bool
    message_carrier_top2: bool
    mechanism_gate_pass: bool
    strict_profile_gate_pass: bool = False
    route_divergence_mean: float
    gain_zero_divergence_mean: float
    surface_resume_gap_mean: float
    message_disabled_divergence_mean: float
    carrier_disabled_divergence_mean: float
    slow_disabled_divergence_mean: float
    control_disabled_divergence_mean: float
    message_carrier_dominance: float
    channel_clamping_order: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def positive_candidates_from_replication_summary(summary_path: Path) -> list[CandidateSpec]:
    payload = load_json(summary_path)
    rows = payload.get("rows", [])
    candidates = []
    for row in rows:
        candidates.append(
            CandidateSpec(
                candidate_id=str(row["top_candidate_id"]),
                candidate_path=Path(str(row["top_candidate_path"])),
                family="track_b_positive",
            )
        )
    return candidates


def write_candidate(path: Path, *, candidate_id: str, genome: dict[str, Any], family: ControlFamily, seed: int | None) -> None:
    payload = {
        "id": candidate_id,
        "generation": 0,
        "index": 0,
        "rank_mode": "control_null",
        "control_family": family,
        "control_seed": seed,
        "genome": genome,
        "metrics": {},
        "reproduction_kind": "unselected_control",
        "parent_ids": [],
        "ancestor_ids": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_control_candidates(config: ControlNullConfig) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    control_dir = config.output_dir / "control_candidates"
    if config.include_default_control:
        candidate_id = "default_unselected_control"
        path = control_dir / f"{candidate_id}.json"
        write_candidate(
            path,
            candidate_id=candidate_id,
            genome=default_genome(int(config.hidden_size), int(config.rank)),
            family="default_unselected",
            seed=None,
        )
        candidates.append(CandidateSpec(candidate_id=candidate_id, candidate_path=path, family="default_unselected"))

    rng = random.Random(int(config.random_seed))
    for idx in range(int(config.random_control_count)):
        seed = rng.randrange(1_000_000_000)
        genome = random_genome(random.Random(seed), int(config.hidden_size), int(config.rank))
        candidate_id = f"random_unselected_control_{idx:03d}"
        path = control_dir / f"{candidate_id}.json"
        write_candidate(path, candidate_id=candidate_id, genome=genome, family="random_unselected", seed=seed)
        candidates.append(CandidateSpec(candidate_id=candidate_id, candidate_path=path, family="random_unselected"))
    return candidates


def row_for_summary(candidate: CandidateSpec, summary: Any) -> CandidateGateRow:
    evidence = dict(summary.evidence_gate)
    by_condition = {str(row["condition"]): row for row in summary.ablation_summary}
    by_resume = {str(row["resume_kind"]): row for row in summary.capsule_summary}
    order = list(summary.channel_necessity_order)
    top2 = set(order[:2]) == {"message", "carrier"}
    pass_gate = bool(
        evidence.get("gain_zero_clean")
        and evidence.get("routes_disabled_divergence_positive")
        and evidence.get("gain_zero_divergence_positive")
        and evidence.get("full_resume_exact")
        and evidence.get("surface_resume_gap_positive")
        and top2
    )
    routes = by_condition.get("routes_disabled", {})
    gain_zero = by_condition.get("gain_zero", {})
    surface = by_resume.get("surface_only", {})
    message = by_condition.get("message_disabled", {})
    carrier = by_condition.get("carrier_disabled", {})
    slow = by_condition.get("slow_disabled", {})
    control = by_condition.get("control_disabled", {})
    message_divergence = float(message.get("causal_divergence_mean", 0.0))
    carrier_divergence = float(carrier.get("causal_divergence_mean", 0.0))
    slow_divergence = float(slow.get("causal_divergence_mean", 0.0))
    control_divergence = float(control.get("causal_divergence_mean", 0.0))
    message_carrier_dominance = 0.5 * (message_divergence + carrier_divergence) - 0.5 * (
        slow_divergence + control_divergence
    )
    return CandidateGateRow(
        candidate_id=candidate.candidate_id,
        candidate_path=str(candidate.candidate_path),
        family=candidate.family,
        gain_zero_clean=bool(evidence.get("gain_zero_clean", False)),
        routes_disabled_divergence_positive=bool(evidence.get("routes_disabled_divergence_positive", False)),
        gain_zero_divergence_positive=bool(evidence.get("gain_zero_divergence_positive", False)),
        full_resume_exact=bool(evidence.get("full_resume_exact", False)),
        surface_resume_gap_positive=bool(evidence.get("surface_resume_gap_positive", False)),
        message_carrier_top2=top2,
        mechanism_gate_pass=pass_gate,
        route_divergence_mean=float(routes.get("causal_divergence_mean", 0.0)),
        gain_zero_divergence_mean=float(gain_zero.get("causal_divergence_mean", 0.0)),
        surface_resume_gap_mean=float(surface.get("final_l2_gap_mean", 0.0)),
        message_disabled_divergence_mean=message_divergence,
        carrier_disabled_divergence_mean=carrier_divergence,
        slow_disabled_divergence_mean=slow_divergence,
        control_disabled_divergence_mean=control_divergence,
        message_carrier_dominance=message_carrier_dominance,
        channel_clamping_order=",".join(order),
    )


def calibrated_strict_thresholds(rows: list[CandidateGateRow]) -> dict[str, float]:
    positives = [row for row in rows if row.family == "track_b_positive" and row.mechanism_gate_pass]
    if not positives:
        return {
            "min_route_divergence": float("inf"),
            "min_gain_zero_divergence": float("inf"),
            "min_message_carrier_dominance": float("inf"),
        }
    return {
        "min_route_divergence": min(row.route_divergence_mean for row in positives),
        "min_gain_zero_divergence": min(row.gain_zero_divergence_mean for row in positives),
        "min_message_carrier_dominance": min(row.message_carrier_dominance for row in positives),
    }


def apply_strict_profile_gate(rows: list[CandidateGateRow], thresholds: dict[str, float]) -> list[CandidateGateRow]:
    updated = []
    for row in rows:
        strict = bool(
            row.mechanism_gate_pass
            and row.route_divergence_mean >= thresholds["min_route_divergence"]
            and row.gain_zero_divergence_mean >= thresholds["min_gain_zero_divergence"]
            and row.message_carrier_dominance >= thresholds["min_message_carrier_dominance"]
        )
        updated.append(row.model_copy(update={"strict_profile_gate_pass": strict}))
    return updated


def characterization_config_for(candidate: CandidateSpec, config: ControlNullConfig) -> CharacterizationConfig:
    return CharacterizationConfig(
        candidate_path=candidate.candidate_path,
        archive_root=candidate.candidate_path.parent.parent,
        output_dir=config.output_dir / "characterizations" / candidate.family / candidate.candidate_id,
        seeds=list(config.seeds),
        perturb_scales=list(config.perturb_scales),
        motifs=list(config.motifs),
        conditions=list(config.conditions),  # type: ignore[arg-type]
        steps=int(config.steps),
        perturb_step=int(config.perturb_step),
        capsule_pause_step=int(config.capsule_pause_step),
        hidden_size=int(config.hidden_size),
        rank=int(config.rank),
        device=config.device,
        write_parameter_signatures=False,
    )


def aggregate_rows(rows: list[CandidateGateRow]) -> list[dict[str, Any]]:
    families = sorted({row.family for row in rows})
    output = []
    for family in families:
        items = [row for row in rows if row.family == family]
        output.append(
            {
                "family": family,
                "candidate_count": len(items),
                "mechanism_gate_pass_count": sum(row.mechanism_gate_pass for row in items),
                "mechanism_gate_pass_rate": mean([1.0 if row.mechanism_gate_pass else 0.0 for row in items]),
                "strict_profile_gate_pass_count": sum(row.strict_profile_gate_pass for row in items),
                "strict_profile_gate_pass_rate": mean([1.0 if row.strict_profile_gate_pass else 0.0 for row in items]),
                "message_carrier_top2_count": sum(row.message_carrier_top2 for row in items),
                "message_carrier_top2_rate": mean([1.0 if row.message_carrier_top2 else 0.0 for row in items]),
                "mean_route_divergence": mean([row.route_divergence_mean for row in items]),
                "mean_gain_zero_divergence": mean([row.gain_zero_divergence_mean for row in items]),
                "mean_surface_resume_gap": mean([row.surface_resume_gap_mean for row in items]),
                "mean_message_carrier_dominance": mean([row.message_carrier_dominance for row in items]),
                "std_message_carrier_dominance": pstdev([row.message_carrier_dominance for row in items])
                if len(items) > 1
                else 0.0,
            }
        )
    return output


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p-value for higher pass rate in row 1."""

    total = a + b + c + d
    row1 = a + b
    col1 = a + c
    lower = max(0, row1 - (total - col1))
    upper = min(row1, col1)
    denominator = math.comb(total, row1)
    if denominator == 0:
        return 1.0
    return float(
        sum(
            math.comb(col1, value) * math.comb(total - col1, row1 - value) / denominator
            for value in range(max(a, lower), upper + 1)
        )
    )


def pass_rate_comparisons(rows: list[CandidateGateRow]) -> list[dict[str, Any]]:
    comparisons = []
    for gate_name in ("mechanism_gate_pass", "strict_profile_gate_pass"):
        positives = [row for row in rows if row.family == "track_b_positive"]
        for control_name, controls in (
            ("random_unselected", [row for row in rows if row.family == "random_unselected"]),
            ("all_unselected", [row for row in rows if row.family != "track_b_positive"]),
        ):
            positive_pass = sum(bool(getattr(row, gate_name)) for row in positives)
            control_pass = sum(bool(getattr(row, gate_name)) for row in controls)
            positive_total = len(positives)
            control_total = len(controls)
            positive_rate = positive_pass / positive_total if positive_total else 0.0
            control_rate = control_pass / control_total if control_total else 0.0
            comparisons.append(
                {
                    "gate": gate_name,
                    "control_family": control_name,
                    "track_b_pass": positive_pass,
                    "track_b_total": positive_total,
                    "control_pass": control_pass,
                    "control_total": control_total,
                    "track_b_pass_rate": positive_rate,
                    "control_pass_rate": control_rate,
                    "enrichment_factor": positive_rate / control_rate if control_rate > 0.0 else None,
                    "fisher_exact_greater_p": fisher_exact_greater(
                        positive_pass,
                        positive_total - positive_pass,
                        control_pass,
                        control_total - control_pass,
                    ),
                }
            )
    return comparisons


def run_control_null_comparison(config: ControlNullConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "config.json", config.model_dump(mode="json"))
    candidates = positive_candidates_from_replication_summary(config.replication_summary_path)
    candidates.extend(generate_control_candidates(config))

    rows: list[CandidateGateRow] = []
    for candidate in candidates:
        summary = run_characterization(characterization_config_for(candidate, config))
        rows.append(row_for_summary(candidate, summary))
    thresholds = calibrated_strict_thresholds(rows)
    rows = apply_strict_profile_gate(rows, thresholds)

    row_dicts = [row.model_dump(mode="json") for row in rows]
    aggregate = aggregate_rows(rows)
    comparisons = pass_rate_comparisons(rows)
    payload = {
        "experiment": "gate_state_control_nulls_20260515",
        "candidate_count": len(rows),
        "config": config.model_dump(mode="json"),
        "strict_profile_gate": {
            "thresholds": thresholds,
            "definition": (
                "The strict profile gate requires the original binary gate, route/gain-zero divergence "
                "at least as large as the weakest Track B positive, and message/carrier clamping "
                "dominance at least as large as the weakest Track B positive."
            ),
        },
        "aggregate": aggregate,
        "pass_rate_comparisons": comparisons,
        "rows": row_dicts,
        "interpretation": (
            "Matched mechanism-gate comparison. Track B positives and unselected controls "
            "use the same route-disabled, gain-zero, channel-clamping, and capsule criteria."
        ),
    }
    write_csv(config.output_dir / "control_null_rows.csv", row_dicts)
    write_csv(config.output_dir / "control_null_aggregate.csv", aggregate)
    write_json(config.output_dir / "control_null_summary.json", payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-summary-path", type=Path, default=DEFAULT_REPLICATION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="96,97,98")
    parser.add_argument("--perturb-scales", default="0.35")
    parser.add_argument("--motifs", default="basis:0,gaussian:0")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--perturb-step", type=int, default=64)
    parser.add_argument("--capsule-pause-step", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--random-control-count", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=5152026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny grid and one random control.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = parse_csv_ints(args.seeds)
    scales = parse_csv_floats(args.perturb_scales)
    motifs = parse_motifs(args.motifs)
    steps = args.steps
    perturb_step = args.perturb_step
    pause = args.capsule_pause_step
    random_count = args.random_control_count
    if args.smoke:
        seeds = seeds[:1]
        scales = scales[:1]
        motifs = motifs[:1]
        steps = min(steps, 12)
        perturb_step = min(perturb_step, 4)
        pause = min(pause, 4)
        random_count = 1

    payload = run_control_null_comparison(
        ControlNullConfig(
            replication_summary_path=args.replication_summary_path,
            output_dir=args.output_dir,
            seeds=seeds,
            perturb_scales=scales,
            motifs=motifs,
            steps=steps,
            perturb_step=perturb_step,
            capsule_pause_step=pause,
            hidden_size=args.hidden_size,
            rank=args.rank,
            random_control_count=random_count,
            random_seed=args.random_seed,
            device=args.device,
        )
    )
    print(f"wrote {args.output_dir / 'control_null_summary.json'}")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
