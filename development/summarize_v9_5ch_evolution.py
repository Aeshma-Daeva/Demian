#!/usr/bin/env python3
"""Summarize v9 five-channel evolution island outputs.

This turns per-island candidate and diagnostics files into compact summaries,
including the v10.0 predecessor evidence used by Demian v1 planning. It does
not rerun evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


V9_3_REFERENCE = {
    "duty": 0.1992,
    "event": 0.0844,
    "phase": 1.4314,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sd(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def metric(candidate: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = candidate.get("metrics", {}).get(name, default)
    return float(value) if isinstance(value, int | float) else default


def candidate_brief(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics", {})
    generation = int(candidate.get("generation", 0))
    generation_of_origin = candidate.get("generation_of_origin", generation)
    lineage_age = generation - int(generation_of_origin) if generation_of_origin is not None else None
    return {
        "island": candidate.get("_island"),
        "id": candidate.get("id"),
        "generation": candidate.get("generation"),
        "rank_score": candidate.get("rank_score"),
        "release_duty_cycle": metrics.get("release_duty_cycle"),
        "release_geometric_event": metrics.get("release_geometric_event"),
        "phase_transition_score": metrics.get("phase_transition_score"),
        "internal_richness": metrics.get("internal_richness"),
        "channel_separation": metrics.get("channel_separation"),
        "mathematical_curiosity": metrics.get("mathematical_curiosity"),
        "geometric_coherence": metrics.get("geometric_coherence"),
        "regimes": metrics.get("regimes"),
        "ancestor_ids": candidate.get("ancestor_ids"),
        "generation_of_origin": generation_of_origin,
        "mutation_count": candidate.get("mutation_count"),
        "parent_ids": candidate.get("parent_ids"),
        "reproduction_kind": candidate.get("reproduction_kind"),
        "lineage_age": lineage_age,
    }


def duty_histogram(duties: list[float]) -> dict[str, int]:
    return {
        "zero": sum(1 for value in duties if value == 0),
        "gt_0_le_0.06": sum(1 for value in duties if 0 < value <= 0.06),
        "gt_0.06_le_0.14": sum(1 for value in duties if 0.06 < value <= 0.14),
        "gt_0.14_le_0.18": sum(1 for value in duties if 0.14 < value <= 0.18),
        "gt_0.18_le_0.30": sum(1 for value in duties if 0.18 < value <= 0.30),
        "gt_0.30_le_0.60": sum(1 for value in duties if 0.30 < value <= 0.60),
        "gt_0.60": sum(1 for value in duties if value > 0.60),
    }


def event_phase_bins(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "low_event_high_phase": sum(
            1 for c in candidates if metric(c, "release_geometric_event") < 0.25 and metric(c, "phase_transition_score") >= 1.0
        ),
        "high_event_high_phase": sum(
            1 for c in candidates if metric(c, "release_geometric_event") >= 0.25 and metric(c, "phase_transition_score") >= 1.0
        ),
        "high_event_low_phase": sum(
            1 for c in candidates if metric(c, "release_geometric_event") >= 0.25 and metric(c, "phase_transition_score") < 1.0
        ),
        "low_event_low_phase": sum(
            1 for c in candidates if metric(c, "release_geometric_event") < 0.25 and metric(c, "phase_transition_score") < 1.0
        ),
    }


def distance_to_reference(candidate: dict[str, Any], reference: dict[str, float]) -> float:
    duty = (metric(candidate, "release_duty_cycle") - reference["duty"]) / 0.20
    event = (metric(candidate, "release_geometric_event") - reference["event"]) / 0.25
    phase = (metric(candidate, "phase_transition_score") - reference["phase"]) / 1.50
    return math.sqrt(duty * duty + event * event + phase * phase)


def load_candidates(roots: list[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, root in enumerate(roots, start=1):
        for path in sorted((root / "candidates").glob("*.json")):
            candidate = load_json(path)
            candidate["_island"] = f"island_{index}"
            candidates.append(candidate)
    return candidates


def summarize(candidates: list[dict[str, Any]], experiment: str, eval_seed: int | None) -> dict[str, Any]:
    generations = sorted({int(candidate["generation"]) for candidate in candidates})
    per_generation: list[dict[str, Any]] = []
    for generation in generations:
        generation_candidates = [c for c in candidates if int(c["generation"]) == generation]
        ranked = sorted(generation_candidates, key=lambda c: float(c.get("rank_score", 0.0)), reverse=True)
        top10 = ranked[:10]
        top20 = ranked[:20]
        duties = [metric(c, "release_duty_cycle") for c in generation_candidates]
        events = [metric(c, "release_geometric_event") for c in generation_candidates]
        phases = [metric(c, "phase_transition_score") for c in generation_candidates]
        regime_counts: dict[str, int] = {}
        for candidate in generation_candidates:
            for regime, count in candidate.get("metrics", {}).get("regimes", {}).items():
                regime_counts[regime] = regime_counts.get(regime, 0) + int(count)
        top10_ancestors = sorted({a for c in top10 for a in c.get("ancestor_ids", [])})
        per_generation.append(
            {
                "generation": generation,
                "candidate_count": len(generation_candidates),
                "mean_duty": mean(duties),
                "sd_duty": sd(duties),
                "median_duty": quantile(duties, 0.5),
                "mean_event": mean(events),
                "sd_event": sd(events),
                "median_event": quantile(events, 0.5),
                "mean_phase": mean(phases),
                "sd_phase": sd(phases),
                "median_phase": quantile(phases, 0.5),
                "duty_histogram": duty_histogram(duties),
                "regime_counts": regime_counts,
                "top10_distinct_ancestor_count": len(top10_ancestors),
                "top10_ancestor_ids": top10_ancestors,
                "top10_mean_mutation_count": mean([float(c.get("mutation_count", 0)) for c in top10]),
                "top10_max_mutation_count": max((int(c.get("mutation_count", 0)) for c in top10), default=0),
                "top10_mean_lineage_age": mean(
                    [
                        float(int(c.get("generation", 0)) - int(c.get("generation_of_origin", c.get("generation", 0))))
                        for c in top10
                    ]
                ),
                "top10_max_lineage_age": max(
                    (
                        int(c.get("generation", 0)) - int(c.get("generation_of_origin", c.get("generation", 0)))
                        for c in top10
                    ),
                    default=0,
                ),
                "top20_event_phase_bins": event_phase_bins(top20),
                "best": candidate_brief(ranked[0]),
                "top5": [candidate_brief(c) for c in ranked[:5]],
            }
        )

    first5 = per_generation[:5]
    last5 = per_generation[-5:]
    duty_first5 = mean([g["mean_duty"] for g in first5])
    duty_last5 = mean([g["mean_duty"] for g in last5])
    event_first5 = mean([g["mean_event"] for g in first5])
    event_last5 = mean([g["mean_event"] for g in last5])
    phase_first5 = mean([g["mean_phase"] for g in first5])
    phase_last5 = mean([g["mean_phase"] for g in last5])
    final_generation = generations[-1]
    final_candidates = [c for c in candidates if int(c["generation"]) == final_generation]
    final_best = max(final_candidates, key=lambda c: float(c.get("rank_score", 0.0)))
    best_overall = max(candidates, key=lambda c: float(c.get("rank_score", 0.0)))
    nearest_final = min(final_candidates, key=lambda c: distance_to_reference(c, V9_3_REFERENCE))
    nearest_overall = min(candidates, key=lambda c: distance_to_reference(c, V9_3_REFERENCE))
    return {
        "experiment": experiment,
        "eval_seed": eval_seed,
        "candidate_count": len(candidates),
        "generation_count": len(generations),
        "final_generation": final_generation,
        "best_overall": candidate_brief(best_overall),
        "final_best": candidate_brief(final_best),
        "v9_3_reference_target_used_for_similarity": V9_3_REFERENCE,
        "nearest_final_to_v9_3_sparseish_phenotype": {
            "distance": distance_to_reference(nearest_final, V9_3_REFERENCE),
            **candidate_brief(nearest_final),
        },
        "nearest_overall_to_v9_3_sparseish_phenotype": {
            "distance": distance_to_reference(nearest_overall, V9_3_REFERENCE),
            **candidate_brief(nearest_overall),
        },
        "curve_summary": {
            "duty_first5_mean": duty_first5,
            "duty_last5_mean": duty_last5,
            "event_first5_mean": event_first5,
            "event_last5_mean": event_last5,
            "phase_first5_mean": phase_first5,
            "phase_last5_mean": phase_last5,
            "duty_delta_last5_minus_first5": duty_last5 - duty_first5,
            "event_delta_last5_minus_first5": event_last5 - event_first5,
            "phase_delta_last5_minus_first5": phase_last5 - phase_first5,
            "duty_range": [min(g["mean_duty"] for g in per_generation), max(g["mean_duty"] for g in per_generation)],
            "event_range": [min(g["mean_event"] for g in per_generation), max(g["mean_event"] for g in per_generation)],
            "phase_range": [min(g["mean_phase"] for g in per_generation), max(g["mean_phase"] for g in per_generation)],
        },
        "per_generation": per_generation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize v9 five-channel evolution island outputs")
    parser.add_argument("roots", nargs="+", help="island output directories")
    parser.add_argument("--experiment", default="v9-five-channel-evolution")
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--out", required=True, help="summary JSON path")
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    candidates = load_candidates(roots)
    if not candidates:
        raise SystemExit("no candidate JSON files found under supplied roots")
    summary = summarize(candidates, args.experiment, args.eval_seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    print(f"candidates={summary['candidate_count']} generations={summary['generation_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
