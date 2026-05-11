"""Generation-level diagnostics for release evolution."""

from __future__ import annotations

from statistics import mean
from typing import Any

from development.evolution.scoring import scalar_rank


def duty_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    bins = {
        "zero": 0,
        "gt_0_le_0.06": 0,
        "gt_0.06_le_0.14": 0,
        "gt_0.14_le_0.18": 0,
        "gt_0.18_le_0.30": 0,
        "gt_0.30_le_0.60": 0,
        "gt_0.60": 0,
    }
    for row in rows:
        duty = float(row["metrics"].get("release_duty_cycle", 0.0))
        if duty == 0.0:
            bins["zero"] += 1
        elif duty <= 0.06:
            bins["gt_0_le_0.06"] += 1
        elif duty <= 0.14:
            bins["gt_0.06_le_0.14"] += 1
        elif duty <= 0.18:
            bins["gt_0.14_le_0.18"] += 1
        elif duty <= 0.30:
            bins["gt_0.18_le_0.30"] += 1
        elif duty <= 0.60:
            bins["gt_0.30_le_0.60"] += 1
        else:
            bins["gt_0.60"] += 1
    return bins


def regime_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for regime, count in row["metrics"].get("regimes", {}).items():
            counts[regime] = counts.get(regime, 0) + int(count)
    return counts


def generation_diagnostics(
    evaluated: list[dict[str, Any]],
    *,
    generation: int,
) -> dict[str, Any]:
    top_by_rank = sorted(
        evaluated,
        key=lambda row: float(row["rank_score"]) if "rank_score" in row else scalar_rank(row),
        reverse=True,
    )
    top10 = top_by_rank[:10]
    top20 = top_by_rank[:20]
    distinct_ancestors: set[str] = set()
    for row in top10:
        distinct_ancestors.update(row.get("ancestor_ids", [row["id"]]))
    return {
        "generation": generation,
        "candidate_count": len(evaluated),
        "mean_release_duty_cycle": mean(
            float(row["metrics"].get("release_duty_cycle", 0.0)) for row in evaluated
        ),
        "mean_release_geometric_event": mean(
            float(row["metrics"].get("release_geometric_event", 0.0)) for row in evaluated
        ),
        "mean_phase_transition_score": mean(
            float(row["metrics"].get("phase_transition_score", 0.0)) for row in evaluated
        ),
        "regime_counts": regime_counts(evaluated),
        "duty_histogram": duty_histogram(evaluated),
        "top10_distinct_ancestor_count": len(distinct_ancestors),
        "top10_lineage": [
            {
                "id": row["id"],
                "rank_score": float(row["rank_score"]),
                "ancestor_ids": row.get("ancestor_ids", [row["id"]]),
                "parent_ids": row.get("parent_ids", []),
                "generation_of_origin": int(row.get("generation_of_origin", generation)),
                "mutation_count": int(row.get("mutation_count", 0)),
                "reproduction_kind": row.get("reproduction_kind", ""),
            }
            for row in top10
        ],
        "event_phase_scatter_top20": [
            {
                "id": row["id"],
                "rank_score": float(row["rank_score"]),
                "release_duty_cycle": float(row["metrics"].get("release_duty_cycle", 0.0)),
                "release_geometric_event": float(
                    row["metrics"].get("release_geometric_event", 0.0)
                ),
                "phase_transition_score": float(
                    row["metrics"].get("phase_transition_score", 0.0)
                ),
                "internal_richness": float(row["metrics"].get("internal_richness", 0.0)),
                "channel_separation": float(row["metrics"].get("channel_separation", 0.0)),
                "regimes": row["metrics"].get("regimes", {}),
                "ancestor_ids": row.get("ancestor_ids", [row["id"]]),
                "mutation_count": int(row.get("mutation_count", 0)),
            }
            for row in top20
        ],
    }
