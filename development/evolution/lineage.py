"""Lineage metadata helpers for evolutionary runs."""

from __future__ import annotations

from statistics import mean
from typing import Any


def population_entry(
    genome: dict[str, Any],
    reproduction_kind: str,
    parent_ids: list[str],
    *,
    ancestor_ids: list[str] | None = None,
    generation_of_origin: int | None = None,
    mutation_count: int = 0,
) -> dict[str, Any]:
    entry = {
        "genome": genome,
        "reproduction_kind": reproduction_kind,
        "parent_ids": parent_ids,
        "mutation_count": mutation_count,
    }
    if ancestor_ids is not None:
        entry["ancestor_ids"] = ancestor_ids
    if generation_of_origin is not None:
        entry["generation_of_origin"] = generation_of_origin
    return entry


def merge_ancestor_ids(*parents: dict[str, Any]) -> list[str]:
    ancestors: list[str] = []
    for parent in parents:
        for ancestor_id in parent.get("ancestor_ids", [parent["id"]]):
            if ancestor_id not in ancestors:
                ancestors.append(ancestor_id)
    return ancestors


def lineage_child_entry(
    genome: dict[str, Any],
    reproduction_kind: str,
    parent: dict[str, Any],
    generation_of_origin: int,
) -> dict[str, Any]:
    return population_entry(
        genome,
        reproduction_kind,
        [parent["id"]],
        ancestor_ids=parent.get("ancestor_ids", [parent["id"]]),
        generation_of_origin=generation_of_origin,
        mutation_count=int(parent.get("mutation_count", 0)) + 1,
    )


def lineage_deltas(
    metrics: dict[str, Any],
    parent_ids: list[str],
    rows_by_id: dict[str, dict[str, Any]],
) -> dict[str, float]:
    if not parent_ids:
        return {}
    parents = [
        rows_by_id[parent_id]["metrics"]
        for parent_id in parent_ids
        if parent_id in rows_by_id
    ]
    if not parents:
        return {}
    keys = (
        "release_geometric_event",
        "phase_transition_score",
        "release_duty_cycle",
        "mathematical_curiosity",
        "channel_separation",
    )
    deltas = {}
    for key in keys:
        parent_mean = mean(float(parent.get(key, 0.0)) for parent in parents)
        deltas[f"delta_{key}"] = float(metrics.get(key, 0.0)) - parent_mean
    return deltas


def lineage_stability_score(deltas: dict[str, float]) -> float:
    if not deltas:
        return 0.0
    event_drop = max(0.0, -deltas.get("delta_release_geometric_event", 0.0))
    phase_drop = max(0.0, -deltas.get("delta_phase_transition_score", 0.0))
    duty_drift = abs(deltas.get("delta_release_duty_cycle", 0.0))
    return 1.0 / (1.0 + 120.0 * event_drop + 8.0 * phase_drop + 2.0 * duty_drift)
