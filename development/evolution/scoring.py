"""Ranking and archive selection for release evolution."""

from __future__ import annotations

from typing import Any, Iterable

from development.evolution.config import (
    PARETO_OBJECTIVES,
    PHASE_TRANSITION_RANK_CAP,
    RELEASE_GEOMETRIC_RANK_CAP,
)


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    am = a["metrics"]
    bm = b["metrics"]
    no_worse = all(float(am[key]) >= float(bm[key]) for key in PARETO_OBJECTIVES)
    better = any(float(am[key]) > float(bm[key]) for key in PARETO_OBJECTIVES)
    return no_worse and better


def pareto_front(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    items = list(rows)
    front = [
        row
        for row in items
        if not any(dominates(other, row) for other in items if other is not row)
    ]
    front.sort(key=scalar_rank, reverse=True)
    return front


def scalar_rank(row: dict[str, Any]) -> float:
    components = rank_components(row)
    return sum(components.values())


def rank_components(row: dict[str, Any]) -> dict[str, float]:
    metrics = row["metrics"]
    duty = float(metrics.get("release_duty_cycle", 0.0))
    release_geometric = min(
        1.0,
        float(metrics.get("release_geometric_event", 0.0)) / RELEASE_GEOMETRIC_RANK_CAP,
    )
    phase = min(
        1.0,
        float(metrics.get("phase_transition_score", 0.0)) / PHASE_TRANSITION_RANK_CAP,
    )
    flood_pressure = min(1.0, max(0.0, (duty - 0.18) / 0.42))
    return {
        "internal_richness": 1.0 * float(metrics["internal_richness"]),
        "channel_separation": 1.2 * float(metrics.get("channel_separation", 0.0)),
        "release_geometric_event": 1.6 * release_geometric,
        "phase_transition": 1.2 * phase,
        "mathematical_curiosity": 0.7 * float(metrics.get("mathematical_curiosity", 0.0)),
        "geometric_coherence": 1.0 * float(metrics["geometric_coherence"]),
        "regime_bonus": 0.45 * float(metrics["regime_bonus"]),
        "flood_penalty": -1.4 * flood_pressure,
    }


def rare_release_score(duty: float) -> float:
    if 0.06 <= duty <= 0.14:
        return 1.0
    if duty < 0.06:
        return max(0.0, duty / 0.06)
    if duty <= 0.18:
        return max(0.0, 1.0 - (duty - 0.14) / 0.08)
    if duty <= 0.30:
        return max(0.0, 0.5 * (1.0 - (duty - 0.18) / 0.12))
    return 0.0


def archive_bins(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    bins: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        metrics = row["metrics"]
        regimes = metrics.get("regimes", {})
        regime = max(regimes, key=regimes.get) if regimes else "unknown"
        key = (
            regime,
            band(metrics["internal_richness"], (0.18, 0.36)),
            band(metrics["release_duty_cycle"], (0.03, 0.14)),
            band(metrics.get("channel_separation", 0.0), (0.08, 0.22)),
            band(metrics.get("release_geometric_event", 0.0), (0.002, 0.008)),
            band(metrics.get("phase_transition_score", 0.0), (0.006, 0.024)),
            band(metrics.get("mathematical_curiosity", 0.0), (0.28, 0.42)),
            band(metrics["geometric_coherence"], (0.25, 0.55)),
        )
        if key not in bins or scalar_rank(row) > scalar_rank(bins[key]):
            bins[key] = row
    return sorted(bins.values(), key=scalar_rank, reverse=True)


def band(value: float, cuts: tuple[float, float]) -> str:
    if value < cuts[0]:
        return "low"
    if value < cuts[1]:
        return "mid"
    return "high"
