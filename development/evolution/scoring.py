"""Ranking and archive selection for release evolution."""

from __future__ import annotations

import math
from typing import Any, Iterable

from development.evolution.config import (
    PARETO_OBJECTIVES,
    PHASE_TRANSITION_RANK_CAP,
    RELEASE_GEOMETRIC_RANK_CAP,
)

ENGINEERED_TARGET_RANK_MODE = "engineered_target"
NATIVE_EMERGENCE_RANK_MODE = "native_emergence"
DYNAMIC_SELECTION_PROBE_RANK_MODE = "dynamic_selection_probe"
NATIVE_OBJECTIVE_GATE_STATE = "gate_state"
NATIVE_OBJECTIVE_MORPHOLOGY_ONLY = "morphology_only"
NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY = "morphology_low_duty"
NATIVE_OBJECTIVE_COMBINED_DISCOVERY = "combined_discovery"
NATIVE_OBJECTIVE_INTERNAL_CONSISTENCY = "internal_consistency"
NATIVE_OBJECTIVES = (
    NATIVE_OBJECTIVE_GATE_STATE,
    NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
    NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY,
    NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
    NATIVE_OBJECTIVE_INTERNAL_CONSISTENCY,
)
DEFAULT_NATIVE_OBJECTIVE = NATIVE_OBJECTIVE_GATE_STATE
CAUSAL_MODE_ROUTE_RELEASE = "route_release"
CAUSAL_MODE_GATE_STATE = "gate_state_propagation"
CAUSAL_MODE_COMBINED = "combined"
CAUSAL_MODES = (
    CAUSAL_MODE_ROUTE_RELEASE,
    CAUSAL_MODE_GATE_STATE,
    CAUSAL_MODE_COMBINED,
)
DEFAULT_CAUSAL_MODE = CAUSAL_MODE_ROUTE_RELEASE
GAIN_ZERO_DIVERGENCE_TOLERANCE = 1e-8


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    am = a["metrics"]
    bm = b["metrics"]
    no_worse = all(float(am.get(key, 0.0)) >= float(bm.get(key, 0.0)) for key in PARETO_OBJECTIVES)
    better = any(float(am.get(key, 0.0)) > float(bm.get(key, 0.0)) for key in PARETO_OBJECTIVES)
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
    rank_mode = str(row.get("rank_mode", ENGINEERED_TARGET_RANK_MODE))
    if rank_mode == NATIVE_EMERGENCE_RANK_MODE:
        components = rank_components(row, rank_mode=rank_mode)
        return sum(components.values())
    components = rank_components(row, rank_mode=rank_mode)
    return float(components["rank"])


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def rank_components(
    row: dict[str, Any],
    *,
    rank_mode: str | None = None,
    causal_mode: str | None = None,
    native_objective: str | None = None,
) -> dict[str, float]:
    mode = rank_mode or str(row.get("rank_mode", ENGINEERED_TARGET_RANK_MODE))
    selected_causal_mode = causal_mode or str(row.get("causal_mode", DEFAULT_CAUSAL_MODE))
    selected_native_objective = native_objective or str(row.get("native_objective", DEFAULT_NATIVE_OBJECTIVE))
    if selected_causal_mode not in CAUSAL_MODES:
        raise ValueError(f"unknown causal_mode: {selected_causal_mode}")
    if mode == NATIVE_EMERGENCE_RANK_MODE:
        return native_rank_components(
            row,
            causal_mode=selected_causal_mode,
            native_objective=selected_native_objective,
        )
    if mode == DYNAMIC_SELECTION_PROBE_RANK_MODE:
        return dynamic_selection_probe_rank_components(row)
    if mode == ENGINEERED_TARGET_RANK_MODE:
        return engineered_target_rank_components(row, causal_mode=selected_causal_mode)
    raise ValueError(f"unknown rank_mode: {mode}")


def gain_zero_clean(metrics: dict[str, Any]) -> bool:
    if "gain_zero_clean_fraction" in metrics:
        return float(metrics.get("gain_zero_clean_fraction", 0.0)) >= 1.0
    if "gain_zero_clean" in metrics:
        return bool(metrics.get("gain_zero_clean"))
    return False


def causal_divergence_components(metrics: dict[str, Any]) -> dict[str, float]:
    """Separate route-specific release from persistent clean gain-zero propagation."""
    route_disabled = max(0.0, float(metrics.get("release_causal_divergence", 0.0)))
    has_gain_zero_divergence = "release_gain_zero_release_causal_divergence" in metrics
    gain_zero = max(0.0, float(metrics.get("release_gain_zero_release_causal_divergence", 0.0)))
    clean = gain_zero_clean(metrics)
    route_specific = (
        route_disabled
        if route_disabled > 0.0
        and (not has_gain_zero_divergence or gain_zero <= GAIN_ZERO_DIVERGENCE_TOLERANCE)
        else 0.0
    )
    gate_state = gain_zero if clean and route_disabled > 0.0 and gain_zero > GAIN_ZERO_DIVERGENCE_TOLERANCE else 0.0
    return {
        "release_route_disabled_causal_divergence": route_disabled,
        "release_gain_zero_causal_divergence": gain_zero,
        "route_release_causal_divergence": route_specific,
        "gate_state_causal_divergence": gate_state,
    }


def selected_causal_divergence(metrics: dict[str, Any], causal_mode: str) -> tuple[float, dict[str, float]]:
    components = causal_divergence_components(metrics)
    if causal_mode == CAUSAL_MODE_ROUTE_RELEASE:
        return components["route_release_causal_divergence"], components
    if causal_mode == CAUSAL_MODE_GATE_STATE:
        return components["gate_state_causal_divergence"], components
    if causal_mode == CAUSAL_MODE_COMBINED:
        return (
            components["route_release_causal_divergence"] + components["gate_state_causal_divergence"],
            components,
        )
    raise ValueError(f"unknown causal_mode: {causal_mode}")


def native_morphology_components(metrics: dict[str, Any]) -> dict[str, float]:
    """Shared morphology terms for Track B objective variants."""
    return {
        "internal_richness": 1.0 * float(metrics["internal_richness"]),
        "channel_separation": 1.2 * float(metrics.get("channel_separation", 0.0)),
        "mathematical_curiosity": 0.7 * float(metrics.get("mathematical_curiosity", 0.0)),
        "geometric_coherence": 1.0 * float(metrics["geometric_coherence"]),
    }


def low_duty_preference(metrics: dict[str, Any], *, cap: float = 0.15) -> float:
    """Small capped preference for quieter release duty in morphology searches."""
    duty = clamp01(float(metrics.get("release_duty_cycle", 0.0)))
    return min(float(cap), float(cap) * (1.0 - duty) ** 2)


def native_rank_components(
    row: dict[str, Any],
    *,
    causal_mode: str = DEFAULT_CAUSAL_MODE,
    native_objective: str = DEFAULT_NATIVE_OBJECTIVE,
) -> dict[str, float]:
    """Track B rank variants for mechanism discovery."""
    if native_objective not in NATIVE_OBJECTIVES:
        raise ValueError(f"unknown native_objective: {native_objective}")
    metrics = row["metrics"]
    morphology = native_morphology_components(metrics)
    if native_objective == NATIVE_OBJECTIVE_MORPHOLOGY_ONLY:
        return morphology
    if native_objective == NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY:
        return {
            **morphology,
            "low_duty_preference": low_duty_preference(metrics),
        }
    causal, causal_components = selected_causal_divergence(metrics, causal_mode)
    if native_objective == NATIVE_OBJECTIVE_COMBINED_DISCOVERY:
        return {
            **morphology,
            **causal_components,
            "release_causal_divergence_raw": 0.25 * causal,
            "release_geometric_event": 0.15 * float(metrics.get("release_geometric_event", 0.0)),
            "phase_transition": 0.15 * float(metrics.get("phase_transition_score", 0.0)),
        }
    if native_objective == NATIVE_OBJECTIVE_INTERNAL_CONSISTENCY:
        return {
            **morphology,
            "internal_consistency": 1.5 * float(metrics.get("internal_consistency", 0.0)),
        }
    return {
        **morphology,
        **causal_components,
        "release_causal_divergence_raw": 1.0 * causal,
        "release_geometric_event": 0.3 * float(metrics.get("release_geometric_event", 0.0)),
        "phase_transition": 0.3 * float(metrics.get("phase_transition_score", 0.0)),
    }


def dynamic_selection_probe_phase(generation: int) -> str:
    if generation < 20:
        return "foundation"
    if generation < 40:
        return "sparsity"
    return "timing"


def high_duty_penalty(duty: float) -> float:
    return -2.5 * min(1.0, max(0.0, (float(duty) - 0.18) / 0.30))


def dynamic_selection_probe_rank_components(row: dict[str, Any]) -> dict[str, float]:
    """Three-phase objective schedule for sparse causal timing probes."""
    metrics = row["metrics"]
    generation = int(row.get("generation", 0))
    phase = dynamic_selection_probe_phase(generation)
    morphology = native_morphology_components(metrics)
    raw_causal = max(0.0, float(metrics.get("release_causal_divergence", 0.0)))
    causal_multiplier = clamp01(raw_causal / 0.01)
    duty = float(metrics.get("release_duty_cycle", 0.0))
    duty_band_multiplier = 1.0 if 0.06 <= duty <= 0.18 else 0.0
    event = float(metrics.get("release_geometric_event", 0.0))
    phase_transition = float(metrics.get("phase_transition_score", 0.0))
    timing = float(metrics.get("release_timing_score", 0.0))
    causal_components = causal_divergence_components(metrics)
    rank_terms = {
        **morphology,
        "release_causal_divergence_raw": 1.5 * raw_causal,
        "release_geometric_event": 0.3 * event,
        "phase_transition": 0.3 * phase_transition,
    }
    components = {
        **rank_terms,
        **causal_components,
        "causal_multiplier": causal_multiplier,
        "duty_band_multiplier": duty_band_multiplier,
        "schedule_phase_index": float(("foundation", "sparsity", "timing").index(phase)),
    }
    if phase in {"sparsity", "timing"}:
        rank_terms["release_geometric_event"] = 0.3 * causal_multiplier * event
        rank_terms["phase_transition"] = 0.3 * causal_multiplier * phase_transition
        rank_terms["duty_band_penalty"] = high_duty_penalty(duty)
    if phase == "timing":
        rank_terms["timing_bonus"] = 1.6 * causal_multiplier * duty_band_multiplier * timing
    rank = sum(rank_terms.values())
    components.update(rank_terms)
    return {**components, "rank": rank}


def engineered_target_rank_components(row: dict[str, Any], *, causal_mode: str = DEFAULT_CAUSAL_MODE) -> dict[str, float]:
    """Track A adversarial rank for sparse causal delayed release."""
    metrics = row["metrics"]
    duty = float(metrics.get("release_duty_cycle", 0.0))
    route_causal, causal_components = selected_causal_divergence(metrics, causal_mode)
    causal_multiplier = clamp01(route_causal / 0.01)
    route_causal_clipped = clamp01(route_causal / RELEASE_GEOMETRIC_RANK_CAP)
    phase_clipped = clamp01(float(metrics.get("phase_transition_score", 0.0)) / PHASE_TRANSITION_RANK_CAP)
    duty_band_multiplier = 1.0 if 0.06 <= duty <= 0.18 else 0.0

    score_a_components = {
        "internal_richness": 1.0 * float(metrics["internal_richness"]),
        "channel_separation": 1.2 * float(metrics.get("channel_separation", 0.0)),
        "mathematical_curiosity": 0.7 * float(metrics.get("mathematical_curiosity", 0.0)),
        "geometric_coherence": 1.0 * float(metrics["geometric_coherence"]),
    }
    score_b_components = {
        "release_causal_divergence": 1.6 * causal_multiplier * route_causal_clipped,
        "timing_bonus": 1.6 * causal_multiplier * duty_band_multiplier * float(metrics.get("release_timing_score", 0.0)),
        "phase_transition": 1.2 * causal_multiplier * phase_clipped,
        "regime_bonus": 0.45 * float(metrics["regime_bonus"]),
        "duty_band_penalty": -2.5 * min(1.0, max(0.0, (duty - 0.18) / 0.30)),
    }
    score_a = sum(score_a_components.values())
    score_b = sum(score_b_components.values())
    rank = math.sqrt(max(0.0, score_a) * max(0.0, score_b))
    return {
        **score_a_components,
        **causal_components,
        **score_b_components,
        "causal_multiplier": causal_multiplier,
        "duty_band_multiplier": duty_band_multiplier,
        "score_a": score_a,
        "score_b": score_b,
        "rank": rank,
    }


def legacy_additive_rank_components(row: dict[str, Any]) -> dict[str, float]:
    metrics = row["metrics"]
    duty = float(metrics.get("release_duty_cycle", 0.0))
    causal_raw = float(metrics.get("release_causal_divergence", 0.0))
    release_causal = min(
        1.0,
        max(0.0, causal_raw) / RELEASE_GEOMETRIC_RANK_CAP,
    )
    causal_gate = 1.0 if causal_raw > 0.0 else 0.0
    phase = min(
        1.0,
        float(metrics.get("phase_transition_score", 0.0)) / PHASE_TRANSITION_RANK_CAP,
    ) * causal_gate
    duty_band_pressure = release_duty_band_pressure(duty)
    return {
        "internal_richness": 1.0 * float(metrics["internal_richness"]),
        "channel_separation": 1.2 * float(metrics.get("channel_separation", 0.0)),
        "release_causal_divergence": 1.6 * release_causal,
        "phase_transition": 1.2 * phase,
        "timing_bonus": 0.8 * release_causal * float(metrics.get("release_timing_score", 0.0)),
        "mathematical_curiosity": 0.7 * float(metrics.get("mathematical_curiosity", 0.0)),
        "geometric_coherence": 1.0 * float(metrics["geometric_coherence"]),
        "regime_bonus": 0.45 * float(metrics["regime_bonus"]),
        "duty_band_penalty": -1.4 * duty_band_pressure,
    }


def legacy_additive_scalar_rank(row: dict[str, Any]) -> float:
    components = legacy_additive_rank_components(row)
    return sum(components.values())


def release_duty_band_pressure(duty: float) -> float:
    """Return penalty pressure outside the target causal-release duty band."""
    duty = float(duty)
    if 0.06 <= duty <= 0.18:
        return 0.0
    if duty < 0.06:
        return min(1.0, (0.06 - duty) / 0.06)
    return min(1.0, (duty - 0.18) / 0.42)


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
