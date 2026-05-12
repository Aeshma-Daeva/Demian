"""Shared constants for v9 five-channel and Demian v1 release evolution."""

from __future__ import annotations


OBSERVABLE_KEYS = (
    "fast_norm",
    "slow_norm",
    "message_norm",
    "carrier_norm",
    "control_norm",
    "message_carrier_gap",
    "release_pressure",
    "surface_delta",
    "time_since_perturbation",
    "perturbation_magnitude",
)
LOW_RANK_TARGETS = (
    "message_gate",
    "carrier_gate",
    "release_gate",
    "message_to_carrier",
    "carrier_to_slow",
)
CHANNELS = ("fast", "slow", "control", "message", "carrier")
RELEASE_TARGET_SCALARS = tuple(f"release_to_{name}_scale" for name in CHANNELS)
BASE_EXCLUDED_SCALARS = {
    "plastic_decay",
    "plastic_update_scale",
    "plastic_clip",
    "matrix_delta_scale",
    *RELEASE_TARGET_SCALARS,
}
PARETO_OBJECTIVES = (
    "internal_richness",
    "channel_separation",
    "release_geometric_event",
    "phase_transition_score",
    "mathematical_curiosity",
    "geometric_coherence",
    "regime_bonus",
)
PHASE_DELTA_SCALE = 0.04
PHASE_CURVATURE_SCALE = 0.03
RELEASE_GEOMETRIC_RANK_CAP = 0.25
PHASE_TRANSITION_RANK_CAP = 1.0
