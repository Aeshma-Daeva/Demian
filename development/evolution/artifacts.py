"""Artifact loading and validation for predecessor evolution summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_summary(path: Path | str) -> dict[str, Any]:
    """Load one evolution summary JSON file."""
    return json.loads(Path(path).read_text())


def validate_v10_summary(summary: dict[str, Any]) -> list[str]:
    """Return validation errors for the v10 predecessor summary contract."""
    errors: list[str] = []
    required = {
        "experiment",
        "candidate_count",
        "generation_count",
        "eval_seed",
        "per_generation",
        "final_best",
        "best_overall",
        "curve_summary",
    }
    missing = sorted(required - set(summary))
    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")
        return errors
    if summary["experiment"] != "v10.0-frozen-evolution":
        errors.append(f"unexpected experiment: {summary['experiment']!r}")
    if int(summary["candidate_count"]) != 640:
        errors.append(f"unexpected candidate_count: {summary['candidate_count']!r}")
    if int(summary["generation_count"]) != 20:
        errors.append(f"unexpected generation_count: {summary['generation_count']!r}")
    if int(summary["eval_seed"]) != 94:
        errors.append(f"unexpected eval_seed: {summary['eval_seed']!r}")
    if len(summary["per_generation"]) != int(summary["generation_count"]):
        errors.append("per_generation length does not match generation_count")
    return errors


def v10_summary_digest(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the compact digest used by restart docs and smoke tests."""
    final_best = summary["final_best"]
    metrics = final_best.get("metrics", final_best)
    return {
        "experiment": summary["experiment"],
        "candidate_count": int(summary["candidate_count"]),
        "generation_count": int(summary["generation_count"]),
        "eval_seed": int(summary["eval_seed"]),
        "final_best_id": final_best["id"],
        "final_best_rank": float(final_best["rank_score"]),
        "final_best_duty": float(metrics["release_duty_cycle"]),
        "final_best_event": float(metrics["release_geometric_event"]),
        "final_best_phase": float(metrics["phase_transition_score"]),
        "final_best_regimes": dict(metrics.get("regimes", {})),
    }


def validate_track_b_discovery_summary(summary: dict[str, Any]) -> list[str]:
    """Validate the common Track B discovery summary contract."""
    errors: list[str] = []
    required = {"experiment_label", "output_id", "native_objective", "causal_mode", "generation_count"}
    missing = sorted(required - set(summary))
    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")
    for key in ("experiment_label", "output_id", "native_objective"):
        if key in summary and not str(summary.get(key, "")).strip():
            errors.append(f"{key} must be non-empty")
    if "output_ids" in summary:
        output_ids = [str(item) for item in summary.get("output_ids", [])]
        duplicates = sorted({item for item in output_ids if output_ids.count(item) > 1})
        if duplicates:
            errors.append(f"reused output_ids: {', '.join(duplicates)}")
    return errors


def validate_track_b_characterization_summary(summary: dict[str, Any]) -> list[str]:
    """Validate held-out characterization fields required for mechanism labels."""
    errors: list[str] = []
    required = {
        "held_out_seeds",
        "perturb_steps",
        "channel_necessity_order",
        "gain_zero_cleanliness",
        "capsule_result_flags",
    }
    missing = sorted(required - set(summary))
    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")
    if "held_out_seeds" in summary and not summary["held_out_seeds"]:
        errors.append("held_out_seeds must be non-empty")
    if "perturb_steps" in summary and not summary["perturb_steps"]:
        errors.append("perturb_steps must be non-empty")
    if "capsule_result_flags" in summary and not isinstance(summary["capsule_result_flags"], dict):
        errors.append("capsule_result_flags must be a dict")
    return errors
