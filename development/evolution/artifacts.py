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
