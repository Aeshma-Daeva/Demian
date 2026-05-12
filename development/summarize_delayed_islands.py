#!/usr/bin/env python3
"""Rank delayed-eligibility islands together and validate top candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolve_v9_5ch_release import evaluate_genome, scalar_rank


SUCCESS_DUTY_BAND = (0.06, 0.18)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_candidates(roots: list[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, root in enumerate(roots, start=1):
        for path in sorted((root / "candidates").glob("*.json")):
            candidate = load_json(path)
            candidate["_island"] = f"island_{index}"
            candidate["_root"] = str(root)
            candidates.append(candidate)
    return candidates


def metric(row: dict[str, Any], name: str) -> float:
    return float(row.get("metrics", {}).get(name, 0.0))


def candidate_brief(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "island": candidate.get("_island"),
        "id": candidate.get("id"),
        "generation": candidate.get("generation"),
        "rank_score": float(candidate.get("rank_score", scalar_rank(candidate))),
        "reproduction_kind": candidate.get("reproduction_kind"),
        "release_duty_cycle": metric(candidate, "release_duty_cycle"),
        "release_causal_divergence": metric(candidate, "release_causal_divergence"),
        "release_gain_zero_causal_divergence": metric(candidate, "release_gain_zero_release_causal_divergence"),
        "release_timing_score": metric(candidate, "release_timing_score"),
        "phase_transition_score": metric(candidate, "phase_transition_score"),
        "release_geometric_event": metric(candidate, "release_geometric_event"),
    }


def validate_candidate(
    candidate: dict[str, Any],
    *,
    hidden_size: int,
    steps: int,
    seeds: list[int],
    perturb_step: int,
    perturb_scales: list[float],
    rank: int,
    device: str,
) -> dict[str, Any]:
    result = evaluate_genome(
        candidate["genome"],
        hidden_size=hidden_size,
        steps=steps,
        seeds=seeds,
        perturb_step=perturb_step,
        perturb_scales=perturb_scales,
        rank=rank,
        device=device,
        paired_causal=True,
        keep_trajectories=False,
    )
    row = {
        "island": candidate.get("_island"),
        "id": candidate.get("id"),
        "training_rank_score": float(candidate.get("rank_score", scalar_rank(candidate))),
        "heldout_rank_score": scalar_rank({"metrics": result["aggregate"]}),
        "metrics": result["aggregate"],
    }
    metrics = row["metrics"]
    row["success_flags"] = {
        "duty_in_band": SUCCESS_DUTY_BAND[0] <= float(metrics.get("release_duty_cycle", 0.0)) <= SUCCESS_DUTY_BAND[1],
        "routes_disabled_causal_positive": float(metrics.get("release_causal_divergence", 0.0)) > 0.0,
        "gain_zero_causal_positive": float(metrics.get("release_gain_zero_release_causal_divergence", 0.0)) > 0.0,
        "timing_score_gt_0_3": float(metrics.get("release_timing_score", 0.0)) > 0.3,
    }
    return row


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="delayed island output directories")
    parser.add_argument("--out", required=True, help="summary JSON path")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--heldout-seeds", default="96,97,98")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--perturb-step", type=int, default=64)
    parser.add_argument("--perturb-scales", default="0.35,0.7")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-heldout", action="store_true")
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    candidates = load_candidates(roots)
    if not candidates:
        raise SystemExit("no candidate JSON files found under supplied roots")
    ranked = sorted(candidates, key=scalar_rank, reverse=True)
    top = ranked[: max(1, args.top_n)]
    generations = sorted({int(candidate.get("generation", 0)) for candidate in candidates})
    heldout = []
    if not args.skip_heldout:
        for candidate in top:
            heldout.append(
                validate_candidate(
                    candidate,
                    hidden_size=args.hidden_size,
                    steps=args.steps,
                    seeds=parse_ints(args.heldout_seeds),
                    perturb_step=args.perturb_step,
                    perturb_scales=parse_floats(args.perturb_scales),
                    rank=args.rank,
                    device=args.device,
                )
            )

    success_summary = {
        "any_duty_in_band": any(row["success_flags"]["duty_in_band"] for row in heldout),
        "any_routes_disabled_causal_positive": any(
            row["success_flags"]["routes_disabled_causal_positive"] for row in heldout
        ),
        "any_gain_zero_causal_positive": any(row["success_flags"]["gain_zero_causal_positive"] for row in heldout),
        "any_top5_timing_score_gt_0_3": any(row["success_flags"]["timing_score_gt_0_3"] for row in heldout),
    } if heldout else {}

    payload = {
        "experiment": "demian-v1-delayed-eligibility-four-island",
        "roots": [str(root) for root in roots],
        "candidate_count": len(candidates),
        "generation_count": len(generations),
        "final_generation": generations[-1] if generations else None,
        "top_n": len(top),
        "heldout_seeds": parse_ints(args.heldout_seeds),
        "heldout_scales": parse_floats(args.perturb_scales),
        "ranked_top": [candidate_brief(candidate) for candidate in top],
        "heldout_validation": heldout,
        "success_criteria": {
            "duty_band": list(SUCCESS_DUTY_BAND),
            "routes_disabled_causal_positive": True,
            "gain_zero_causal_positive": True,
            "top5_timing_score_gt_0_3": True,
        },
        "success_summary": success_summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(f"candidates={len(candidates)} top_n={len(top)} heldout={len(heldout)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
