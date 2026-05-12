#!/usr/bin/env python3
"""Summarize Demian v2 dual-track islands and validate top candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolution.scoring import ENGINEERED_TARGET_RANK_MODE, scalar_rank
from development.evolve_v9_5ch_release import evaluate_genome

CHECKPOINTS = (4, 9, 14, 19)
DUTY_BAND = (0.06, 0.18)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def metric(candidate: dict[str, Any], name: str) -> float:
    return float(candidate.get("metrics", {}).get(name, 0.0))


def infer_track(root: Path) -> str:
    config_path = root / "config.json"
    if not config_path.exists():
        return "unknown"
    config = load_json(config_path)
    return str(config.get("rank_mode", "unknown"))


def load_candidates(roots: list[Path]) -> list[dict[str, Any]]:
    candidates = []
    for root in roots:
        track = infer_track(root)
        for path in sorted((root / "candidates").glob("*.json")):
            candidate = load_json(path)
            candidate["_root"] = str(root)
            candidate["_track"] = track
            candidate.setdefault("rank_mode", track)
            candidates.append(candidate)
    return candidates


def brief(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": candidate.get("_root"),
        "track": candidate.get("_track"),
        "id": candidate.get("id"),
        "generation": candidate.get("generation"),
        "rank_score": float(candidate.get("rank_score", scalar_rank(candidate))),
        "reproduction_kind": candidate.get("reproduction_kind"),
        "release_duty_cycle": metric(candidate, "release_duty_cycle"),
        "release_causal_divergence": metric(candidate, "release_causal_divergence"),
        "release_gain_zero_release_causal_divergence": metric(
            candidate,
            "release_gain_zero_release_causal_divergence",
        ),
        "release_timing_score": metric(candidate, "release_timing_score"),
        "phase_transition_score": metric(candidate, "phase_transition_score"),
        "gain_zero_clean_fraction": metric(candidate, "gain_zero_clean_fraction"),
    }


def checkpoint_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for generation in CHECKPOINTS:
        generation_rows = [c for c in candidates if int(c.get("generation", -1)) == generation]
        if not generation_rows:
            continue
        for track in sorted({str(c.get("_track", "unknown")) for c in generation_rows}):
            track_rows = [c for c in generation_rows if str(c.get("_track", "unknown")) == track]
            ranked = sorted(track_rows, key=scalar_rank, reverse=True)
            rows.append(
                {
                    "generation": generation,
                    "human_generation": generation + 1,
                    "track": track,
                    "candidate_count": len(track_rows),
                    "best": brief(ranked[0]),
                    "mean_duty": sum(metric(c, "release_duty_cycle") for c in track_rows) / len(track_rows),
                    "mean_route_causal": sum(metric(c, "release_causal_divergence") for c in track_rows)
                    / len(track_rows),
                    "mean_timing": sum(metric(c, "release_timing_score") for c in track_rows) / len(track_rows),
                }
            )
    return rows


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
        **brief(candidate),
        "heldout_rank_score": scalar_rank(
            {
                "metrics": result["aggregate"],
                "rank_mode": candidate.get("rank_mode", ENGINEERED_TARGET_RANK_MODE),
            }
        ),
        "heldout_metrics": result["aggregate"],
    }
    metrics = result["aggregate"]
    row["flags"] = {
        "duty_in_band": DUTY_BAND[0] <= float(metrics.get("release_duty_cycle", 0.0)) <= DUTY_BAND[1],
        "routes_disabled_causal_positive": float(metrics.get("release_causal_divergence", 0.0)) > 0.0,
        "gain_zero_causal_positive": float(metrics.get("release_gain_zero_release_causal_divergence", 0.0)) > 0.0,
        "timing_score_gt_0_3": float(metrics.get("release_timing_score", 0.0)) > 0.3,
        "gain_zero_clean": float(metrics.get("gain_zero_clean_fraction", 0.0)) >= 1.0,
    }
    row["bypass_candidate"] = (
        candidate.get("_track") == ENGINEERED_TARGET_RANK_MODE
        and (
            not row["flags"]["duty_in_band"]
            or not row["flags"]["timing_score_gt_0_3"]
            or not row["flags"]["gain_zero_clean"]
        )
        and row["flags"]["routes_disabled_causal_positive"]
    )
    return row


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Track A/B island output directories")
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
    by_track = {
        track: sorted(
            [candidate for candidate in candidates if str(candidate.get("_track")) == track],
            key=scalar_rank,
            reverse=True,
        )
        for track in sorted({str(candidate.get("_track")) for candidate in candidates})
    }
    heldout = []
    if not args.skip_heldout:
        for ranked in by_track.values():
            for candidate in ranked[: max(1, args.top_n)]:
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

    payload = {
        "experiment": "demian-v2-dual-track",
        "roots": [str(root) for root in roots],
        "candidate_count": len(candidates),
        "tracks": {
            track: {
                "candidate_count": len(ranked),
                "top": [brief(candidate) for candidate in ranked[: max(1, args.top_n)]],
            }
            for track, ranked in by_track.items()
        },
        "checkpoints": checkpoint_summary(candidates),
        "heldout_seeds": parse_ints(args.heldout_seeds),
        "heldout_validation": heldout,
        "success_summary": {
            "any_engineered_duty_in_band": any(
                row["track"] == ENGINEERED_TARGET_RANK_MODE and row["flags"]["duty_in_band"]
                for row in heldout
            ),
            "any_engineered_timing_score_gt_0_3": any(
                row["track"] == ENGINEERED_TARGET_RANK_MODE and row["flags"]["timing_score_gt_0_3"]
                for row in heldout
            ),
            "any_gain_zero_unclean": any(not row["flags"]["gain_zero_clean"] for row in heldout),
            "bypass_candidate_count": sum(1 for row in heldout if row["bypass_candidate"]),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {out}")
    print(f"candidates={len(candidates)} tracks={','.join(sorted(by_track))} heldout={len(heldout)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
