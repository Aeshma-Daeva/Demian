"""Sweep v7.4 dynamic step-length profiles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (
    bottleneck_run,
    coupled_pair,
    memory_pair,
    perturbation_stress_test,
    resume_continuity_probe,
)


PROFILES: dict[str, dict[str, float]] = {
    "fixed": {
        "dynamic_step_min": 1.0,
        "dynamic_step_max": 1.0,
        "dynamic_step_hold_slowdown": 0.0,
        "dynamic_step_refusal_slowdown": 0.0,
        "dynamic_step_recovery_accel": 0.0,
        "dynamic_step_integrate_accel": 0.0,
    },
    "micro": {
        "dynamic_step_min": 0.90,
        "dynamic_step_max": 1.02,
        "dynamic_step_hold_slowdown": 0.08,
        "dynamic_step_refusal_slowdown": 0.05,
        "dynamic_step_recovery_accel": 0.04,
        "dynamic_step_integrate_accel": 0.03,
    },
    "nano": {
        "dynamic_step_min": 0.96,
        "dynamic_step_max": 1.01,
        "dynamic_step_hold_slowdown": 0.03,
        "dynamic_step_refusal_slowdown": 0.02,
        "dynamic_step_recovery_accel": 0.015,
        "dynamic_step_integrate_accel": 0.010,
    },
    "soft": {
        "dynamic_step_min": 0.85,
        "dynamic_step_max": 1.03,
        "dynamic_step_hold_slowdown": 0.15,
        "dynamic_step_refusal_slowdown": 0.08,
        "dynamic_step_recovery_accel": 0.06,
        "dynamic_step_integrate_accel": 0.04,
    },
    "gentle": {
        "dynamic_step_min": 0.75,
        "dynamic_step_max": 1.05,
        "dynamic_step_hold_slowdown": 0.25,
        "dynamic_step_refusal_slowdown": 0.15,
        "dynamic_step_recovery_accel": 0.10,
        "dynamic_step_integrate_accel": 0.08,
    },
    "medium": {
        "dynamic_step_min": 0.65,
        "dynamic_step_max": 1.08,
        "dynamic_step_hold_slowdown": 0.35,
        "dynamic_step_refusal_slowdown": 0.20,
        "dynamic_step_recovery_accel": 0.15,
        "dynamic_step_integrate_accel": 0.10,
    },
    "current": {
        "dynamic_step_min": 0.55,
        "dynamic_step_max": 1.10,
        "dynamic_step_hold_slowdown": 0.45,
        "dynamic_step_refusal_slowdown": 0.25,
        "dynamic_step_recovery_accel": 0.20,
        "dynamic_step_integrate_accel": 0.15,
    },
    "slow_hold": {
        "dynamic_step_min": 0.50,
        "dynamic_step_max": 1.05,
        "dynamic_step_hold_slowdown": 0.55,
        "dynamic_step_refusal_slowdown": 0.30,
        "dynamic_step_recovery_accel": 0.12,
        "dynamic_step_integrate_accel": 0.08,
    },
    "accelerated": {
        "dynamic_step_min": 0.60,
        "dynamic_step_max": 1.20,
        "dynamic_step_hold_slowdown": 0.35,
        "dynamic_step_refusal_slowdown": 0.20,
        "dynamic_step_recovery_accel": 0.25,
        "dynamic_step_integrate_accel": 0.20,
    },
}


def _score(row: dict[str, Any]) -> float:
    memory_floor_penalty = max(0.0, 0.78 - row["memory_final_cosine"]) * 4.0
    resume_penalty = max(0.0, 0.999 - row["resume_capsule_final_cosine"]) * 10.0
    coupling_penalty = max(0.0, abs(row["coupled_final_cosine"]) - 0.15)
    return (
        1.25 * row["rss_negation_final_cosine"]
        + row["memory_final_cosine"]
        + 0.25 * row["bottleneck_reuse_ratio"]
        + 0.15 * min(row["resume_trajectory_shape_ratio"] / 100.0, 1.0)
        - 0.50 * coupling_penalty
        - memory_floor_penalty
        - resume_penalty
    )


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    profile_names = [name.strip() for name in args.profiles.split(",") if name.strip()]
    for name in profile_names:
        kwargs = dict(PROFILES[name])
        perturb = perturbation_stress_test(
            "demian_native_v7.4",
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            perturb_step=args.perturb_step,
            perturb_scales=[args.rss_perturb_scale],
            perturb_mode="rss_negation",
            device=args.device,
            substrate_kwargs=kwargs,
        )
        memory = memory_pair(
            "demian_native_v7.4",
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            initial_delta=args.initial_delta,
            device=args.device,
            substrate_kwargs=kwargs,
        )
        bottleneck = bottleneck_run(
            "demian_native_v7.4",
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            bottleneck_dim=args.bottleneck_dim,
            bottleneck_interval=args.bottleneck_interval,
            device=args.device,
            substrate_kwargs=kwargs,
        )
        coupled = coupled_pair(
            "demian_native_v7.4",
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            coupling_dim=args.coupling_dim,
            coupling_interval=args.coupling_interval,
            coupling_strength=args.coupling_strength,
            device=args.device,
            substrate_kwargs=kwargs,
        )
        resume = resume_continuity_probe(
            "demian_native_v7.4",
            hidden_size=args.hidden_size,
            seed=args.seed,
            pause_steps=args.pause_steps,
            resume_steps=args.resume_steps,
            device=args.device,
            substrate_kwargs=kwargs,
        )
        case = perturb["cases"][0]
        row = {
            "profile": name,
            "substrate_kwargs": kwargs,
            "rss_negation_final_cosine": float(case["final_cosine_vs_baseline"]),
            "memory_final_cosine": float(memory["final_cosine"]),
            "bottleneck_reuse_ratio": float(bottleneck["reuse_ratio"]),
            "coupled_final_cosine": float(coupled["final_cosine"]),
            "resume_capsule_final_cosine": float(resume["capsule_resume"]["final_cosine_vs_uninterrupted"]),
            "resume_notebook_final_cosine": float(resume["notebook_resume"]["final_cosine_vs_uninterrupted"]),
            "resume_trajectory_shape_ratio": float(resume["continuity_advantage"]["trajectory_shape_ratio"]),
        }
        row["score"] = _score(row)
        rows.append(row)
        print(
            f"{name}: score={row['score']:.4f} rss={row['rss_negation_final_cosine']:.4f} "
            f"mem={row['memory_final_cosine']:.4f} couple={row['coupled_final_cosine']:.4f} "
            f"resume={row['resume_capsule_final_cosine']:.6f}"
        )

    viable = [
        row for row in rows
        if row["memory_final_cosine"] >= args.min_memory_cosine
        and row["rss_negation_final_cosine"] >= args.min_rss_cosine
        and row["resume_capsule_final_cosine"] >= args.min_resume_cosine
    ]
    dynamic_viable = [row for row in viable if row["profile"] != "fixed"]
    candidates = dynamic_viable or viable or rows
    best = max(candidates, key=lambda row: row["score"])
    return {
        "config": vars(args),
        "selection_rule": {
            "min_memory_cosine": args.min_memory_cosine,
            "min_rss_cosine": args.min_rss_cosine,
            "min_resume_cosine": args.min_resume_cosine,
            "preference": "best dynamic profile if viable, otherwise best fixed/overall profile",
        },
        "recommended_profile": best["profile"],
        "recommended_kwargs": best["substrate_kwargs"],
        "recommended": best,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="fixed,nano,micro,soft,gentle,medium,current,slow_hold,accelerated")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--perturb-step", type=int, default=128)
    parser.add_argument("--rss-perturb-scale", type=float, default=0.5)
    parser.add_argument("--initial-delta", type=float, default=0.05)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--bottleneck-interval", type=int, default=16)
    parser.add_argument("--coupling-dim", type=int, default=8)
    parser.add_argument("--coupling-interval", type=int, default=16)
    parser.add_argument("--coupling-strength", type=float, default=0.05)
    parser.add_argument("--pause-steps", type=int, default=128)
    parser.add_argument("--resume-steps", type=int, default=128)
    parser.add_argument("--min-memory-cosine", type=float, default=0.78)
    parser.add_argument("--min-rss-cosine", type=float, default=0.84)
    parser.add_argument("--min-resume-cosine", type=float, default=0.999)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="data/substrate_lab/v74_dynamic_step_sweep_20260429")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_sweep(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"recommended={payload['recommended_profile']}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
