#!/usr/bin/env python3
"""Small parameter sweep for nontrivial self-loop regimes."""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (
    SUBSTRATE_REGISTRY,
    basin_map,
    bottleneck_run,
    coupled_pair,
    memory_pair,
    perturbation_pair,
    regime_score,
)


def _parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Sweep self-loop substrate regimes")
    p.add_argument("--substrates", nargs="*", default=list(SUBSTRATE_REGISTRY.keys()))
    p.add_argument("--hidden-size", type=int, default=48)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--basin-seeds", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--init-scales", default="0.2,0.5,1.0")
    p.add_argument("--feedback-scales", default="0.8,1.0,1.2")
    p.add_argument("--state-gains", default="0.9,1.0,1.1")
    p.add_argument("--perturb-step", type=int, default=64)
    p.add_argument("--perturb-scale", type=float, default=0.04)
    p.add_argument("--initial-delta", type=float, default=0.04)
    p.add_argument("--bottleneck-dim", type=int, default=8)
    p.add_argument("--bottleneck-interval", type=int, default=16)
    p.add_argument("--coupling-dim", type=int, default=8)
    p.add_argument("--coupling-interval", type=int, default=16)
    p.add_argument("--coupling-strength", type=float, default=0.05)
    p.add_argument("--device", default="cpu")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--out-dir", default="data/substrate_sweeps")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_scales = _parse_grid(args.init_scales)
    feedback_scales = _parse_grid(args.feedback_scales)
    state_gains = _parse_grid(args.state_gains)

    all_rows = []
    print("Substrate regime sweep")
    print(f"out_dir: {out_dir}")

    for substrate in args.substrates:
        print()
        print(f"[{substrate}]")
        for init_scale, feedback_scale, state_gain in product(init_scales, feedback_scales, state_gains):
            cfg = {
                "init_scale": init_scale,
                "feedback_scale": feedback_scale,
                "state_gain": state_gain,
            }
            basin = basin_map(
                substrate_name=substrate,
                hidden_size=args.hidden_size,
                steps=args.steps,
                seeds=list(range(args.seed, args.seed + args.basin_seeds)),
                device=args.device,
                substrate_kwargs=cfg,
            )
            pert = perturbation_pair(
                substrate_name=substrate,
                hidden_size=args.hidden_size,
                steps=args.steps,
                seed=args.seed,
                perturb_step=args.perturb_step,
                perturb_scale=args.perturb_scale,
                device=args.device,
                substrate_kwargs=cfg,
            )
            mem = memory_pair(
                substrate_name=substrate,
                hidden_size=args.hidden_size,
                steps=args.steps,
                seed=args.seed,
                initial_delta=args.initial_delta,
                device=args.device,
                substrate_kwargs=cfg,
            )
            bott = bottleneck_run(
                substrate_name=substrate,
                hidden_size=args.hidden_size,
                steps=args.steps,
                seed=args.seed,
                bottleneck_dim=args.bottleneck_dim,
                bottleneck_interval=args.bottleneck_interval,
                device=args.device,
                substrate_kwargs=cfg,
            )
            couple = coupled_pair(
                substrate_name=substrate,
                hidden_size=args.hidden_size,
                steps=args.steps,
                seed=args.seed,
                coupling_dim=args.coupling_dim,
                coupling_interval=args.coupling_interval,
                coupling_strength=args.coupling_strength,
                device=args.device,
                substrate_kwargs=cfg,
            )
            score = regime_score(basin, pert, mem, bott, couple)
            row = {
                "substrate": substrate,
                **cfg,
                "score": score,
                "attractors": sorted({r.attractor_type for r in basin}),
                "interiors": sorted({r.interior_class for r in basin}),
                "mean_delta": sum(r.mean_delta for r in basin) / len(basin),
                "mean_cycle_period": sum(r.cycle_period for r in basin) / len(basin),
                "mean_covariance_rank": sum(r.covariance_rank for r in basin) / len(basin),
                "perturb_final_cosine": pert["final_cosine"],
                "memory_final_cosine": mem["final_cosine"],
                "bottleneck_unique_codes": bott["unique_codes"],
                "bottleneck_code_entropy": bott["code_entropy"],
                "coupling_initial_cosine": couple["initial_cosine"],
                "coupling_final_cosine": couple["final_cosine"],
                "coupling_mean_cosine": couple["mean_cosine"],
            }
            all_rows.append(row)

        sub_rows = [r for r in all_rows if r["substrate"] == substrate]
        sub_rows.sort(key=lambda r: r["score"], reverse=True)
        best = sub_rows[: min(3, len(sub_rows))]
        for row in best:
            print(
                "  score={:.3f} init={:.2f} fb={:.2f} gain={:.2f} "
                "cycle={:.2f} pert={:.4f} mem={:.4f} codes={} cfinal={:.4f} interior={}".format(
                    row["score"],
                    row["init_scale"],
                    row["feedback_scale"],
                    row["state_gain"],
                    row["mean_cycle_period"],
                    row["perturb_final_cosine"],
                    row["memory_final_cosine"],
                    row["bottleneck_unique_codes"],
                    row["coupling_final_cosine"],
                    ",".join(row["interiors"]),
                )
            )

    all_rows.sort(key=lambda r: r["score"], reverse=True)
    payload = {
        "config": {
            "hidden_size": args.hidden_size,
            "steps": args.steps,
            "basin_seeds": args.basin_seeds,
            "seed": args.seed,
            "init_scales": init_scales,
            "feedback_scales": feedback_scales,
            "state_gains": state_gains,
            "top_k": args.top_k,
        },
        "top": all_rows[: args.top_k],
        "all": all_rows,
    }
    out_path = out_dir / "ranked_regimes.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
