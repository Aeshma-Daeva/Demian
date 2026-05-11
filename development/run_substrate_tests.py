#!/usr/bin/env python3
"""Run the self-loop substrate empirical battery."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (
    basin_map,
    bottleneck_run,
    coupled_pair,
    list_substrate_specs,
    memory_pair,
    perturbation_pair,
    save_json,
    summarize_dual_gru_family,
    summarize_dual_gru_v3b_regimes,
    summarize_by_interior_class,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run self-loop substrate tests")
    p.add_argument(
        "--substrates",
        nargs="*",
        default=[
            "dual_gru_v3b:current",
            "dual_gru_v3b:tight",
            "dual_gru_v3b:threshold",
            "dual_gru_v3",
        ],
    )
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--basin-seeds", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--perturb-step", type=int, default=128)
    p.add_argument("--perturb-scale", type=float, default=0.05)
    p.add_argument("--initial-delta", type=float, default=0.05)
    p.add_argument("--bottleneck-dim", type=int, default=8)
    p.add_argument("--bottleneck-interval", type=int, default=16)
    p.add_argument("--coupling-dim", type=int, default=8)
    p.add_argument("--coupling-interval", type=int, default=16)
    p.add_argument("--coupling-strength", type=float, default=0.05)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="data/substrate_lab")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "config": {
            "hidden_size": args.hidden_size,
            "steps": args.steps,
            "basin_seeds": args.basin_seeds,
            "seed": args.seed,
            "perturb_step": args.perturb_step,
            "perturb_scale": args.perturb_scale,
            "initial_delta": args.initial_delta,
            "bottleneck_dim": args.bottleneck_dim,
            "bottleneck_interval": args.bottleneck_interval,
            "coupling_dim": args.coupling_dim,
            "coupling_interval": args.coupling_interval,
            "coupling_strength": args.coupling_strength,
            "device": args.device,
        },
        "substrates": {},
    }

    print("Substrate empirical battery")
    print(f"out_dir: {out_dir}")

    for name in args.substrates:
        if name not in list_substrate_specs():
            raise SystemExit(f"Unknown substrate: {name}")

        print()
        print(f"[{name}]")
        basin = basin_map(
            substrate_name=name,
            hidden_size=args.hidden_size,
            steps=args.steps,
            seeds=list(range(args.seed, args.seed + args.basin_seeds)),
            device=args.device,
        )
        pert = perturbation_pair(
            substrate_name=name,
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            perturb_step=args.perturb_step,
            perturb_scale=args.perturb_scale,
            device=args.device,
        )
        mem = memory_pair(
            substrate_name=name,
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            initial_delta=args.initial_delta,
            device=args.device,
        )
        bott = bottleneck_run(
            substrate_name=name,
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            bottleneck_dim=args.bottleneck_dim,
            bottleneck_interval=args.bottleneck_interval,
            device=args.device,
        )
        couple = coupled_pair(
            substrate_name=name,
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            coupling_dim=args.coupling_dim,
            coupling_interval=args.coupling_interval,
            coupling_strength=args.coupling_strength,
            device=args.device,
        )
        class_summary = summarize_by_interior_class(
            substrate_name=name,
            hidden_size=args.hidden_size,
            steps=args.steps,
            seeds=list(range(args.seed, args.seed + args.basin_seeds)),
            perturb_step=args.perturb_step,
            perturb_scale=args.perturb_scale,
            initial_delta=args.initial_delta,
            bottleneck_dim=args.bottleneck_dim,
            bottleneck_interval=args.bottleneck_interval,
            coupling_dim=args.coupling_dim,
            coupling_interval=args.coupling_interval,
            coupling_strength=args.coupling_strength,
            device=args.device,
        )

        summary = {
            "basin": [asdict(b) for b in basin],
            "by_interior_class": class_summary,
            "perturbation": pert,
            "memory": mem,
            "bottleneck": bott,
            "coupling": couple,
        }
        aggregate["substrates"][name] = summary
        save_json(summary, out_dir / f"{name}.json")

        attractors = {}
        interiors = {}
        for row in basin:
            attractors[row.attractor_type] = attractors.get(row.attractor_type, 0) + 1
            interiors[row.interior_class] = interiors.get(row.interior_class, 0) + 1

        mean_norm = sum(r.mean_norm for r in basin) / len(basin)
        mean_delta = sum(r.mean_delta for r in basin) / len(basin)
        mean_cycle = sum(r.cycle_period for r in basin) / len(basin)

        print(f"  basins:            {len(basin)}")
        print(f"  attractors:        {attractors}")
        print(f"  interiors:         {interiors}")
        print(f"  mean_norm:         {mean_norm:.4f}")
        print(f"  mean_delta:        {mean_delta:.4f}")
        print(f"  mean_cycle_period: {mean_cycle:.2f}")
        print(f"  perturb final cos: {pert['final_cosine']:.4f}")
        print(f"  memory final cos:  {mem['final_cosine']:.4f}")
        print(f"  bottleneck reuse:  {bott['reuse_ratio']:.4f}")
        print(f"  coupled final cos: {couple['final_cosine']:.4f}")
        if class_summary:
            print(f"  by interior:       {class_summary}")

    save_json(aggregate, out_dir / "summary.json")
    dual_gru_family = summarize_dual_gru_family(
        hidden_size=args.hidden_size,
        steps=args.steps,
        seeds=list(range(args.seed, args.seed + args.basin_seeds)),
        perturb_step=args.perturb_step,
        perturb_scale=args.perturb_scale,
        initial_delta=args.initial_delta,
        bottleneck_dim=args.bottleneck_dim,
        bottleneck_interval=args.bottleneck_interval,
        coupling_dim=args.coupling_dim,
        coupling_interval=args.coupling_interval,
        coupling_strength=args.coupling_strength,
        device=args.device,
        family=[name for name in args.substrates if name.startswith("dual_gru")],
    )
    save_json(dual_gru_family, out_dir / "dual_gru_family_summary.json")
    v3b_regimes = summarize_dual_gru_v3b_regimes(
        hidden_size=args.hidden_size,
        steps=args.steps,
        seeds=list(range(args.seed, args.seed + args.basin_seeds)),
        perturb_step=args.perturb_step,
        perturb_scale=args.perturb_scale,
        initial_delta=args.initial_delta,
        bottleneck_dim=args.bottleneck_dim,
        bottleneck_interval=args.bottleneck_interval,
        coupling_dim=args.coupling_dim,
        coupling_interval=args.coupling_interval,
        coupling_strength=args.coupling_strength,
        device=args.device,
    )
    save_json(v3b_regimes, out_dir / "dual_gru_v3b_regimes.json")
    print()
    print(f"Saved: {out_dir / 'summary.json'}")
    print(f"Saved: {out_dir / 'dual_gru_family_summary.json'}")
    print(f"Saved: {out_dir / 'dual_gru_v3b_regimes.json'}")


if __name__ == "__main__":
    main()
