#!/usr/bin/env python3
"""Report the controllable regime family of dual_gru_v3b."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import save_json, summarize_dual_gru_v3b_regimes


def parse_args():
    p = argparse.ArgumentParser(description="Report dual_gru_v3b regimes")
    p.add_argument("--regimes", nargs="*", default=["current", "edge", "tight", "threshold"])
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
    p.add_argument("--out", default="data/substrate_lab/dual_gru_v3b_regimes.json")
    return p.parse_args()


def main():
    args = parse_args()
    report = summarize_dual_gru_v3b_regimes(
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
        regimes=args.regimes,
    )

    print("dual_gru_v3b regime report")
    print(f"focus_regime: {report['focus_regime']}")
    for name, row in report["substrates"].items():
        print()
        print(f"[{name}]")
        print(f"  interiors:                 {row['interior_counts']}")
        print(f"  accumulating_fp_share:     {row['accumulating_fixed_point_share']:.4f}")
        print(f"  mean_delta:                {row['mean_delta']:.4f}")
        print(f"  mean_message_norm:         {row['mean_message_norm']:.4f}")
        print(f"  mean_message_contraction:  {row['mean_message_contraction']:.4f}")
        print(f"  bottleneck_code_entropy:   {row['mean_bottleneck_code_entropy']:.4f}")

    out_path = Path(args.out)
    save_json(report, out_path)
    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
