#!/usr/bin/env python3
"""Targeted stress tests for substrate residual-risk diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (
    coupling_stress_test,
    perturbation_stress_test,
    save_json,
    trajectory_map_pair,
)


def _parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Run targeted substrate stress tests")
    p.add_argument("--substrate", default="dual_gru_v3b:current")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--perturb-step", type=int, default=128)
    p.add_argument("--perturb-scales", default="0.02,0.05,0.1,0.2")
    p.add_argument("--perturb-mode", choices=["noise", "rss_negation", "surface_negation"], default="noise")
    p.add_argument("--coupling-dim", type=int, default=8)
    p.add_argument("--coupling-interval", type=int, default=16)
    p.add_argument("--coupling-strengths", default="0.02,0.05,0.1,0.2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="data/substrate_stress")
    p.add_argument("--save-trajectory-map", action="store_true")
    p.add_argument("--trajectory-perturb-scale", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    perturb = perturbation_stress_test(
        substrate_name=args.substrate,
        hidden_size=args.hidden_size,
        steps=args.steps,
        seed=args.seed,
        perturb_step=args.perturb_step,
        perturb_scales=_parse_grid(args.perturb_scales),
        perturb_mode=args.perturb_mode,
        device=args.device,
    )
    coupling = coupling_stress_test(
        substrate_name=args.substrate,
        hidden_size=args.hidden_size,
        steps=args.steps,
        seed=args.seed,
        coupling_dim=args.coupling_dim,
        coupling_interval=args.coupling_interval,
        coupling_strengths=_parse_grid(args.coupling_strengths),
        device=args.device,
    )

    payload = {
        "substrate": args.substrate,
        "config": {
            "hidden_size": args.hidden_size,
            "steps": args.steps,
            "seed": args.seed,
            "perturb_step": args.perturb_step,
            "perturb_scales": _parse_grid(args.perturb_scales),
            "perturb_mode": args.perturb_mode,
            "coupling_dim": args.coupling_dim,
            "coupling_interval": args.coupling_interval,
            "coupling_strengths": _parse_grid(args.coupling_strengths),
            "device": args.device,
        },
        "perturbation_stress": perturb,
        "coupling_stress": coupling,
    }

    save_json(payload, out_dir / f"{args.substrate}_stress.json")

    if args.save_trajectory_map:
        mapping = trajectory_map_pair(
            substrate_name=args.substrate,
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            perturb_step=args.perturb_step,
            perturb_scale=args.trajectory_perturb_scale,
            perturb_mode=args.perturb_mode,
            device=args.device,
        )
        save_json(
            mapping,
            out_dir / f"{args.substrate}_trajectory_map.json",
        )

    print("Substrate stress tests")
    print(f"out_dir: {out_dir}")
    print(f"[{args.substrate}]")
    for case in perturb["cases"]:
        print(
            "  perturb scale={:.3f} final_cos={:.4f} peak(msg)={:.3f} peak_ratio(msg)={:.3f}".format(
                case["perturb_scale"],
                case["final_cosine_vs_baseline"],
                case["summary"]["max_message_norm"],
                case["peak_norm_ratio_vs_baseline"]["message"],
            )
        )
    for case in coupling["cases"]:
        print(
            "  coupling strength={:.3f} final_cos={:.4f} peak(msg)={:.3f}".format(
                case["coupling_strength"],
                case["final_cosine"],
                case["peak_component_norms"]["message"],
            )
        )
    if args.save_trajectory_map:
        print("  trajectory map saved for perturb mode={} scale={:.3f}".format(args.perturb_mode, args.trajectory_perturb_scale))
    print(f"Saved: {out_dir / f'{args.substrate}_stress.json'}")


if __name__ == "__main__":
    main()
