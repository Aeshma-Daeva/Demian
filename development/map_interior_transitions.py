#!/usr/bin/env python3
"""Map fixed-point interior class transitions across local parameter neighborhoods."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import map_interior_class_transitions, save_json


def _parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Map interior class transitions")
    p.add_argument("--substrate", default="dual_gru_v3b:current")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=192)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--basin-seeds", type=int, default=4)
    p.add_argument("--init-scales", default="0.5")
    p.add_argument("--feedback-scales", default="0.8,1.0,1.2")
    p.add_argument("--state-gains", default="0.9,1.0,1.1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="data/interior_transitions")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed, args.seed + args.basin_seeds))
    payload = map_interior_class_transitions(
        substrate_name=args.substrate,
        hidden_size=args.hidden_size,
        steps=args.steps,
        seeds=seeds,
        init_scales=_parse_grid(args.init_scales),
        feedback_scales=_parse_grid(args.feedback_scales),
        state_gains=_parse_grid(args.state_gains),
        device=args.device,
    )

    save_json(payload, out_dir / f"{args.substrate}_interior_transitions.json")

    print("Interior transition map")
    print(f"out_dir: {out_dir}")
    print(f"[{args.substrate}]")
    for seed, counts in payload["transitions"].items():
        print(f"  seed {seed}: {counts}")
    print(f"Saved: {out_dir / f'{args.substrate}_interior_transitions.json'}")


if __name__ == "__main__":
    main()
