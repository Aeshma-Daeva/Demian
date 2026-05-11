#!/usr/bin/env python3
"""Map interior transitions across dual_gru_v3b message-channel controls."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (
    map_dual_gru_v3b_message_transitions,
    save_json,
)


def _parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Map dual_gru_v3b message transitions")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=192)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--basin-seeds", type=int, default=4)
    p.add_argument("--message-self-retentions", default="0.0,0.36,0.72,1.08")
    p.add_argument("--slow-carry-scales", default="0.0,0.04,0.08,0.12")
    p.add_argument("--message-drive-scales", default="0.0,0.09,0.18,0.27")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="data/interior_transitions/dual_gru_v3b_message_transitions.json")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = list(range(args.seed, args.seed + args.basin_seeds))
    payload = map_dual_gru_v3b_message_transitions(
        hidden_size=args.hidden_size,
        steps=args.steps,
        seeds=seeds,
        message_self_retentions=_parse_grid(args.message_self_retentions),
        slow_carry_scales=_parse_grid(args.slow_carry_scales),
        message_drive_scales=_parse_grid(args.message_drive_scales),
        device=args.device,
    )

    out_path = Path(args.out)
    save_json(payload, out_path)

    print("dual_gru_v3b message transition map")
    for seed, counts in payload["transitions"].items():
        print(f"  seed {seed}: {counts}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
