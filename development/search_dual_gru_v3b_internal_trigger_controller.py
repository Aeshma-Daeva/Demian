#!/usr/bin/env python3
"""Search internal trigger controller schedules for dual_gru_v3b boundary regimes."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (
    save_json,
    search_dual_gru_v3b_internal_trigger_controller,
)


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Search dual_gru_v3b internal trigger controllers")
    p.add_argument("--substrate", default="dual_gru_v3b:edge")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--center-step", type=int, default=120)
    p.add_argument("--pulse-radii", default="4,6,8,10,12")
    p.add_argument("--pulse-strides", default="4,5,6")
    p.add_argument("--pulse-strengths", default="0.2,0.4,0.8,1.2")
    p.add_argument("--coupling-dims", default="8,16,32")
    p.add_argument("--objective", default="induce,suppress")
    p.add_argument("--no-scan-offsets", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="data/substrate_stress/dual_gru_v3b_internal_trigger_controller_search.json")
    return p.parse_args()


def main():
    args = parse_args()
    objectives = [x.strip() for x in args.objective.split(",") if x.strip()]
    payload = {"runs": []}
    for objective in objectives:
        run = search_dual_gru_v3b_internal_trigger_controller(
            substrate_name=args.substrate,
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            center_step=args.center_step,
            pulse_radii=_parse_ints(args.pulse_radii),
            pulse_strides=_parse_ints(args.pulse_strides),
            pulse_strengths=_parse_floats(args.pulse_strengths),
            coupling_dims=_parse_ints(args.coupling_dims),
            objective=objective,
            scan_offsets=not args.no_scan_offsets,
            device=args.device,
        )
        payload["runs"].append(run)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, out_path)

    print("dual_gru_v3b internal trigger controller search")
    for run in payload["runs"]:
        best = run["best_case"]
        print(f"  objective={run['objective']}")
        if best is None:
            print("    no cases")
            continue
        print(
            "    "
            f"best stride={best['pulse_stride']} radius={best['pulse_radius']} "
            f"offset={best['offset']} strength={best['pulse_strength']:.3f} "
            f"coupling_dim={best['coupling_dim']} "
            f"markers={best['self_triggered_entry_markers']}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
