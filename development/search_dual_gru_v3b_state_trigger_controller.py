#!/usr/bin/env python3
"""Search state-conditioned internal trigger controllers for dual_gru_v3b."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (
    save_json,
    search_dual_gru_v3b_state_trigger_controller,
)


def parse_args():
    p = argparse.ArgumentParser(description="Search dual_gru_v3b state trigger controllers")
    p.add_argument("--substrate", default="dual_gru_v3b:edge")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--coupling-dim", type=int, default=32)
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--objective", default="sparse_induce,liminal_cheat,suppress")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="data/substrate_stress/dual_gru_v3b_state_trigger_controller_search.json")
    return p.parse_args()


def main():
    args = parse_args()
    objectives = [x.strip() for x in args.objective.split(",") if x.strip()]
    payload = {"runs": []}
    for objective in objectives:
        payload["runs"].append(
            search_dual_gru_v3b_state_trigger_controller(
                substrate_name=args.substrate,
                hidden_size=args.hidden_size,
                steps=args.steps,
                seed=args.seed,
                objective=objective,
                coupling_dim=args.coupling_dim,
                trials=args.trials,
                device=args.device,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(payload, out_path)

    print("dual_gru_v3b state trigger controller search")
    for run in payload["runs"]:
        best = run["best_case"]
        print(f"  objective={run['objective']}")
        if best is None:
            print("    no cases")
            continue
        print(
            "    "
            f"mode={best.get('mode')} "
            f"score={best['score']:.3f} markers={best['markers']} "
            f"active_steps={best['active_steps']} duty_cycle={best['duty_cycle']:.3f} "
            f"mean_strength={best['mean_strength']:.3f} mean_controller_state={best['mean_controller_state']:.3f}"
        )
        print(f"    params={best['controller_params']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
