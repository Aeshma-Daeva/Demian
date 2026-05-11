#!/usr/bin/env python3
"""Scan perturbation timing anomalies in the dual_gru_v3b edge regime."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import save_json, scan_dual_gru_v3b_edge_anomalies


def _parse_int_grid(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_float_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Scan dual_gru_v3b:edge anomalies")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--perturb-steps", default="16,32,48,64,80,96,112,128,160,192,224")
    p.add_argument("--perturb-scales", default="0.02,0.05,0.1,0.15,0.2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="data/substrate_stress/dual_gru_v3b_edge_anomalies.json")
    return p.parse_args()


def main():
    args = parse_args()
    payload = scan_dual_gru_v3b_edge_anomalies(
        hidden_size=args.hidden_size,
        steps=args.steps,
        seed=args.seed,
        perturb_steps=_parse_int_grid(args.perturb_steps),
        perturb_scales=_parse_float_grid(args.perturb_scales),
        device=args.device,
    )

    out_path = Path(args.out)
    save_json(payload, out_path)

    print("dual_gru_v3b edge anomaly scan")
    for row in payload["top_anomalies"][:10]:
        print(
            "  step={:>3} scale={:.3f} final_cos={:.4f} msg_ratio={:.3f} marker_shift={:.1f} score={:.3f}".format(
                row["perturb_step"],
                row["perturb_scale"],
                row["final_cosine"],
                row["peak_message_ratio"],
                row["marker_shift"],
                row["anomaly_score"],
            )
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
