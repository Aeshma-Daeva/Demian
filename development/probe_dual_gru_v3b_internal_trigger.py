#!/usr/bin/env python3
"""Probe internal self-trigger schedules for dual_gru_v3b boundary regimes."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import save_json, self_coupling_schedule_pair


def parse_args():
    p = argparse.ArgumentParser(description="Probe dual_gru_v3b self-trigger schedules")
    p.add_argument("--substrate", default="dual_gru_v3b:edge")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--center-step", type=int, default=120)
    p.add_argument("--pulse-radius", type=int, default=12)
    p.add_argument("--pulse-strides", default="4,5,6")
    p.add_argument("--pulse-scale", type=float, default=0.2)
    p.add_argument("--coupling-dim", type=int, default=8)
    p.add_argument("--scan-offsets", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="data/substrate_stress/dual_gru_v3b_internal_trigger_probe.json")
    return p.parse_args()


def parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def build_schedule(center_step: int, pulse_radius: int, pulse_stride: int, pulse_scale: float, offset: int, steps: int) -> dict[int, float]:
    lo = max(1, center_step - pulse_radius + offset)
    hi = min(steps, center_step + pulse_radius)
    return {step: pulse_scale for step in range(lo, hi + 1, pulse_stride)}


def main():
    args = parse_args()
    cases = []
    strides = parse_csv_ints(args.pulse_strides)

    for stride in strides:
        offsets = range(stride) if args.scan_offsets else [0]
        for offset in offsets:
            schedule = build_schedule(
                center_step=args.center_step,
                pulse_radius=args.pulse_radius,
                pulse_stride=stride,
                pulse_scale=args.pulse_scale,
                offset=offset,
                steps=args.steps,
            )
            mapping = self_coupling_schedule_pair(
                substrate_name=args.substrate,
                hidden_size=args.hidden_size,
                steps=args.steps,
                seed=args.seed,
                trigger_schedule=schedule,
                coupling_dim=args.coupling_dim,
                device=args.device,
            )
            clean = mapping["clean"]
            trig = mapping["self_triggered"]
            cases.append(
                {
                    "pulse_stride": stride,
                    "offset": offset,
                    "schedule": schedule,
                    "clean_entry_markers": clean["entry_markers"],
                    "self_triggered_entry_markers": trig["entry_markers"],
                    "clean_summary": clean["summary"],
                    "self_triggered_summary": trig["summary"],
                }
            )

    out = {
        "substrate": args.substrate,
        "seed": args.seed,
        "steps": args.steps,
        "hidden_size": args.hidden_size,
        "center_step": args.center_step,
        "pulse_radius": args.pulse_radius,
        "pulse_strides": strides,
        "pulse_scale": args.pulse_scale,
        "coupling_dim": args.coupling_dim,
        "scan_offsets": args.scan_offsets,
        "cases": cases,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, out_path)

    print("dual_gru_v3b internal trigger probe")
    print(f"  substrate={args.substrate}")
    print(f"  center_step={args.center_step} radius={args.pulse_radius} scale={args.pulse_scale}")
    for case in cases:
        print(
            "  "
            f"stride={case['pulse_stride']} offset={case['offset']} "
            f"markers={case['self_triggered_entry_markers']}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
