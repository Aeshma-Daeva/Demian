#!/usr/bin/env python3
"""Probe and amplify the dual_gru_v3b edge boundary around specific steps."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (
    amplify_dual_gru_v3b_edge_window,
    probe_dual_gru_v3b_edge_step,
    save_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Probe dual_gru_v3b:edge local boundary window")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--perturb-step", type=int, default=120)
    p.add_argument("--perturb-scale", type=float, default=0.2)
    p.add_argument("--window-radius", type=int, default=12)
    p.add_argument("--amplify", action="store_true")
    p.add_argument("--pulse-radius", type=int, default=12)
    p.add_argument("--pulse-stride", type=int, default=4)
    p.add_argument("--pulse-scale", type=float, default=0.12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="data/substrate_stress")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_dual_gru_v3b_edge_step(
        hidden_size=args.hidden_size,
        steps=args.steps,
        seed=args.seed,
        perturb_step=args.perturb_step,
        perturb_scale=args.perturb_scale,
        window_radius=args.window_radius,
        device=args.device,
    )
    probe_path = out_dir / f"dual_gru_v3b_edge_probe_step{args.perturb_step}_scale{str(args.perturb_scale).replace('.', 'p')}.json"
    save_json(probe, probe_path)

    print("dual_gru_v3b edge probe")
    print(f"  step={args.perturb_step} scale={args.perturb_scale}")
    print(f"  clean markers: {probe['clean_entry_markers']}")
    print(f"  pert  markers: {probe['perturbed_entry_markers']}")
    print(f"Saved: {probe_path}")

    if args.amplify:
        amp = amplify_dual_gru_v3b_edge_window(
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            center_step=args.perturb_step,
            pulse_radius=args.pulse_radius,
            pulse_stride=args.pulse_stride,
            pulse_scale=args.pulse_scale,
            device=args.device,
        )
        amp_path = out_dir / f"dual_gru_v3b_edge_amplify_step{args.perturb_step}_scale{str(args.pulse_scale).replace('.', 'p')}.json"
        save_json(amp, amp_path)
        print("dual_gru_v3b edge amplification")
        print(f"  schedule: {amp['perturb_schedule']}")
        print(f"  clean markers: {amp['clean_entry_markers']}")
        print(f"  pert  markers: {amp['perturbed_entry_markers']}")
        print(f"  peak_message_ratio: {amp['peak_message_ratio']:.3f}")
        print(f"  marker_shift: {amp['marker_shift']:.1f}")
        print(f"Saved: {amp_path}")


if __name__ == "__main__":
    main()
