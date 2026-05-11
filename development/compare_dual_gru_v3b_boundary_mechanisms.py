#!/usr/bin/env python3
"""Compare edge/threshold boundary mechanisms at specific perturbation steps."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import save_json, trajectory_map_pair


def parse_args():
    p = argparse.ArgumentParser(description="Compare dual_gru_v3b boundary mechanisms")
    p.add_argument("--substrates", default="dual_gru_v3b:edge,dual_gru_v3b:threshold")
    p.add_argument("--perturb-steps", default="112,120,128")
    p.add_argument("--perturb-scale", type=float, default=0.2)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--window-radius", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--out",
        default="data/substrate_stress/dual_gru_v3b_boundary_mechanism_compare.json",
    )
    return p.parse_args()


def parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_csv_text(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def find_step(traj: list[dict[str, float]], step: int) -> dict[str, float]:
    return traj[step - 1]


def local_window(traj: list[dict[str, float]], center_step: int, radius: int) -> list[dict[str, float]]:
    lo = max(1, center_step - radius)
    hi = min(len(traj), center_step + radius)
    out = []
    for step in range(lo, hi + 1):
        row = traj[step - 1]
        out.append(
            {
                "step": step,
                "fast_state_delta": row["fast_state_delta"],
                "slow_state_delta": row["slow_state_delta"],
                "message_state_delta": row["message_state_delta"],
                "fast_state_norm": row["fast_state_norm"],
                "slow_state_norm": row["slow_state_norm"],
                "message_state_norm": row["message_state_norm"],
            }
        )
    return out


def immediate_spike_signature(
    clean_traj: list[dict[str, float]],
    pert_traj: list[dict[str, float]],
    perturb_step: int,
) -> dict[str, dict[str, float]]:
    clean = find_step(clean_traj, perturb_step)
    pert = find_step(pert_traj, perturb_step)
    return {
        "fast": {
            "clean_delta": clean["fast_state_delta"],
            "perturbed_delta": pert["fast_state_delta"],
            "jump_ratio": pert["fast_state_delta"] / (clean["fast_state_delta"] + 1e-10),
        },
        "slow": {
            "clean_delta": clean["slow_state_delta"],
            "perturbed_delta": pert["slow_state_delta"],
            "jump_ratio": pert["slow_state_delta"] / (clean["slow_state_delta"] + 1e-10),
        },
        "message": {
            "clean_delta": clean["message_state_delta"],
            "perturbed_delta": pert["message_state_delta"],
            "jump_ratio": pert["message_state_delta"] / (clean["message_state_delta"] + 1e-10),
        },
    }


def mechanism_summary(clean_markers: dict[str, int | None], pert_markers: dict[str, int | None]) -> dict[str, object]:
    keys = (
        "message_contraction_takeoff_step",
        "message_norm_takeoff_step",
        "slow_norm_takeoff_step",
    )
    shifts = {}
    for key in keys:
        clean = clean_markers.get(key)
        pert = pert_markers.get(key)
        if clean is None or pert is None:
            shifts[key] = None
        else:
            shifts[key] = pert - clean

    return {
        "clean_marker_order": [key for key in keys if clean_markers.get(key) is not None],
        "perturbed_marker_order": [key for key in keys if pert_markers.get(key) is not None],
        "marker_shifts": shifts,
        "preserved_message_norm_takeoff": (
            clean_markers.get("message_norm_takeoff_step") is not None
            and clean_markers.get("message_norm_takeoff_step") == pert_markers.get("message_norm_takeoff_step")
        ),
        "created_message_norm_takeoff": (
            clean_markers.get("message_norm_takeoff_step") is None
            and pert_markers.get("message_norm_takeoff_step") is not None
        ),
        "created_slow_norm_takeoff": (
            clean_markers.get("slow_norm_takeoff_step") is None
            and pert_markers.get("slow_norm_takeoff_step") is not None
        ),
    }


def compare_case(
    substrate: str,
    hidden_size: int,
    steps: int,
    seed: int,
    perturb_step: int,
    perturb_scale: float,
    window_radius: int,
    device: str,
) -> dict[str, object]:
    mapping = trajectory_map_pair(
        substrate_name=substrate,
        hidden_size=hidden_size,
        steps=steps,
        seed=seed,
        perturb_step=perturb_step,
        perturb_scale=perturb_scale,
        device=device,
    )
    clean = mapping["clean"]
    pert = mapping["perturbed"]
    return {
        "substrate": substrate,
        "perturb_step": perturb_step,
        "perturb_scale": perturb_scale,
        "clean_entry_markers": clean["entry_markers"],
        "perturbed_entry_markers": pert["entry_markers"],
        "mechanism": mechanism_summary(clean["entry_markers"], pert["entry_markers"]),
        "immediate_spike": immediate_spike_signature(clean["trajectory"], pert["trajectory"], perturb_step),
        "local_window": {
            "clean": local_window(clean["trajectory"], perturb_step, window_radius),
            "perturbed": local_window(pert["trajectory"], perturb_step, window_radius),
        },
        "clean_summary": clean["summary"],
        "perturbed_summary": pert["summary"],
    }


def main():
    args = parse_args()
    substrates = parse_csv_text(args.substrates)
    perturb_steps = parse_csv_ints(args.perturb_steps)

    cases = []
    for substrate in substrates:
        for perturb_step in perturb_steps:
            cases.append(
                compare_case(
                    substrate=substrate,
                    hidden_size=args.hidden_size,
                    steps=args.steps,
                    seed=args.seed,
                    perturb_step=perturb_step,
                    perturb_scale=args.perturb_scale,
                    window_radius=args.window_radius,
                    device=args.device,
                )
            )

    out = {
        "seed": args.seed,
        "steps": args.steps,
        "hidden_size": args.hidden_size,
        "perturb_scale": args.perturb_scale,
        "perturb_steps": perturb_steps,
        "substrates": substrates,
        "cases": cases,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, out_path)

    print("dual_gru_v3b boundary mechanism comparison")
    print(f"  substrates={substrates}")
    print(f"  perturb_steps={perturb_steps}")
    print(f"  perturb_scale={args.perturb_scale}")
    for case in cases:
        mech = case["mechanism"]
        print(
            "  "
            f"{case['substrate']} step={case['perturb_step']} "
            f"preserved_msg={mech['preserved_message_norm_takeoff']} "
            f"created_msg={mech['created_message_norm_takeoff']} "
            f"created_slow={mech['created_slow_norm_takeoff']}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
