#!/usr/bin/env python3
"""Compare coherent induction and trickster bypass modes for dual_gru_v3b."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import save_json, state_conditioned_self_trigger_pair
from development.substrate_lab import classify_dual_gru_v3b_autonomy_mode


def parse_args():
    p = argparse.ArgumentParser(description="Report dual_gru_v3b autonomy modes")
    p.add_argument("--substrate", default="dual_gru_v3b:edge")
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--coupling-dim", type=int, default=32)
    p.add_argument(
        "--induce-search",
        default="data/substrate_stress/dual_gru_v3b_state_trigger_controller_search_recalibrated.json",
    )
    p.add_argument(
        "--cheat-search",
        default="data/substrate_stress/dual_gru_v3b_state_trigger_controller_search_cheat.json",
    )
    p.add_argument(
        "--suppress-search",
        default="data/substrate_stress/dual_gru_v3b_state_trigger_controller_search_recalibrated.json",
    )
    p.add_argument(
        "--out",
        default="data/substrate_stress/dual_gru_v3b_autonomy_modes.json",
    )
    return p.parse_args()


def load_best_case(path: Path, objective: str) -> dict:
    data = json.loads(path.read_text())
    run = next(r for r in data["runs"] if r["objective"] == objective)
    return run["best_case"]


def compress_runs(steps: list[int]) -> list[dict[str, int]]:
    if not steps:
        return []
    runs = []
    start = prev = steps[0]
    for step in steps[1:]:
        if step == prev + 1:
            prev = step
            continue
        runs.append({"start": start, "end": prev, "length": prev - start + 1})
        start = prev = step
    runs.append({"start": start, "end": prev, "length": prev - start + 1})
    return runs


def dominant_gains(params: dict) -> list[dict[str, float | str]]:
    keys = [
        "fast_norm_gain",
        "slow_norm_gain",
        "message_norm_gain",
        "fast_delta_gain",
        "message_delta_gain",
        "slow_delta_gain",
        "message_slow_tension_gain",
        "message_slow_ratio_gain",
        "slow_growth_pressure_gain",
        "slow_message_gap_gain",
        "boundary_pressure_gain",
        "delta_skew_gain",
        "controller_state_gain",
        "window_gain",
        "phase_sin",
        "phase_cos",
        "output_state_gain",
        "output_window_gain",
    ]
    rows = []
    for key in keys:
        if key not in params:
            continue
        value = float(params[key])
        rows.append(
            {
                "name": key,
                "value": value,
                "abs_value": abs(value),
                "sign": "positive" if value >= 0.0 else "negative",
            }
        )
    rows.sort(key=lambda row: row["abs_value"], reverse=True)
    return rows[:8]


def mean_observer_on_triggers(mapping: dict) -> dict[str, float]:
    trace = mapping["observer_trace"]
    steps = mapping["trigger_steps"]
    if not steps:
        return {}
    keys = trace[0].keys()
    out = {}
    for key in keys:
        vals = [float(trace[s - 1][key]) for s in steps]
        out[key] = sum(vals) / len(vals)
    return out


def summarize_mode(name: str, mapping: dict, best_case: dict) -> dict:
    classification = classify_dual_gru_v3b_autonomy_mode(mapping)
    return {
        "mode": name,
        "autonomy_class": classification["mode"],
        "autonomy_rationale": classification["rationale"],
        "markers": mapping["state_triggered"]["entry_markers"],
        "trigger_steps": mapping["trigger_steps"],
        "trigger_runs": compress_runs(mapping["trigger_steps"]),
        "active_steps": len(mapping["trigger_steps"]),
        "duty_cycle": best_case.get("duty_cycle"),
        "mean_strength": best_case.get("mean_strength"),
        "mean_controller_state": best_case.get("mean_controller_state"),
        "mean_observer_on_triggers": mean_observer_on_triggers(mapping),
        "dominant_gains": dominant_gains(best_case["controller_params"]),
        "controller_params": best_case["controller_params"],
        "classification": classification,
    }


def main():
    args = parse_args()
    induce_best = load_best_case(Path(args.induce_search), "induce")
    cheat_best = load_best_case(Path(args.cheat_search), "cheat")
    suppress_best = load_best_case(Path(args.suppress_search), "suppress")

    induce_map = state_conditioned_self_trigger_pair(
        args.substrate,
        hidden_size=args.hidden_size,
        steps=args.steps,
        seed=args.seed,
        controller_params=induce_best["controller_params"],
        coupling_dim=args.coupling_dim,
        device="cpu",
    )
    cheat_map = state_conditioned_self_trigger_pair(
        args.substrate,
        hidden_size=args.hidden_size,
        steps=args.steps,
        seed=args.seed,
        controller_params=cheat_best["controller_params"],
        coupling_dim=args.coupling_dim,
        device="cpu",
    )
    suppress_map = state_conditioned_self_trigger_pair(
        args.substrate,
        hidden_size=args.hidden_size,
        steps=args.steps,
        seed=args.seed,
        controller_params=suppress_best["controller_params"],
        coupling_dim=args.coupling_dim,
        device="cpu",
    )

    payload = {
        "substrate": args.substrate,
        "induce": summarize_mode("induce", induce_map, induce_best),
        "cheat": summarize_mode("cheat", cheat_map, cheat_best),
        "suppress": summarize_mode("suppress", suppress_map, suppress_best),
    }
    out_path = Path(args.out)
    save_json(payload, out_path)

    print("dual_gru_v3b autonomy modes")
    for key in ["induce", "cheat", "suppress"]:
        mode = payload[key]
        print(f"  {key}: class={mode['autonomy_class']} markers={mode['markers']} active={mode['active_steps']} runs={mode['trigger_runs'][:4]}")
        print(f"    dominant_gains={[row['name'] for row in mode['dominant_gains'][:5]]}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
