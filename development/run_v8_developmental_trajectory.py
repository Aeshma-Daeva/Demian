"""Probe v8 developmental trajectory: per-step channel evolution.

Measures channel norm, tightness, gate activity over steps 1-256.
Focus on representative seeds from population to understand what drives
recovery variation.

Usage:
  python3 development/run_v8_developmental_trajectory.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from development.substrate_lab import (
    DemianNativeV8Substrate,
    SelfLoopRunner,
)

HIDDEN_SIZE = 32
STEPS = 256
DEVICE = "cuda"
# Representative seeds: 105 (high recovery), 109 (low recovery), 99 (mid-low), 94 (mid-high)
SEEDS = [94, 99, 105, 109]
OUT_DIR = Path("data/substrate_lab/v8_development_trajectory_20260506")

CHANNEL_NAMES = [
    "fast", "slow", "carrier", "short_support", "packet",
    "control_short", "control_long", "tightness"
]


def run_trajectory(seed, steps=STEPS):
    """Run a single trajectory, capture per-step channel norms + tightness."""
    torch.manual_seed(seed)
    model = DemianNativeV8Substrate(HIDDEN_SIZE)
    model.to(DEVICE)
    model.eval()
    runner = SelfLoopRunner(model, device=DEVICE)

    state = model.initial_state(1, runner.device)
    trajectory = {name: [] for name in CHANNEL_NAMES}
    trajectory["state_norm"] = []
    trajectory["residual_delta"] = []
    trajectory["tightness_raw"] = []
    trajectory["state_gate"] = []  # gate = sigmoid from state_vector blending

    with torch.no_grad():
        for step_idx in range(1, steps + 1):
            state = model.step(state)
            fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state

            # Channel norms
            trajectory["fast"].append(fast.norm().item())
            trajectory["slow"].append(slow.norm().item())
            trajectory["carrier"].append(long_carrier.norm().item())
            trajectory["short_support"].append(short_support.norm().item())
            trajectory["packet"].append(packet.norm().item())
            trajectory["control_short"].append(control_short.norm().item())
            trajectory["control_long"].append(control_long.norm().item())
            trajectory["tightness"].append(tightness.item())
            trajectory["tightness_raw"].append(tightness.item())
            trajectory["state_norm"].append(
                model.state_vector(state).norm().item()
            )

            # Step aux for gates, release, residual delta
            aux = getattr(model, '_step_aux', {})
            if aux:
                trajectory["state_gate"].append(aux.get("tightness_expose_gate_mean", None))

    # Summary metrics
    summary = {
        "seed": seed,
        "steps": steps,
        "final_state_norm": trajectory["state_norm"][-1] if trajectory["state_norm"] else 0,
        "mean_state_norm": float(np.mean(trajectory["state_norm"])),
        "std_state_norm": float(np.std(trajectory["state_norm"])),
        "final_tightness": trajectory["tightness"][-1],
        "mean_tightness": float(np.mean(trajectory["tightness"])),
        "tightness_stability": float(np.std(trajectory["tightness"][-50:])),  # last 50 steps
        "channel_final_norms": {
            ch: trajectory[ch][-1] for ch in CHANNEL_NAMES
        },
        "channel_mean_norms": {
            ch: float(np.mean(trajectory[ch])) for ch in CHANNEL_NAMES
        },
        "channel_stability": {
            ch: float(np.std(trajectory[ch][-50:])) for ch in CHANNEL_NAMES
        },
    }

    # Critical period: where do channels differentiate from init noise?
    # Measure: steps until channel norms stabilize within 10% of final value
    for ch in CHANNEL_NAMES:
        vals = trajectory[ch]
        final = vals[-1] if vals else 1.0
        if final == 0:
            continue
        # Find first step where channel stays within 20% of final for 10+ steps
        stable_from = None
        for i in range(len(vals) - 10):
            window = vals[i:i+10]
            if all(abs(v - final) / (abs(final) + 1e-8) < 0.2 for v in window):
                stable_from = i + 1
                break
        summary["channel_" + ch + "_stabilize_step"] = stable_from

    return {
        "summary": summary,
        "trajectory": trajectory,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for seed in SEEDS:
        print("seed=" + str(seed))
        result = run_trajectory(seed)
        results[str(seed)] = result["summary"]

        # Save trajectory (large)
        traj_out = {}
        for k, v in result["trajectory"].items():
            if k == "state_gate":
                traj_out[k] = [x for x in v if x is not None]
                continue
            traj_out[k] = v
        with open(OUT_DIR / ("trajectory_seed_" + str(seed) + ".json"), "w") as f:
            json.dump(traj_out, f)

    # Print analysis
    print()
    print("=== Developmental Analysis ===")
    print()
    for s in SEEDS:
        r = results[str(s)]
        line = "seed=" + str(s) + ": "
        line += "final_norm=" + str(round(r["final_state_norm"], 4)) + " "
        line += "mean_norm=" + str(round(r["mean_state_norm"], 4)) + " "
        line += "final_tight=" + str(round(r["final_tightness"], 4)) + " "
        line += "tight_stab=" + str(round(r["tightness_stability"], 4))
        print(line)

        # Channel final norms sorted
        norms = r["channel_final_norms"]
        sorted_norms = sorted(norms.items(), key=lambda x: -x[1])
        print("  Top 3 channels (final norm):", [(ch, round(v, 4)) for ch, v in sorted_norms[:3]])

        # Stabilization steps
        stab = {k: v for k, v in r.items() if k.startswith("channel_") and k.endswith("_stabilize_step")}
        for ch, step in stab.items():
            name = ch.replace("channel_", "").replace("_stabilize_step", "")
            if step is not None:
                print("  " + name + " stabilizes at step=" + str(step))

        # Channel stability (last 50 steps)
        chan_stab = r["channel_stability"]
        sorted_stab = sorted(chan_stab.items(), key=lambda x: x[1])
        print("  Stablest channels:", [(ch, round(v, 6)) for ch, v in sorted_stab[:3]])
        print("  Least stable:", sorted_stab[-2:])
        print()

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "config": {"hidden_size": HIDDEN_SIZE, "steps": STEPS, "seeds": SEEDS, "device": DEVICE},
            "results": results,
        }, f, indent=2)
    print("Done: " + str(OUT_DIR / "summary.json"))


if __name__ == "__main__":
    main()
