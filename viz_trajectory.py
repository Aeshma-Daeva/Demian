#!/usr/bin/env python3
"""Simple trajectory viewer: prints metrics table to terminal."""
import argparse
import json
import glob
import os

COLORS = {
    "focused": "\033[36m",
    "distributed": "\033[32m",
    "diffuse": "\033[33m",
}
RESET = "\033[0m"


def find_latest(data_dir="data"):
    pattern = os.path.join(data_dir, "trajectory_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No trajectory files in {data_dir}")
    return files[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to trajectory JSON")
    args = parser.parse_args()

    path = args.file or find_latest()
    with open(path) as f:
        traj = json.load(f)

    if not traj:
        print("Empty trajectory.")
        return

    n = len(traj)
    modes = [s["mode"] for s in traj]
    switches = sum(1 for i in range(1, n) if modes[i] != modes[i-1])
    avg_coh = sum(s["temporal_coherence"] for s in traj) / n
    rnorm_vals = [s["residual_norm"] for s in traj]
    kurt_vals = [s["kurtosis"] for s in traj]
    peak_vals = [s["n_peaks"] for s in traj]
    ent_vals = [s["entropy"] for s in traj]
    dom_vals = [s["dominance_ratio"] for s in traj]
    delta_vals = [s["residual_delta"] for s in traj]

    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    print(f"\n  Trajectory: {path}")
    print(f"  Steps: {n}  |  Mode switches: {switches}  |  Avg coherence: {avg_coh:.3f}")
    print(f"  Mode counts: {mode_counts}")
    print(f"  Residual norm: {min(rnorm_vals):.2f} - {max(rnorm_vals):.2f} (avg {sum(rnorm_vals)/n:.2f})")
    print(f"  Kurtosis: {min(kurt_vals):.0f} - {max(kurt_vals):.0f}")
    print(f"  Peaks: {min(peak_vals)} - {max(peak_vals)}")
    print(f"  Entropy: {min(ent_vals):.2f} - {max(ent_vals):.2f}")
    print(f"  Dominance ratio: {min(dom_vals):.1f} - {max(dom_vals):.1f}")
    print(f"  Residual delta: {min(delta_vals):.2f} - {max(delta_vals):.2f}")
    print()

    # Print summary at 4-step intervals
    print(f"  {'step':>4}  {'mode':>12}  {'r_norm':>7}  {'delta':>6}  {'kurt':>9}  {'peaks':>5}  {'ent':>7}  {'dom':>6}  {'coherence':>6}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*7}  {'─'*6}  {'─'*9}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*7}")
    for s in traj:
        if s["step"] % 4 == 0:
            mode = s["mode"].upper()
            color = COLORS.get(s["mode"], RESET)
            print(
                f"  {color}{s['step']:4d}  {mode:>12}  "
                f"{s['residual_norm']:7.2f}  {s['residual_delta']:6.2f}  "
                f"{s['kurtosis']:9.0f}  {s['n_peaks']:5d}  "
                f"{s['entropy']:7.2f}  {s['dominance_ratio']:6.1f}  "
                f"{s['temporal_coherence']:6.3f}{RESET}"
            )
    print()

    # Mini ASCII chart: residual norm over time
    print("  Residual norm over time:")
    width = 60
    r_min = min(rnorm_vals)
    r_max = max(rnorm_vals)
    r_range = r_max - r_min or 1
    samples = min(n, 80)
    step_size = n / samples
    for i in range(samples):
        idx = int(i * step_size)
        val = rnorm_vals[idx]
        bar_len = int((val - r_min) / r_range * width)
        print(f"  {idx:3d} |{'█' * bar_len}{'░' * (width - bar_len)}| {val:.2f}")


if __name__ == "__main__":
    main()
