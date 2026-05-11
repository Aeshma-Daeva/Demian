"""Trace seed-95 collapse through native lineage: v3, v6, v7, v7.1, v7.2, v7.4."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from development.substrate_lab import (
    perturbation_stress_test,
    coupling_stress_test,
)

HIDDEN_SIZE = 32
STEPS = 128
SEED = 95
PERTURB_SCALES = [0.25, 0.5, 0.75, 1.0]
PERTURB_STEP = 64
COUPLING_STRENGTHS = [0.02, 0.05, 0.1, 0.2]
COUPLING_DIM = 8
COUPLING_INTERVAL = 16
DEVICE = "cuda"
OUT_DIR = Path("data/substrate_lab/v74_lineage_collapse_20260505")

SUBSTRATES = [
    "demian_native_v3",
    "demian_native_v6",
    "demian_native_v7",
    "demian_native_v7.1",
    "demian_native_v7.2",
    "demian_native_v7.4",
    "demian_native_v8",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for name in SUBSTRATES:
        print(f"\n=== {name} ===", flush=True)
        torch.cuda.empty_cache()

        pert = perturbation_stress_test(
            substrate_name=name,
            hidden_size=HIDDEN_SIZE,
            steps=STEPS,
            seed=SEED,
            perturb_step=PERTURB_STEP,
            perturb_scales=PERTURB_SCALES,
            perturb_mode="rss_negation",
            device=DEVICE,
        )
        coup = coupling_stress_test(
            substrate_name=name,
            hidden_size=HIDDEN_SIZE,
            steps=STEPS,
            seed=SEED,
            coupling_dim=COUPLING_DIM,
            coupling_interval=COUPLING_INTERVAL,
            coupling_strengths=COUPLING_STRENGTHS,
            device=DEVICE,
        )

        bl = pert["baseline"]
        results[name] = {
            "baseline": {
                "attractor_type": bl["attractor_type"],
                "interior_class": bl["interior_class"],
                "cov_rank": bl["covariance_rank"],
                "compression": bl["compression_ratio"],
                "coherence": bl["mean_coherence"],
                "flow_dim": bl["flow_dimension"],
                "cycle_period": bl["cycle_period"],
                "two_cycle_amplitude": bl["two_cycle_amplitude"],
            },
            "perturb_recovery": {
                str(c["perturb_scale"]): {
                    "final_cos": c["final_cosine_vs_baseline"],
                    "peak_msg_ratio": c["peak_norm_ratio_vs_baseline"]["message"],
                }
                for c in pert["cases"]
            },
            "coupling_final": {
                str(c["coupling_strength"]): c["final_cosine"]
                for c in coup["cases"]
            },
        }
        print(f"    perturb rec: {results[name]['perturb_recovery']}", flush=True)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "config": {"hidden_size": HIDDEN_SIZE, "steps": STEPS, "seed": SEED, "device": DEVICE},
            "substrates": SUBSTRATES,
            "results": results,
        }, f, indent=2)
    print(f"\nDone: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
