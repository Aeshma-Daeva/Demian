"""v8 genotype substrate vs v7.4 stress comparison."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from development.substrate_lab import (
    DemianNativeV8Substrate,
    DemianNativeV74Substrate,
    SelfLoopRunner,
    perturbation_stress_test,
    coupling_stress_test,
)

HIDDEN_SIZE = 32
STEPS = 128
SEEDS = [94, 95, 96, 97]
PERTURB_SCALES = [0.25, 0.5, 0.75, 1.0]
PERTURB_STEP = 64
COUPLING_STRENGTHS = [0.02, 0.05, 0.1, 0.2]
COUPLING_DIM = 8
COUPLING_INTERVAL = 16
DEVICE = "cuda"
OUT_DIR = Path("data/substrate_lab/v8_baseline_compare_20260505")

SUBSTRATES = {
    "demian_native_v8": (DemianNativeV8Substrate, {}),
    "demian_native_v7.4": (DemianNativeV74Substrate, {}),
}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, (cls, kwargs) in SUBSTRATES.items():
        torch.cuda.empty_cache()
        seed_results = {}
        for seed in SEEDS:
            print(f"{name} seed={seed}")
            pert = perturbation_stress_test(
                substrate_name=name,
                hidden_size=HIDDEN_SIZE,
                steps=STEPS,
                seed=seed,
                perturb_step=PERTURB_STEP,
                perturb_scales=PERTURB_SCALES,
                perturb_mode="rss_negation",
                device=DEVICE,
                substrate_kwargs={},
            )
            coup_pert = coupling_stress_test(
                substrate_name=name,
                hidden_size=HIDDEN_SIZE,
                steps=STEPS,
                seed=seed,
                coupling_dim=COUPLING_DIM,
                coupling_interval=COUPLING_INTERVAL,
                coupling_strengths=COUPLING_STRENGTHS,
                device=DEVICE,
                substrate_kwargs={},
            )
            seed_results[str(seed)] = {
                "perturbation": pert,
                "coupling": coup_pert,
            }
        results[name] = seed_results

    # Aggregate
    summary = {}
    for name in SUBSTRATES:
        seeds_data = results[name]
        baselines = [seeds_data[str(s)]["perturbation"]["baseline"] for s in SEEDS]
        summary[name] = {
            "attractor_types": [b["attractor_type"] for b in baselines],
            "interior_classes": [b["interior_class"] for b in baselines],
            "cov_rank_mean": sum(b["covariance_rank"] for b in baselines) / len(baselines),
            "compression_mean": sum(b["compression_ratio"] for b in baselines) / len(baselines),
            "coherence_mean": sum(b["mean_coherence"] for b in baselines) / len(baselines),
            "flow_dim_mean": sum(b["flow_dimension"] for b in baselines) / len(baselines),
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"config": {"hidden_size": HIDDEN_SIZE, "steps": STEPS, "seeds": SEEDS, "device": DEVICE}, "results": results, "aggregate": summary}, f, indent=2)
    print(f"Done: {OUT_DIR / 'summary.json'}")

if __name__ == "__main__":
    main()
