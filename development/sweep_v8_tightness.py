"""Sweep v8 initial_tightness_bias — map phase space."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from development.substrate_lab import (
    DemianNativeV8Substrate,
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
OUT_DIR = Path("data/substrate_lab/v8_tightness_sweep_20260505")

TIGHTNESS_VALUES = [-0.5, -0.35, -0.2, 0.0, 0.2, 0.35, 0.5]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for tb in TIGHTNESS_VALUES:
        kwargs = {"initial_tightness_bias": tb}
        label = f"tightness_{tb}"
        print(f"\n=== tightness_bias={tb} ===")
        seed_results = {}

        for seed in SEEDS:
            print(f"  seed={seed}", end="", flush=True)
            pert = perturbation_stress_test(
                substrate_name="demian_native_v8",
                hidden_size=HIDDEN_SIZE,
                steps=STEPS,
                seed=seed,
                perturb_step=PERTURB_STEP,
                perturb_scales=PERTURB_SCALES,
                perturb_mode="rss_negation",
                device=DEVICE,
                substrate_kwargs=kwargs,
            )
            coup = coupling_stress_test(
                substrate_name="demian_native_v8",
                hidden_size=HIDDEN_SIZE,
                steps=STEPS,
                seed=seed,
                coupling_dim=COUPLING_DIM,
                coupling_interval=COUPLING_INTERVAL,
                coupling_strengths=COUPLING_STRENGTHS,
                device=DEVICE,
                substrate_kwargs=kwargs,
            )
            seed_results[str(seed)] = {"perturbation": pert, "coupling": coup}
            print(".", flush=True)

        all_results[label] = seed_results
        torch.cuda.empty_cache()

    # Aggregate across tightness values
    summary = {}
    for label in all_results:
        seeds_data = all_results[label]
        baselines = [seeds_data[str(s)]["perturbation"]["baseline"] for s in SEEDS]
        perturb_cases = []
        for s in SEEDS:
            for c in seeds_data[str(s)]["perturbation"]["cases"]:
                perturb_cases.append({
                    "seed": s,
                    "scale": c["perturb_scale"],
                    "final_cosine_vs_baseline": c["final_cosine_vs_baseline"],
                })
        coupling_cases = []
        for s in SEEDS:
            for c in seeds_data[str(s)]["coupling"]["cases"]:
                coupling_cases.append({
                    "seed": s,
                    "strength": c["coupling_strength"],
                    "final_cosine": c["final_cosine"],
                    "mean_cosine": c["mean_cosine"],
                })
        summary[label] = {
            "attractor_types": [b["attractor_type"] for b in baselines],
            "interior_classes": list(set(b["interior_class"] for b in baselines)),
            "cov_rank_mean": sum(b["covariance_rank"] for b in baselines) / len(baselines),
            "compression_mean": sum(b["compression_ratio"] for b in baselines) / len(baselines),
            "flow_dim_mean": sum(b["flow_dimension"] for b in baselines) / len(baselines),
            "coherence_mean": sum(b["mean_coherence"] for b in baselines) / len(baselines),
            "perturb_recovery_mean": {
                str(scale): sum(
                    c["final_cosine_vs_baseline"]
                    for c in perturb_cases
                    if c["scale"] == scale
                ) / len(SEEDS)
                for scale in PERTURB_SCALES
            },
            "coupling_final_mean": {
                str(strength): sum(
                    c["final_cosine"]
                    for c in coupling_cases
                    if c["strength"] == strength
                ) / len(SEEDS)
                for strength in COUPLING_STRENGTHS
            },
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "config": {
                "hidden_size": HIDDEN_SIZE,
                "steps": STEPS,
                "seeds": SEEDS,
                "tightness_values": TIGHTNESS_VALUES,
                "device": DEVICE,
            },
            "results": all_results,
            "aggregate": summary,
        }, f, indent=2)
    print(f"\nDone: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
