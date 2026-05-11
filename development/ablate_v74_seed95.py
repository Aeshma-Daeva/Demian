"""Ablate v7.4 organs on seed 95 to find the collapse mechanism."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from development.substrate_lab import (
    DemianNativeV74Substrate,
    SelfLoopRunner,
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
OUT_DIR = Path("data/substrate_lab/v74_seed95_ablation_20260505")

# Each variant zeros out one organ path
VARIANTS = {
    "v74_full": {},
    "no_self_policy": {"self_policy_scale": 0.0},
    "no_quarantine": {"quarantine_scale": 0.0, "quarantine_retention": 0.0},
    "no_dynamic_step": {
        "dynamic_step_min": 1.0,
        "dynamic_step_max": 1.0,
        "dynamic_step_hold_slowdown": 0.0,
        "dynamic_step_refusal_slowdown": 0.0,
        "dynamic_step_recovery_accel": 0.0,
        "dynamic_step_integrate_accel": 0.0,
    },
    "no_dynamic_topology": {"dynamic_topology_coupling_scale": 0.0},
    "no_self_potential": {"self_potential_retention": 0.0},
    "no_ancestry_drive": {"ancestry_write_scale": 0.0},
    "no_recovery_drive": {"recovery_drive_scale": 0.0},
    "no_refusal_boundary": {"refusal_boundary_scale": 0.0},
    "no_topology_organs": {"self_topology_scale": 0.0, "hold_topology_scale": 0.0},
    "pressure_policy_full": {"pressure_policy_blend": 1.0},
    # Cumulative: strip back toward v3
    "strip_to_v72": {
        "self_potential_retention": 0.0,
        "quarantine_retention": 0.0,
        "self_policy_scale": 0.0,
        "quarantine_scale": 0.0,
        "dynamic_topology_coupling_scale": 0.0,
        "self_topology_scale": 0.0,
        "hold_topology_scale": 0.0,
        "dynamic_step_min": 1.0,
        "dynamic_step_max": 1.0,
        "dynamic_step_hold_slowdown": 0.0,
        "dynamic_step_refusal_slowdown": 0.0,
        "dynamic_step_recovery_accel": 0.0,
        "dynamic_step_integrate_accel": 0.0,
        "ancestry_write_scale": 0.0,
        "recovery_drive_scale": 0.0,
        "refusal_boundary_scale": 0.0,
    },
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for label, kwargs in VARIANTS.items():
        print(f"\n=== {label} ===")
        torch.cuda.empty_cache()

        pert = perturbation_stress_test(
            substrate_name="demian_native_v7.4",
            hidden_size=HIDDEN_SIZE,
            steps=STEPS,
            seed=SEED,
            perturb_step=PERTURB_STEP,
            perturb_scales=PERTURB_SCALES,
            perturb_mode="rss_negation",
            device=DEVICE,
            substrate_kwargs=kwargs,
        )
        coup = coupling_stress_test(
            substrate_name="demian_native_v7.4",
            hidden_size=HIDDEN_SIZE,
            steps=STEPS,
            seed=SEED,
            coupling_dim=COUPLING_DIM,
            coupling_interval=COUPLING_INTERVAL,
            coupling_strengths=COUPLING_STRENGTHS,
            device=DEVICE,
            substrate_kwargs=kwargs,
        )

        bl = pert["baseline"]
        all_results[label] = {
            "baseline": {
                "attractor_type": bl["attractor_type"],
                "interior_class": bl["interior_class"],
                "cov_rank": bl["covariance_rank"],
                "compression": bl["compression_ratio"],
                "coherence": bl["mean_coherence"],
            },
            "perturb_recovery": {
                str(c["perturb_scale"]): c["final_cosine_vs_baseline"]
                for c in pert["cases"]
            },
            "coupling_final": {
                str(c["coupling_strength"]): c["final_cosine"]
                for c in coup["cases"]
            },
        }
        print(f"    perturb rec: {all_results[label]['perturb_recovery']}")
        print(".", flush=True)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "config": {"hidden_size": HIDDEN_SIZE, "steps": STEPS, "seed": SEED, "device": DEVICE},
            "variants": VARIANTS,
            "results": all_results,
        }, f, indent=2)
    print(f"\nDone: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
