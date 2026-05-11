"""Coupled-decoupled probe: does coupling leave a persistent trace without transmission organs?

Two v8 instances: independent 64 steps → couple 64 steps → decouple 64 steps.
Measure cosine before/during/after coupling.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from development.substrate_lab import DemianNativeV8Substrate, SelfLoopRunner, _fixed_projection

HIDDEN_SIZE = 32
INDEPENDENT_STEPS = 64
COUPLE_STEPS = 64
DECOUPLE_STEPS = 64
TOTAL_STEPS = INDEPENDENT_STEPS + COUPLE_STEPS + DECOUPLE_STEPS
SEEDS = [(94, 95), (96, 97), (98, 99), (100, 101), (102, 103), (104, 105), (106, 107), (108, 109)]
COUPLING_STRENGTHS = [0.02, 0.05, 0.1, 0.2]
COUPLING_DIM = 8
COUPLING_INTERVAL = 16
DEVICE = "cuda"
OUT_DIR = Path("data/substrate_lab/v8_coupled_decoupled_20260505")


def run_pair(seed_a, seed_b, coupling_strength):
    torch.manual_seed(seed_a)
    model_a = DemianNativeV8Substrate(hidden_size=HIDDEN_SIZE)
    torch.manual_seed(seed_b)
    model_b = DemianNativeV8Substrate(hidden_size=HIDDEN_SIZE)
    proj = _fixed_projection(HIDDEN_SIZE, COUPLING_DIM, seed_a * 1000 + seed_b)

    runner_a = SelfLoopRunner(model_a, device=DEVICE)
    runner_b = SelfLoopRunner(model_b, device=DEVICE)

    state_a = model_a.initial_state(1, runner_a.device)
    state_b = model_b.initial_state(1, runner_b.device)

    cosines = []
    phases = []

    for step_idx in range(1, TOTAL_STEPS + 1):
        state_a = model_a.step(state_a)
        state_b = model_b.step(state_b)

        h_a = model_a.state_vector(state_a).view(-1).detach().float().cpu()
        h_b = model_b.state_vector(state_b).view(-1).detach().float().cpu()
        cos = float(torch.nn.functional.cosine_similarity(h_a, h_b, dim=0).item())

        phase = "independent" if step_idx <= INDEPENDENT_STEPS else (
            "coupled" if step_idx <= INDEPENDENT_STEPS + COUPLE_STEPS else "decoupled"
        )
        cosines.append(cos)
        phases.append(phase)

        # Coupling phase: inject at interval
        if phase == "coupled" and step_idx % COUPLING_INTERVAL == 0:
            h_a_dev = h_a.to(runner_a.device)
            h_b_dev = h_b.to(runner_b.device)
            state_a = model_a.inject_coupling_message(state_a, h_b_dev, coupling_strength)
            state_b = model_b.inject_coupling_message(state_b, h_a_dev, coupling_strength)

    # Metrics per phase
    independent_cos = cosines[:INDEPENDENT_STEPS]
    coupled_cos = cosines[INDEPENDENT_STEPS:INDEPENDENT_STEPS + COUPLE_STEPS]
    decoupled_cos = cosines[INDEPENDENT_STEPS + COUPLE_STEPS:]

    return {
        "independent_mean_cos": sum(independent_cos) / len(independent_cos),
        "independent_final_cos": independent_cos[-1],
        "coupled_mean_cos": sum(coupled_cos) / len(coupled_cos),
        "coupled_final_cos": coupled_cos[-1],
        "decoupled_mean_cos": sum(decoupled_cos) / len(decoupled_cos),
        "decoupled_final_cos": decoupled_cos[-1],
        "coupling_shift": coupled_cos[-1] - independent_cos[-1],
        "decoupling_persistence": decoupled_cos[-1] - independent_cos[-1],  # trace left after decoupling
        "decoupled_mean_vs_independent_mean": sum(decoupled_cos) / len(decoupled_cos) - sum(independent_cos) / len(independent_cos),
        "phases": phases,
        "cosines": cosines,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for strength in COUPLING_STRENGTHS:
        print(f"\n=== coupling_strength={strength} ===", flush=True)
        pair_results = {}
        for seed_a, seed_b in SEEDS:
            print(f"  pair ({seed_a},{seed_b})", end="", flush=True)
            r = run_pair(seed_a, seed_b, strength)
            pair_results[f"{seed_a}_{seed_b}"] = r
            torch.cuda.empty_cache()
            print(".", flush=True)
        all_results[str(strength)] = pair_results

    # Aggregate
    agg = {}
    for strength_key in all_results:
        pairs = all_results[strength_key]
        persistence_vals = [p["decoupling_persistence"] for p in pairs.values()]
        shift_vals = [p["coupling_shift"] for p in pairs.values()]
        ind_final = [p["independent_final_cos"] for p in pairs.values()]
        coup_final = [p["coupled_final_cos"] for p in pairs.values()]
        dec_final = [p["decoupled_final_cos"] for p in pairs.values()]
        n = len(persistence_vals)
        agg[strength_key] = {
            "coupling_shift_mean": sum(shift_vals) / n,
            "coupling_shift_std": (sum((v - sum(shift_vals)/n)**2 for v in shift_vals) / n) ** 0.5,
            "decoupling_persistence_mean": sum(persistence_vals) / n,
            "decoupling_persistence_std": (sum((v - sum(persistence_vals)/n)**2 for v in persistence_vals) / n) ** 0.5,
            "independent_final_cos_mean": sum(ind_final) / n,
            "coupled_final_cos_mean": sum(coup_final) / n,
            "decoupled_final_cos_mean": sum(dec_final) / n,
            "n_persistent": sum(1 for v in persistence_vals if abs(v) > 0.01),
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "config": {
                "hidden_size": HIDDEN_SIZE,
                "independent_steps": INDEPENDENT_STEPS,
                "couple_steps": COUPLE_STEPS,
                "decouple_steps": DECOUPLE_STEPS,
                "seeds": SEEDS,
                "coupling_strengths": COUPLING_STRENGTHS,
                "coupling_dim": COUPLING_DIM,
                "coupling_interval": COUPLING_INTERVAL,
                "device": DEVICE,
            },
            "aggregate": agg,
            "results": all_results,
        }, f, indent=2)
    print(f"\nDone: {OUT_DIR / 'summary.json'}")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
