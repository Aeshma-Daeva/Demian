"""v8 (7-channel) vs v8.5 (5-channel) across step lengths."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from development.substrate_lab import (
    DemianNativeV8Substrate,
    DemianNativeV85Substrate,
    SelfLoopRunner,
)

HIDDEN_SIZE = 32
STEPS_LIST = [128, 256, 512, 1024]
SEEDS = [94, 95, 96, 97]
DEVICE = "cuda"
PERTURB_STEP_RATIO = 0.5  # perturb at half total steps
PERTURB_SCALE = 1.0
COUPLING_STRENGTHS = [0.02, 0.05, 0.1]
COUPLING_DIM = 8
COUPLING_INTERVAL_RATIO = 0.125  # interval = steps/8
OUT_DIR = Path("data/substrate_lab/v8_v85_step_compare_20260505")


def run_single(model_class, kwargs, steps, seed):
    torch.manual_seed(seed)
    model = model_class(hidden_size=HIDDEN_SIZE, **kwargs)
    runner = SelfLoopRunner(model, device=DEVICE)
    base_state = model.initial_state(1, runner.device)

    perturb_step = int(steps * PERTURB_STEP_RATIO)

    # Baseline
    _, baseline_summary, baseline_final = runner.run(steps=steps, seed=seed, initial_state=base_state)

    # Perturbed
    torch.manual_seed(seed)
    _, pert_summary, pert_final = runner.run(
        steps=steps, seed=seed, initial_state=base_state,
        perturb_step=perturb_step, perturb_scale=PERTURB_SCALE, perturb_mode="rss_negation",
    )
    pert_rec = float(torch.nn.functional.cosine_similarity(baseline_final, pert_final, dim=0).item())

    # Coupling with another instance (seed+1)
    torch.manual_seed(seed)
    model_a = model_class(hidden_size=HIDDEN_SIZE, **kwargs)
    torch.manual_seed(seed + 1)
    model_b = model_class(hidden_size=HIDDEN_SIZE, **kwargs)
    runner_a = SelfLoopRunner(model_a, device=DEVICE)
    runner_b = SelfLoopRunner(model_b, device=DEVICE)

    state_a = model_a.initial_state(1, runner_a.device)
    state_b = model_b.initial_state(1, runner_b.device)
    coupling_interval = max(1, int(steps * COUPLING_INTERVAL_RATIO))
    coup_results = {}

    for strength in COUPLING_STRENGTHS:
        sa, sb = state_a, state_b
        initial_cos = None
        final_cos = None
        for step_idx in range(1, steps + 1):
            sa = model_a.step(sa)
            sb = model_b.step(sb)
            if step_idx == 1:
                ha = model_a.state_vector(sa).view(-1).float()
                hb = model_b.state_vector(sb).view(-1).float()
                initial_cos = float(torch.nn.functional.cosine_similarity(ha, hb, dim=0).item())
            if step_idx % coupling_interval == 0:
                ha = model_a.state_vector(sa).view(-1)
                hb = model_b.state_vector(sb).view(-1)
                sa = model_a.inject_coupling_message(sa, hb, strength)
                sb = model_b.inject_coupling_message(sb, ha, strength)
        ha = model_a.state_vector(sa).view(-1).float()
        hb = model_b.state_vector(sb).view(-1).float()
        final_cos = float(torch.nn.functional.cosine_similarity(ha, hb, dim=0).item())
        coup_results[str(strength)] = {"initial_cos": initial_cos, "final_cos": final_cos, "shift": final_cos - (initial_cos or 0)}

    return {
        "attractor": baseline_summary.attractor_type,
        "interior": baseline_summary.interior_class,
        "cov_rank": baseline_summary.covariance_rank,
        "compression": baseline_summary.compression_ratio,
        "coherence": baseline_summary.mean_coherence,
        "flow_dim": baseline_summary.flow_dimension,
        "perturb_recovery": pert_rec,
        "coupling": coup_results,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for label, cls, kwargs in [
        ("v8_7ch", DemianNativeV8Substrate, {}),
        ("v85_5ch", DemianNativeV85Substrate, {}),
    ]:
        print(f"\n{'='*50}")
        print(f"{label}")
        print(f"{'='*50}")
        step_results = {}
        for steps in STEPS_LIST:
            print(f"  steps={steps}", flush=True)
            seed_results = {}
            for seed in SEEDS:
                print(f"    seed={seed}", end="", flush=True)
                r = run_single(cls, kwargs, steps, seed)
                seed_results[str(seed)] = r
                torch.cuda.empty_cache()
                print(".", flush=True)
            # aggregate across seeds
            recs = [seed_results[str(s)]["perturb_recovery"] for s in SEEDS]
            step_results[str(steps)] = {
                "seeds": seed_results,
                "perturb_rec_mean": sum(recs) / len(recs),
                "perturb_rec_min": min(recs),
                "perturb_rec_max": max(recs),
                "attractors": list(set(seed_results[str(s)]["attractor"] for s in SEEDS)),
                "cov_rank_mean": sum(seed_results[str(s)]["cov_rank"] for s in SEEDS) / len(SEEDS),
                "compression_mean": sum(seed_results[str(s)]["compression"] for s in SEEDS) / len(SEEDS),
                "coherence_mean": sum(seed_results[str(s)]["coherence"] for s in SEEDS) / len(SEEDS),
            }
        all_results[label] = step_results

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "config": {
                "hidden_size": HIDDEN_SIZE,
                "steps": STEPS_LIST,
                "seeds": SEEDS,
                "perturb_scale": PERTURB_SCALE,
                "device": DEVICE,
            },
            "results": all_results,
        }, f, indent=2)

    # Summary table
    print("\n\n=== COMPARISON ===")
    for label in all_results:
        print(f"\n{label}:")
        for steps in STEPS_LIST:
            a = all_results[label][str(steps)]
            print(
                f"  steps={steps}: rec={a['perturb_rec_mean']:.4f} "
                f"[{a['perturb_rec_min']:.4f}-{a['perturb_rec_max']:.4f}] "
                f"attr={a['attractors']} cov={a['cov_rank_mean']:.3f} "
                f"comp={a['compression_mean']:.3f} coh={a['coherence_mean']:.4f}"
            )

    print(f"\nDone: {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
