"""Multi-channel ablation: test V8 without slow AND long_carrier simultaneously."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from development.substrate_lab import (
    DemianNativeV8Substrate,
    SelfLoopRunner,
)
import json

HIDDEN_SIZE = 32
STEPS = 128
DEVICE = "cuda"
PERTURB_STEP = 64
PERTURB_SCALES = [0.25, 0.5, 0.75, 1.0]
SEEDS = [94, 95, 96, 97]

# mapping: channel_name -> key in state_components dict
ABLATIONS = {
    "no_slow+carrier": ["slow", "carrier"],
    "no_slow+support": ["slow", "short_support"],
    "no_slow": ["slow"],
    "no_carrier": ["carrier"],
    "no_slow+carrier+support": ["slow", "carrier", "short_support"],
    "no_control_short": ["control_short"],
    "full": [],
}


def zero_ablation(orig_step, model, zero_names):
    """Wrap step to zero specified channels."""
    def wrapped(state):
        comps = model.state_components(state)
        z = {
            k: (torch.zeros_like(v) if k in zero_names else v)
            for k, v in comps.items()
        }
        sz = (
            z["fast"], z["slow"], z["carrier"], z["short_support"],
            z["packet"], z["control_short"], z["control_long"], z["tightness"]
        )
        return orig_step(sz)
    return wrapped


results = {}
for ablation_name, zero_channels in ABLATIONS.items():
    print(f"\n=== {ablation_name} ===", flush=True)
    seed_results = {}
    for seed in SEEDS:
        print(f"  seed={seed}", end="", flush=True)
        model = DemianNativeV8Substrate(hidden_size=HIDDEN_SIZE)
        orig_step = model.step
        model.step = zero_ablation(orig_step, model, zero_channels)

        runner = SelfLoopRunner(model, device=DEVICE)
        torch.manual_seed(seed)
        base_state = model.initial_state(1, runner.device)

        _, baseline_summary, baseline_final = runner.run(
            steps=STEPS, seed=seed, initial_state=base_state,
        )

        cases = []
        for scale in PERTURB_SCALES:
            torch.manual_seed(seed)
            _, summary, final_state = runner.run(
                steps=STEPS, seed=seed, initial_state=base_state,
                perturb_step=PERTURB_STEP, perturb_scale=scale, perturb_mode="rss_negation",
            )
            cos = float(torch.nn.functional.cosine_similarity(baseline_final, final_state, dim=0).item())
            cases.append({"scale": scale, "final_cosine": cos})

        seed_results[str(seed)] = {
            "baseline": {
                "attractor_type": baseline_summary.attractor_type,
                "interior_class": baseline_summary.interior_class,
                "cov_rank": baseline_summary.covariance_rank,
                "compression": baseline_summary.compression_ratio,
                "final_norm": baseline_summary.final_norm,
                "mean_delta": baseline_summary.mean_delta,
            },
            "perturb_recovery": {str(c["scale"]): c["final_cosine"] for c in cases},
        }
        model.step = orig_step
        torch.cuda.empty_cache()
        print(".", flush=True)
    results[ablation_name] = seed_results

# aggregate
agg = {}
for ablation_name in ABLATIONS:
    rec = {}
    for s in SEEDS:
        for scale in PERTURB_SCALES:
            rec.setdefault(str(scale), []).append(
                results[ablation_name][str(s)]["perturb_recovery"][str(scale)]
            )
    agg[ablation_name] = {
        scale: sum(vals) / len(vals) for scale, vals in rec.items()
    }
    # also mean across all scales
    all_rec = []
    for scale in PERTURB_SCALES:
        all_rec.extend(rec[str(scale)])
    agg[ablation_name]["mean_all"] = sum(all_rec) / len(all_rec)

    # over seeds, rec at 1.0
    agg[ablation_name]["mean_rec_1_0"] = sum(
        results[ablation_name][str(s)]["perturb_recovery"]["1.0"] for s in SEEDS
    ) / len(SEEDS)

print("\n\n=== AGGREGATE ===")
for name in ABLATIONS:
    a = agg[name]
    print(f"{name:30s}: rec_0.25={a['0.25']:.4f} rec_0.5={a['0.5']:.4f} rec_0.75={a['0.75']:.4f} rec_1.0={a['1.0']:.4f} mean={a['mean_all']:.4f}")

# save
out = {
    "config": {
        "hidden_size": HIDDEN_SIZE,
        "steps": STEPS,
        "seeds": SEEDS,
        "device": DEVICE,
        "ablation_configs": list(ABLATIONS.keys()),
    },
    "results": results,
    "aggregate": agg,
}
out_path = Path("data/substrate_lab/v8_multi_ablation_20260507/summary.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(f"\nSaved to {out_path}")
