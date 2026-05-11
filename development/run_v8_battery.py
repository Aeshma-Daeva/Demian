"""v8 battery: channel ablation + bottleneck sweep + population diversity — all in one consolidated run."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from development.substrate_lab import (
    DemianNativeV8Substrate,
    SelfLoopRunner,
    perturbation_stress_test,
)

HIDDEN_SIZE = 32
STEPS = 128
DEVICE = "cuda"
PERTURB_STEP = 64
PERTURB_SCALES = [0.25, 0.5, 0.75, 1.0]

SEEDS_ABL = [94, 95, 96, 97]
SEEDS_POP = list(range(94, 110))


# ── Channel ablation ──────────────────────

CHANNELS = ["short_support", "control_short", "control_long", "packet", "long_carrier", "slow"]
# fast and tightness cannot be removed — fast is the surface, tightness is the sole organ


class V8NoChannel(DemianNativeV8Substrate):
    """v8 with one channel zeroed out. _zero_channel set per instance."""
    _zero_channel: str = ""

    def step(self, state):
        # zero the target channel in state
        comps = self.state_components(state)
        zeroed = tuple(
            torch.zeros_like(t) if name == self._zero_channel else t
            for name, t in comps.items()
        )
        state_zeroed = (
            zeroed[0], zeroed[1], zeroed[2], zeroed[3],
            zeroed[4], zeroed[5], zeroed[6], zeroed[7],
        )
        return super().step(state_zeroed)

    def _get_name(self):
        return "demian_native_v8"


# Monkey-patch for distinct registry names
def _make_ablated_substrate(name, channel):
    cls = type(f"V8NoChannel_{channel}", (V8NoChannel,), {"_zero_channel": channel})
    return cls


def run_channel_ablation():
    out = {}
    for ch in CHANNELS:
        print(f"\n=== ablate: {ch} ===", flush=True)
        seed_results = {}
        for seed in SEEDS_ABL:
            print(f"  seed={seed}", end="", flush=True)
            # Build substrate directly
            model = DemianNativeV8Substrate(hidden_size=HIDDEN_SIZE)
            comps0 = model.state_components(model.initial_state(1, torch.device(DEVICE)))
            zero_map = {name: (name == ch) for name in comps0}

            # Use a direct approach: run perturbation stress via runner
            # We need to wrap step to zero the channel
            orig_step = model.step

            def make_step(ch_name):
                def wrapped_step(state):
                    comps = model.state_components(state)
                    z = {
                        k: (torch.zeros_like(v) if k == ch_name else v)
                        for k, v in comps.items()
                    }
                    sz = (z["fast"], z["slow"], z["carrier"], z["short_support"],
                          z["packet"], z["control_short"], z["control_long"], z["tightness"])
                    return orig_step(sz)
                return wrapped_step

            model.step = make_step(ch)

            runner = SelfLoopRunner(model, device=DEVICE)
            torch.manual_seed(seed)
            base_state = model.initial_state(1, runner.device)

            # baseline
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
                cases.append({
                    "scale": scale,
                    "final_cosine_vs_baseline": float(
                        torch.nn.functional.cosine_similarity(baseline_final, final_state, dim=0).item()
                    ),
                })

            seed_results[str(seed)] = {
                "baseline": {
                    "attractor_type": baseline_summary.attractor_type,
                    "cov_rank": baseline_summary.covariance_rank,
                    "compression": baseline_summary.compression_ratio,
                },
                "perturb_recovery": {str(c["scale"]): c["final_cosine_vs_baseline"] for c in cases},
            }
            model.step = orig_step
            torch.cuda.empty_cache()
            print(".", flush=True)

        out[ch] = seed_results

    # aggregate
    agg = {}
    for ch in CHANNELS:
        rec = []
        for s in SEEDS_ABL:
            for scale in PERTURB_SCALES:
                rec.append(out[ch][str(s)]["perturb_recovery"][str(scale)])
        agg[ch] = {
            "perturb_recovery_mean_1.0": sum(
                out[ch][str(s)]["perturb_recovery"]["1.0"] for s in SEEDS_ABL
            ) / len(SEEDS_ABL),
            "perturb_recovery_mean_all": sum(rec) / len(rec),
            "attractors": [out[ch][str(s)]["baseline"]["attractor_type"] for s in SEEDS_ABL],
        }
    return out, agg


# ── Bottleneck dimension sweep ─────────────────────

BOTTLENECK_DIMS = [2, 4, 6, 8]


def run_bottleneck_sweep():
    out = {}
    for bdim in BOTTLENECK_DIMS:
        print(f"\n=== bottleneck_dim={bdim} ===", flush=True)
        kwargs = {
            "carrier_dim": bdim,
            "support_dim": bdim,
            "packet_dim": bdim,
            "control_dim": bdim,
        }
        seed_results = {}
        for seed in SEEDS_ABL:
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
            bl = pert["baseline"]
            seed_results[str(seed)] = {
                "baseline": {
                    "attractor_type": bl["attractor_type"],
                    "cov_rank": bl["covariance_rank"],
                    "compression": bl["compression_ratio"],
                },
                "perturb_recovery": {
                    str(c["perturb_scale"]): c["final_cosine_vs_baseline"]
                    for c in pert["cases"]
                },
            }
            torch.cuda.empty_cache()
            print(".", flush=True)
        out[str(bdim)] = seed_results

    agg = {}
    for bdim in BOTTLENECK_DIMS:
        k = str(bdim)
        rec = []
        for s in SEEDS_ABL:
            for scale in PERTURB_SCALES:
                rec.append(out[k][str(s)]["perturb_recovery"][str(scale)])
        agg[k] = {
            "perturb_recovery_mean_1.0": sum(
                out[k][str(s)]["perturb_recovery"]["1.0"] for s in SEEDS_ABL
            ) / len(SEEDS_ABL),
            "perturb_recovery_mean_all": sum(rec) / len(rec),
            "attractors": [out[k][str(s)]["baseline"]["attractor_type"] for s in SEEDS_ABL],
            "cov_rank_mean": sum(out[k][str(s)]["baseline"]["cov_rank"] for s in SEEDS_ABL) / len(SEEDS_ABL),
        }
    return out, agg


# ── Population diversity ───────────────────


def run_population():
    out = {}
    for seed in SEEDS_POP:
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
        )
        bl = pert["baseline"]
        out[str(seed)] = {
            "attractor_type": bl["attractor_type"],
            "interior_class": bl["interior_class"],
            "cov_rank": bl["covariance_rank"],
            "compression": bl["compression_ratio"],
            "coherence": bl["mean_coherence"],
            "flow_dim": bl["flow_dimension"],
            "final_norm": bl["final_norm"],
            "perturb_recovery_1.0": next(
                c["final_cosine_vs_baseline"] for c in pert["cases"] if c["perturb_scale"] == 1.0
            ),
        }
        torch.cuda.empty_cache()
        print(".", flush=True)

    # diversity metrics
    attrs = [out[str(s)]["attractor_type"] for s in SEEDS_POP]
    cov_ranks = [out[str(s)]["cov_rank"] for s in SEEDS_POP]
    comps = [out[str(s)]["compression"] for s in SEEDS_POP]
    recs = [out[str(s)]["perturb_recovery_1.0"] for s in SEEDS_POP]
    agg = {
        "n_seeds": len(SEEDS_POP),
        "attractor_uniformity": len(set(attrs)) == 1,
        "attractor_types": list(set(attrs)),
        "interior_classes": list(set(out[str(s)]["interior_class"] for s in SEEDS_POP)),
        "cov_rank_mean": sum(cov_ranks) / len(cov_ranks),
        "cov_rank_std": (sum((c - sum(cov_ranks)/len(cov_ranks))**2 for c in cov_ranks) / len(cov_ranks)) ** 0.5,
        "compression_mean": sum(comps) / len(comps),
        "compression_std": (sum((c - sum(comps)/len(comps))**2 for c in comps) / len(comps)) ** 0.5,
        "perturb_recovery_1.0_mean": sum(recs) / len(recs),
        "perturb_recovery_1.0_std": (sum((r - sum(recs)/len(recs))**2 for r in recs) / len(recs)) ** 0.5,
        "perturb_recovery_1.0_min": min(recs),
        "perturb_recovery_1.0_max": max(recs),
    }
    return out, agg


# ── Main ───────────────────

def main():
    base = Path("data/substrate_lab")

    # Channel ablation
    print("=" * 50)
    print("CHANNEL ABLATION")
    print("=" * 50)
    abl_results, abl_agg = run_channel_ablation()
    abl_dir = base / "v8_channel_ablation_20260505"
    abl_dir.mkdir(parents=True, exist_ok=True)
    with open(abl_dir / "summary.json", "w") as f:
        json.dump({
            "config": {"hidden_size": HIDDEN_SIZE, "steps": STEPS, "seeds": SEEDS_ABL, "device": DEVICE},
            "channels_ablated": CHANNELS,
            "aggregate": abl_agg,
            "results": abl_results,
        }, f, indent=2)
    print(f"\nChannel ablation done: {abl_dir}/summary.json")
    print(json.dumps(abl_agg, indent=2))

    # Bottleneck sweep
    print("\n" + "=" * 50)
    print("BOTTLENECK SWEEP")
    print("=" * 50)
    bn_results, bn_agg = run_bottleneck_sweep()
    bn_dir = base / "v8_bottleneck_sweep_20260505"
    bn_dir.mkdir(parents=True, exist_ok=True)
    with open(bn_dir / "summary.json", "w") as f:
        json.dump({
            "config": {"hidden_size": HIDDEN_SIZE, "steps": STEPS, "seeds": SEEDS_ABL, "device": DEVICE},
            "bottleneck_dims": BOTTLENECK_DIMS,
            "aggregate": bn_agg,
            "results": bn_results,
        }, f, indent=2)
    print(f"\nBottleneck sweep done: {bn_dir}/summary.json")
    print(json.dumps(bn_agg, indent=2))

    # Population
    print("\n" + "=" * 50)
    print("POPULATION DIVERSITY (16 seeds)")
    print("=" * 50)
    pop_results, pop_agg = run_population()
    pop_dir = base / "v8_population_20260505"
    pop_dir.mkdir(parents=True, exist_ok=True)
    with open(pop_dir / "summary.json", "w") as f:
        json.dump({
            "config": {"hidden_size": HIDDEN_SIZE, "steps": STEPS, "seeds": SEEDS_POP, "device": DEVICE},
            "aggregate": pop_agg,
            "results": pop_results,
        }, f, indent=2)
    print(f"\nPopulation done: {pop_dir}/summary.json")
    print(json.dumps(pop_agg, indent=2))


if __name__ == "__main__":
    main()
