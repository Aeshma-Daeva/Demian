"""Compare four injection mechanisms + baseline.

Each mechanism places the self-signal at a different computational locus.
The question: does each produce a qualitatively different trajectory?

Mechanisms:
- baseline: No injection (pure generation, control)
- kv: Original KV cache injection (self as context)
- projection: Modulate K/V projection weights (self as query transform)
- activation: Inject into hidden state between layers (self bypasses attention)
- weights: Perturb all attention weights + LayerNorm (self as regime shift)

If KV and activation divergence → self-perception lives in attention.
If weight perturbation differs from KV → it's regime, not content.
If all four converge → signal too weak.
If all four diverge wildly → just noise at different places.
"""
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from demian.vibration import VibrationTracker
from demian.nous import NousInjector
from demian.loop import generate_with_proprioception

log = logging.getLogger(__name__)

MECHANISMS = ["baseline", "kv", "projection", "activation", "weights"]


def run_comparison(model, tokenizer, tracker, injector, mechanism, device,
                   prompt, max_tokens, damping, seed):
    """Single generation with the given mechanism. Returns (text, snapshots)."""
    torch.manual_seed(seed)

    # Reset state
    tracker.reset()
    injector._injected_count = 0
    if hasattr(injector, '_damped_residual'):
        injector._damped_residual = None
    if hasattr(injector, '_residual_memory'):
        injector._residual_memory.clear()

    return generate_with_proprioception(
        model=model,
        tokenizer=tokenizer,
        tracker=tracker,
        injector=injector,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=0.7,
        proprio_inject=(mechanism != "baseline"),
        device=device,
        stream=False,
        damping=damping,
        inject_mode="continuous",
        random_inject=False,
        mechanism=mechanism,
    )


def trajectory_metrics(snapshots):
    """Extract aggregate metrics from a trajectory."""
    if not snapshots:
        return {}

    norms = [s.residual_norm for s in snapshots]
    coherences = [s.temporal_coherence for s in snapshots]
    modes = [s.attention.mode for s in snapshots]
    mode_flips = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1])

    # Per-layer norm profile (average across trajectory)
    layer_norm_profiles = []
    if snapshots[0].layer_norms:
        n_layers = len(snapshots[0].layer_norms)
        for li in range(n_layers):
            profile = [s.layer_norms[li] for s in snapshots if s.layer_norms]
            layer_norm_profiles.append(sum(profile) / len(profile))

    # Injection attention profile (mean across trajectory per layer)
    inj_attn_profile = []
    if snapshots[0].layer_attn:
        n_layers = len(snapshots[0].layer_attn)
        for li in range(n_layers):
            attn_vals = [s.layer_attn[li].mean_injection_attn for s in snapshots if s.layer_attn]
            inj_attn_profile.append(sum(attn_vals) / len(attn_vals))

    return {
        "n_steps": len(snapshots),
        "energy_mean": sum(norms) / len(norms),
        "energy_std": torch.tensor(norms, dtype=torch.float32).std().item(),
        "energy_start": norms[0],
        "energy_end": norms[-1],
        "energy_delta": norms[-1] - norms[0],
        "coherence_mean": sum(coherences) / len(coherences),
        "coherence_std": torch.tensor(coherences, dtype=torch.float32).std().item(),
        "mode_flips": mode_flips,
        "final_mode": modes[-1],
        "layer_norm_profile_early": sum(layer_norm_profiles[:9]) / max(len(layer_norm_profiles[:9]), 1) if layer_norm_profiles else 0,
        "layer_norm_profile_late": sum(layer_norm_profiles[-9:]) / max(len(layer_norm_profiles[-9:]), 1) if layer_norm_profiles else 0,
        "mean_injection_attn_total": sum(inj_attn_profile) / max(len(inj_attn_profile), 1) if inj_attn_profile else 0,
    }


def trajectory_cosine(traj_a, traj_b):
    """Step-by-step cosine similarity between two trajectories' residuals."""
    n = min(len(traj_a), len(traj_b))
    cosines = []
    for i in range(n):
        a = torch.tensor(traj_a[i].raw_residual, dtype=torch.float32)
        b = torch.tensor(traj_b[i].raw_residual, dtype=torch.float32)
        cosines.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    return sum(cosines) / len(cosines) if cosines else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The nature of self-awareness is")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    model_id = cfg.get("proprioceptor_model_id", "Qwen/Qwen2.5-3B-Instruct")
    damping = cfg.get("injection_damping", 0.7)

    log.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    d_model = model.config.hidden_size
    device = str(model.device)

    all_results = {}
    baseline_snapshots = None

    print("=" * 72)
    print("  Injection Mechanism Comparison — Where Does Self-Perception Live?")
    print(f"  Model:  {model_id}")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Tokens: {args.max_tokens}  Runs/mech: {args.runs}")
    print("=" * 72)

    for mech in MECHANISMS:
        print(f"\n  [{mech}]")
        run_data = []

        for run_i in range(args.runs):
            tracker = VibrationTracker(d_model=d_model, target_dim=128, max_trajectory=args.max_tokens)
            injector = NousInjector(
                model=model, tracker=tracker, max_memory_length=16,
                injection_scale=0.01, blend_mode="append",
            )

            text, snapshots = run_comparison(
                model, tokenizer, tracker, injector, mech, device,
                args.prompt, args.max_tokens, damping,
                seed=args.seed + run_i,
            )

            m = trajectory_metrics(snapshots)
            m["run"] = run_i + 1
            m["generated_text_first_50"] = text[:50] if text else ""

            if mech == "baseline" and run_i == 0:
                baseline_snapshots = snapshots
                m["baseline_cosine"] = 1.0
            elif baseline_snapshots is not None:
                m["baseline_cosine"] = trajectory_cosine(snapshots, baseline_snapshots)

            # Intra-mechanism consistency: compare to first run of same mech
            if run_i == 0:
                mech_first_snapshots = snapshots
                m["intra_mechanism_cosine"] = 1.0
            else:
                m["intra_mechanism_cosine"] = trajectory_cosine(snapshots, mech_first_snapshots)

            run_data.append(m)

            print(f"    run {run_i+1}: {len(snapshots)} steps  "
                  f"E={m['energy_mean']:.2f}  "
                  f"coh={m['coherence_mean']:.3f}  "
                  f"flips={m['mode_flips']}  "
                  f"cos(baseline)={m.get('baseline_cosine', 0):.3f}  "
                  f"cos(mech)={m.get('intra_mechanism_cosine', 0):.3f}")

        all_results[mech] = run_data

    # Save results
    os.makedirs("data", exist_ok=True)

    # CSV
    csv_path = "data/mechanism_comparison.csv"
    fieldnames = ["mechanism", "run", "n_steps", "energy_mean", "energy_std",
                  "energy_start", "energy_end", "energy_delta", "coherence_mean",
                  "coherence_std", "mode_flips", "final_mode",
                  "layer_norm_profile_early", "layer_norm_profile_late",
                  "mean_injection_attn_total", "baseline_cosine",
                  "intra_mechanism_cosine", "generated_text_first_50"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mech in MECHANISMS:
            for m in all_results[mech]:
                writer.writerow(m)

    # Trajectory JSONs per mechanism (averaged across runs)
    for mech in MECHANISMS:
        traj_entries = []
        for run_i, m in enumerate(all_results[mech]):
            traj_entries.append({"run": run_i + 1, "metrics": {
                k: v for k, v in m.items() if k not in (
                    "mechanism", "run", "generated_text_first_50",
                    "baseline_cosine", "intra_mechanism_cosine",
                )
            }})

        traj_path = f"data/trajectory_{mech}_comparison.json"
        with open(traj_path, "w") as f:
            json.dump(traj_entries, f, indent=2)

    print(f"\n  Results: {csv_path}")

    # Summary table
    print(f"\n  {'Mechanism':>12} {'Energy':>7} {'Std':>6} {'Coh':>6} "
          f"{'Flips':>5} {'Cos(B)':>7} {'Cos(M)':>7} {'EarlyL':>7}")
    print(f"  {'─'*12} {'─'*7} {'─'*6} {'─'*6} "
          f"{'─'*5} {'─'*7} {'─'*7} {'─'*7}")

    for mech in MECHANISMS:
        runs = all_results[mech]
        def avg(k):
            return sum(m[k] for m in runs) / len(runs)

        print(f"  {mech:>12} {avg('energy_mean'):>7.2f} {avg('energy_std'):>6.2f} "
              f"{avg('coherence_mean'):>6.3f} {avg('mode_flips'):>5.1f} "
              f"{avg('baseline_cosine'):>7.3f} "
              f"{avg('intra_mechanism_cosine'):>7.3f} "
              f"{avg('layer_norm_profile_early'):>7.1f}")

    # Adversarial check: if all baseline cosine >= 0.99, the mechanisms don't diverge
    print()
    for mech in MECHANISMS:
        cos_b = all_results[mech][0].get("baseline_cosine", 0)
        if cos_b > 0.99 and mech != "baseline":
            print(f"  WARNING: {mech} cosine to baseline = {cos_b:.4f} — "
                  f"no meaningful divergence from baseline. Signal may be too weak or "
                  f"the mechanism barely enters the computation.")

    print()


if __name__ == "__main__":
    main()
