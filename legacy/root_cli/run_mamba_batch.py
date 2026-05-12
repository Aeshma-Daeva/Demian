"""Mamba reservoir batch: run N times and compare to transformer baseline.

Outputs a side-by-side structural comparison:
    - Transformer reservoir (loaded from existing batch_summary.json if available)
    - Mamba + SSM cache (recurrent mode)
    - Mamba - SSM cache (stateless mode)

The key comparison: does period-2 oscillation (ac2=0.963 in transformer)
require attention structure, or does any self-referential loop produce it?
"""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from legacy.demian_runtime.mamba_reservoir import run_mamba_reservoir

log = logging.getLogger(__name__)


def _summarize(traj):
    energy  = np.array([s["energy"]            for s in traj])
    coh     = np.array([s["temporal_coherence"] for s in traj])
    vel     = np.array([s["velocity_align"]     for s in traj])
    cent    = np.array([s["spectral_centroid"]  for s in traj])
    lwr     = np.array([s["layer_work_ratio"]   for s in traj])

    autocorr_lag2 = (np.corrcoef(energy[2:], energy[:-2])[0, 1]
                     if len(energy) > 4 else 0.0)

    state_spread = n_switches = 0.0
    if len(energy) > 10:
        steady = energy[int(len(energy) * 0.1):]
        median_e = np.median(steady)
        hi = steady[steady >= median_e]
        lo = steady[steady <  median_e]
        if len(hi) > 0 and len(lo) > 0:
            state_spread = float(np.mean(hi) - np.mean(lo))
            labels = (steady >= median_e).astype(int)
            n_switches = int(np.sum(labels[1:] != labels[:-1]))

    return dict(
        steps=len(traj),
        energy_mean=float(np.mean(energy)),
        energy_std=float(np.std(energy)),
        energy_min=float(np.min(energy)),
        energy_max=float(np.max(energy)),
        energy_range_90=float(np.percentile(energy, 90) - np.percentile(energy, 10)),
        coherence_mean=float(np.mean(coh)),
        coherence_std=float(np.std(coh)),
        velocity_align_mean=float(np.mean(vel)),
        velocity_align_std=float(np.std(vel)),
        spectral_centroid_mean=float(np.mean(cent)),
        spectral_centroid_std=float(np.std(cent)),
        layer_work_ratio_mean=float(np.mean(lwr)),
        autocorr_lag2=float(autocorr_lag2),
        period2_state_spread=float(state_spread),
        period2_switches=int(n_switches),
    )


def _print_run(label, s):
    print("  {:30s}  E={:.4f}±{:.4f}  coh={:.4f}  vel={:.4f}  cent={:.4f}  ac2={:.4f}  sw={}".format(
        label,
        s["energy_mean"], s["energy_std"],
        s["coherence_mean"],
        s["velocity_align_mean"],
        s["spectral_centroid_mean"],
        s["autocorr_lag2"],
        s["period2_switches"],
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--runs",  type=int, default=3,
                        help="Runs per mode (cache + no-cache). 3 confirms determinism.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--transformer-baseline", type=str,
                        default="data/reservoir_batch/batch_summary.json",
                        help="Path to existing transformer batch_summary.json for comparison")
    parser.add_argument("--seed-residual", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    model_id = args.model or cfg.get("mamba_model_id", "state-spaces/mamba-2.8b-hf")

    log.info("Loading: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # bfloat16: avoids SSM exp(Δ) overflow that occurs with float16
    # when injecting residuals outside the embedding's learned distribution.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    d_model = model.config.hidden_size
    device = str(next(model.parameters()).device)

    seed_residual = None
    if args.seed_residual and Path(args.seed_residual).exists():
        seed_residual = torch.load(args.seed_residual, map_location="cpu", weights_only=True)
        if seed_residual.dim() > 1:
            seed_residual = seed_residual.view(-1)[:d_model]

    data_dir_base = "data/mamba_batch"
    os.makedirs(data_dir_base, exist_ok=True)

    results = {"cache": {}, "no_cache": {}}

    for mode, persist in [("cache", True), ("no_cache", False)]:
        print()
        print("  " + "=" * 72)
        print("  Mode: " + ("SSM cache ON (recurrent)" if persist else "SSM cache OFF (stateless)"))
        print("  " + "=" * 72)

        for run_i in range(1, args.runs + 1):
            run_dir = os.path.join(data_dir_base, mode, "run_" + str(run_i).zfill(2))
            os.makedirs(run_dir, exist_ok=True)

            # Clear checkpoint so each run starts fresh
            ckpt = os.path.join(run_dir, "mamba_checkpoint.pt")
            if os.path.exists(ckpt):
                os.remove(ckpt)

            print()
            print("  Run {}/{} ({})".format(run_i, args.runs, mode))

            traj = run_mamba_reservoir(
                model=model,
                d_model=d_model,
                device=device,
                max_steps=args.steps,
                dump_interval=max(1, args.steps // 10),
                data_dir=run_dir,
                seed_residual=seed_residual,
                persist_ssm_state=persist,
            )

            s = _summarize(traj)
            s["run"] = run_i
            results[mode]["run_" + str(run_i).zfill(2)] = s
            _print_run("  run_{} ({})".format(run_i, mode), s)

            torch.cuda.empty_cache()

    # Save results
    out_path = os.path.join(data_dir_base, "mamba_batch_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    print()
    print("  " + "=" * 80)
    print("  STRUCTURAL COMPARISON: transformer vs mamba")
    print("  " + "=" * 80)
    print()
    print("  {:30s}  {:>8}  {:>7}  {:>7}  {:>7}  {:>7}  {:>5}".format(
        "system", "E_mean", "coh", "vel", "cent", "ac2", "sw"
    ))
    print("  " + "-" * 80)

    # Transformer baseline
    tf_path = Path(args.transformer_baseline)
    if tf_path.exists():
        tf_data = json.loads(tf_path.read_text())
        tf_runs = list(tf_data.values())
        tf = {k: np.mean([r[k] for r in tf_runs])
              for k in ["energy_mean", "coherence_mean", "velocity_align_mean",
                        "spectral_centroid_mean", "autocorr_lag2", "period2_switches"]}
        print("  {:30s}  {:>8.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>5.0f}".format(
            "transformer (Qwen2.5-3B)",
            tf["energy_mean"], tf["coherence_mean"], tf["velocity_align_mean"],
            tf["spectral_centroid_mean"], tf["autocorr_lag2"], tf["period2_switches"],
        ))
    else:
        print("  transformer baseline not found at: " + str(tf_path))

    for mode, label in [("cache", "mamba cache=ON  (recurrent)"),
                        ("no_cache", "mamba cache=OFF (stateless)")]:
        mode_runs = list(results[mode].values())
        if not mode_runs:
            continue
        keys = ["energy_mean", "coherence_mean", "velocity_align_mean",
                "spectral_centroid_mean", "autocorr_lag2", "period2_switches"]
        agg = {k: np.mean([r[k] for r in mode_runs]) for k in keys}
        print("  {:30s}  {:>8.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>5.0f}".format(
            label,
            agg["energy_mean"], agg["coherence_mean"], agg["velocity_align_mean"],
            agg["spectral_centroid_mean"], agg["autocorr_lag2"], agg["period2_switches"],
        ))

    print()
    print("  ac2 > 0.9  → period-2 limit cycle confirmed")
    print("  ac2 < 0.3  → no period-2 (fixed point or aperiodic)")
    print("  vel ≈ -1.0 → every step reverses direction (period-2 signature)")
    print()
    print("  Full results: " + out_path)


if __name__ == "__main__":
    main()
