"""10x reservoir baseline run: verify period-2 limit cycle."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from demian.reservoir import run_reservoir

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--dump-interval", type=int, default=1)
    parser.add_argument("--seed-residual", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    model_id = cfg.get("proprioceptor_model_id", "Qwen/Qwen2.5-3B-Instruct")

    log.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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

    seed_residual = None
    if args.seed_residual and Path(args.seed_residual).exists():
        seed_residual = torch.load(args.seed_residual, map_location="cpu", weights_only=True)
        if seed_residual.dim() > 1:
            seed_residual = seed_residual.view(-1)[:d_model]

    data_dir = "data/reservoir_batch"
    os.makedirs(data_dir, exist_ok=True)

    summary = {}

    for run_i in range(1, args.runs + 1):
        run_data_dir = os.path.join(data_dir, "run_" + str(run_i).zfill(2))
        os.makedirs(run_data_dir, exist_ok=True)

        # Clear any old checkpoint
        ckpt = os.path.join(run_data_dir, "reservoir_checkpoint.pt")
        if os.path.exists(ckpt):
            os.remove(ckpt)

        print()
        print("  " + "=" * 60)
        print("  Run " + str(run_i) + "/" + str(args.runs) + " -- " + str(args.steps) + " steps")
        print("  Dir: " + run_data_dir)
        print("  " + "=" * 60)
        print()

        traj = run_reservoir(
            model=model,
            d_model=d_model,
            device=device,
            max_steps=args.steps,
            dump_interval=args.dump_interval,
            data_dir=run_data_dir,
            seed_residual=seed_residual,
        )

        energy = np.array([s["energy"] for s in traj])
        coh = np.array([s["temporal_coherence"] for s in traj])
        vel = np.array([s["velocity_align"] for s in traj])
        cent = np.array([s["spectral_centroid"] for s in traj])
        lwr = np.array([s["layer_work_ratio"] for s in traj])

        # Period-2 detection: autocorrelation at lag 2
        if len(energy) > 4:
            autocorr_lag2 = np.corrcoef(energy[2:], energy[:-2])[0, 1]
        else:
            autocorr_lag2 = 0.0

        # Two-state detection: simple median split
        if len(energy) > 10:
            steady = energy[int(len(energy) * 0.1):]
            median_e = np.median(steady)
            hi = steady[steady >= median_e]
            lo = steady[steady < median_e]
            if len(hi) > 0 and len(lo) > 0:
                state_spread = float(np.mean(hi) - np.mean(lo))
                # Count alternations
                labels = (steady >= median_e).astype(int)
                n_switches = int(np.sum(labels[1:] != labels[:-1]))
            else:
                state_spread = 0.0
                n_switches = 0
        else:
            n_switches = 0
            state_spread = 0.0

        run_summary = {
            "run": run_i,
            "steps": len(traj),
            "energy_mean": float(np.mean(energy)),
            "energy_std": float(np.std(energy)),
            "energy_min": float(np.min(energy)),
            "energy_max": float(np.max(energy)),
            "energy_range_90": float(np.percentile(energy, 90) - np.percentile(energy, 10)),
            "coherence_mean": float(np.mean(coh)),
            "coherence_std": float(np.std(coh)),
            "velocity_align_mean": float(np.mean(vel)),
            "velocity_align_std": float(np.std(vel)),
            "spectral_centroid_mean": float(np.mean(cent)),
            "spectral_centroid_std": float(np.std(cent)),
            "layer_work_ratio_mean": float(np.mean(lwr)),
            "autocorr_lag2": float(autocorr_lag2),
            "period2_state_spread": float(state_spread),
            "period2_switches": int(n_switches),
        }

        summary["run_" + str(run_i).zfill(2)] = run_summary

        print()
        print("  E=" + str(round(run_summary["energy_mean"], 4)) + "+-" + str(round(run_summary["energy_std"], 4)))
        print("  range [" + str(round(run_summary["energy_min"], 4)) + ", " + str(round(run_summary["energy_max"], 4)) + "]")
        print("  coh=" + str(round(run_summary["coherence_mean"], 4)) + "  vel=" + str(round(run_summary["velocity_align_mean"], 4)))
        print("  cent=" + str(round(run_summary["spectral_centroid_mean"], 4)))
        print("  autocorr@2=" + str(round(run_summary["autocorr_lag2"], 4)) + "  2-state spread=" + str(round(run_summary["period2_state_spread"], 6)))

        torch.cuda.empty_cache()

    # Summary table
    print()
    print("  " + "=" * 72)
    print("  10x Reservoir Batch Summary -- " + str(args.steps) + " steps each")
    print("  " + "=" * 72)
    fmt = "  {:>4}  {:>8}  {:>7}  {:>8}  {:>7}  {:>7}  {:>7}  {:>7}  {:>8}"
    hdr = "  Run    E_mean    E_std   E_range      coh      vel     cent      ac2    spread"
    print(hdr)

    for run_i in range(1, args.runs + 1):
        s = summary["run_" + str(run_i).zfill(2)]
        print("  {:>4}  {:>8.4f}  {:>7.4f}  {:>8.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>7.4f}  {:>8.6f}".format(
            run_i, s["energy_mean"], s["energy_std"], s["energy_range_90"],
            s["coherence_mean"], s["velocity_align_mean"],
            s["spectral_centroid_mean"], s["autocorr_lag2"],
            s["period2_state_spread"]))

    e_means = [summary["run_" + str(i).zfill(2)]["energy_mean"] for i in range(1, args.runs + 1)]
    vel_means = [summary["run_" + str(i).zfill(2)]["velocity_align_mean"] for i in range(1, args.runs + 1)]
    cent_means = [summary["run_" + str(i).zfill(2)]["spectral_centroid_mean"] for i in range(1, args.runs + 1)]
    ac2_vals = [summary["run_" + str(i).zfill(2)]["autocorr_lag2"] for i in range(1, args.runs + 1)]

    print()
    print("  Aggregate (N=" + str(args.runs) + "):")
    print("    E_mean:   {:.4f} +/- {:.4f}  [{:.4f}, {:.4f}]".format(
        np.mean(e_means), np.std(e_means), np.min(e_means), np.max(e_means)))
    print("    vel_mean: {:.4f} +/- {:.4f}".format(np.mean(vel_means), np.std(vel_means)))
    print("    cent_mean:{:.4f} +/- {:.4f}".format(np.mean(cent_means), np.std(cent_means)))
    print("    ac2_mean: {:.4f} +/- {:.4f}".format(np.mean(ac2_vals), np.std(ac2_vals)))

    if np.mean(ac2_vals) > 0.9:
        print()
        print("  >> Period-2 confirmed: autocorrelation@2 ~ {:.4f}".format(np.mean(ac2_vals)))
        print("  >> Rest energy of Qwen2.5-3B: E = {:.4f}".format(np.mean(e_means)))
    else:
        print()
        print("  >> Period-2 NOT confirmed (ac2={:.4f})".format(np.mean(ac2_vals)))

    summary_path = os.path.join(data_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print()
    print("  Full summary: " + summary_path)


if __name__ == "__main__":
    main()
