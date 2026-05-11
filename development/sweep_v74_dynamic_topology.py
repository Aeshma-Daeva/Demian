"""Sweep v7.4 dynamic topology coupling and choose a conservative default."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (
    _make_substrate,
    bottleneck_run,
    coupled_pair,
    memory_pair,
    perturbation_stress_test,
)


def _parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _trace_topology(
    coupling_scale: float,
    hidden_size: int,
    steps: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    torch.manual_seed(seed)
    model = _make_substrate(
        "demian_native_v7.4",
        hidden_size,
        {"dynamic_topology_coupling_scale": coupling_scale},
    )
    runner_device = torch.device(device)
    state = model.initial_state(1, runner_device)
    injection_norms: list[float] = []
    authority: list[float] = []
    shadow_norms: list[float] = []
    base_topology_norms: list[float] = []

    with torch.no_grad():
        for _ in range(steps):
            state = model.step(state)
            aux = model._step_aux
            injection_norms.append(float(aux.get("v74_dynamic_topology_injection_norm", 0.0)))
            authority.append(float(aux.get("v74_dynamic_topology_authority_mean", 0.0)))
            shadow_norms.append(float(aux.get("v74_topology_state_norm", 0.0)))
            base_topology_norms.append(float(aux.get("topology_state_norm", 0.0)))

    return {
        "dynamic_topology_injection_last": injection_norms[-1] if injection_norms else 0.0,
        "dynamic_topology_injection_max": max(injection_norms) if injection_norms else 0.0,
        "dynamic_topology_authority_mean": sum(authority) / len(authority) if authority else 0.0,
        "dynamic_topology_shadow_last": shadow_norms[-1] if shadow_norms else 0.0,
        "inherited_topology_last": base_topology_norms[-1] if base_topology_norms else 0.0,
    }


def _score(row: dict[str, Any]) -> float:
    perturb_cos = row["rss_negation_final_cosine"]
    memory_cos = row["memory_final_cosine"]
    bottleneck_reuse = row["bottleneck_reuse_ratio"]
    message_norm = row["baseline_mean_message_norm"]
    coupling_abs = abs(row["coupled_final_cosine"])
    injection = row["dynamic_topology_injection_max"]

    topology_bonus = min(injection / 0.001, 1.0)
    message_penalty = max(0.0, (message_norm - 12.0) / 12.0)
    coupling_penalty = max(0.0, coupling_abs - 0.25)
    recovery_floor_penalty = max(0.0, 0.80 - perturb_cos) * 3.0
    memory_floor_penalty = max(0.0, 0.78 - memory_cos) * 3.0

    return (
        1.25 * perturb_cos
        + 1.00 * memory_cos
        + 0.35 * bottleneck_reuse
        + 0.20 * topology_bonus
        - 0.30 * message_penalty
        - 0.50 * coupling_penalty
        - recovery_floor_penalty
        - memory_floor_penalty
    )


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for coupling_scale in _parse_floats(args.coupling_scales):
        kwargs = {"dynamic_topology_coupling_scale": coupling_scale}
        perturb = perturbation_stress_test(
            substrate_name="demian_native_v7.4",
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            perturb_step=args.perturb_step,
            perturb_scales=[args.rss_perturb_scale],
            perturb_mode="rss_negation",
            device=args.device,
            substrate_kwargs=kwargs,
        )
        memory = memory_pair(
            "demian_native_v7.4",
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            initial_delta=args.initial_delta,
            device=args.device,
            substrate_kwargs=kwargs,
        )
        bottleneck = bottleneck_run(
            "demian_native_v7.4",
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            bottleneck_dim=args.bottleneck_dim,
            bottleneck_interval=args.bottleneck_interval,
            device=args.device,
            substrate_kwargs=kwargs,
        )
        coupled = coupled_pair(
            "demian_native_v7.4",
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            coupling_dim=args.coupling_dim,
            coupling_interval=args.coupling_interval,
            coupling_strength=args.coupling_strength,
            device=args.device,
            substrate_kwargs=kwargs,
        )
        trace = _trace_topology(
            coupling_scale=coupling_scale,
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=args.seed,
            device=args.device,
        )
        baseline = perturb["baseline"]
        case = perturb["cases"][0]
        row = {
            "dynamic_topology_coupling_scale": coupling_scale,
            "rss_negation_final_cosine": float(case["final_cosine_vs_baseline"]),
            "rss_negation_peak_message_ratio": float(case["peak_norm_ratio_vs_baseline"]["message"]),
            "memory_final_cosine": float(memory["final_cosine"]),
            "bottleneck_reuse_ratio": float(bottleneck["reuse_ratio"]),
            "bottleneck_unique_codes": float(bottleneck["unique_codes"]),
            "bottleneck_code_entropy": float(bottleneck["code_entropy"]),
            "coupled_final_cosine": float(coupled["final_cosine"]),
            "coupled_mean_cosine": float(coupled["mean_cosine"]),
            "baseline_mean_message_norm": float(baseline["mean_message_norm"]),
            **trace,
        }
        row["score"] = _score(row)
        rows.append(row)
        print(
            f"scale={coupling_scale:.6f} score={row['score']:.4f} "
            f"rss={row['rss_negation_final_cosine']:.4f} "
            f"mem={row['memory_final_cosine']:.4f} "
            f"reuse={row['bottleneck_reuse_ratio']:.4f} "
            f"inj={row['dynamic_topology_injection_max']:.6g}"
        )

    viable = [
        row for row in rows
        if row["rss_negation_final_cosine"] >= args.min_rss_cosine
        and row["memory_final_cosine"] >= args.min_memory_cosine
        and row["baseline_mean_message_norm"] <= args.max_mean_message_norm
    ]
    dynamic_viable = [
        row for row in viable
        if row["dynamic_topology_coupling_scale"] > 0.0
        and row["dynamic_topology_injection_max"] >= args.min_dynamic_topology_injection
    ]
    candidates = dynamic_viable or viable or rows
    best = max(candidates, key=lambda row: (row["score"], row["dynamic_topology_coupling_scale"]))
    return {
        "config": vars(args),
        "selection_rule": {
            "min_rss_cosine": args.min_rss_cosine,
            "min_memory_cosine": args.min_memory_cosine,
            "max_mean_message_norm": args.max_mean_message_norm,
            "min_dynamic_topology_injection": args.min_dynamic_topology_injection,
            "fallback": "max score if no candidate passes thresholds",
        },
        "recommended_dynamic_topology_coupling_scale": best["dynamic_topology_coupling_scale"],
        "recommended": best,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling-scales", default="0,0.003,0.006,0.012,0.024,0.048,0.096")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--perturb-step", type=int, default=128)
    parser.add_argument("--rss-perturb-scale", type=float, default=0.5)
    parser.add_argument("--initial-delta", type=float, default=0.05)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--bottleneck-interval", type=int, default=16)
    parser.add_argument("--coupling-dim", type=int, default=8)
    parser.add_argument("--coupling-interval", type=int, default=16)
    parser.add_argument("--coupling-strength", type=float, default=0.05)
    parser.add_argument("--min-rss-cosine", type=float, default=0.80)
    parser.add_argument("--min-memory-cosine", type=float, default=0.78)
    parser.add_argument("--max-mean-message-norm", type=float, default=12.0)
    parser.add_argument("--min-dynamic-topology-injection", type=float, default=5e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="data/substrate_lab/v74_dynamic_topology_sweep_20260429")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_sweep(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"recommended={payload['recommended_dynamic_topology_coupling_scale']:.6f}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
