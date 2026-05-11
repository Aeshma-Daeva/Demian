"""Run first-pass v7.4 organ ablations.

This is a falsification runner, not a parameter optimizer. It asks whether
named v7.4 organs change behavior under existing stress probes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (  # noqa: E402
    SelfLoopRunner,
    _code_stats,
    _decode_code,
    _encode_state,
    _fixed_projection,
    _make_substrate,
    _manual_step_trace,
    _mean_abs_metric_gap,
    _clone_substrate_state,
)


NANO_STEP = {
    "dynamic_step_min": 0.96,
    "dynamic_step_max": 1.01,
    "dynamic_step_hold_slowdown": 0.03,
    "dynamic_step_refusal_slowdown": 0.02,
    "dynamic_step_recovery_accel": 0.015,
    "dynamic_step_integrate_accel": 0.010,
}

FIXED_STEP = {
    "dynamic_step_min": 1.0,
    "dynamic_step_max": 1.0,
    "dynamic_step_hold_slowdown": 0.0,
    "dynamic_step_refusal_slowdown": 0.0,
    "dynamic_step_recovery_accel": 0.0,
    "dynamic_step_integrate_accel": 0.0,
}

ROUTE_NAMES = [
    "v74_viability_mean",
    "v74_ownership_mean",
    "v74_tension_mean",
    "v74_hold_pressure_mean",
    "v74_resolution_open_mean",
    "v74_topology_state_norm",
    "v74_dynamic_topology_injection_norm",
    "topology_state_norm",
]


def _parse_ints(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def _zero_index(state: object, index: int) -> object:
    if not isinstance(state, tuple) or len(state) <= index:
        return state
    items = list(state)
    items[index] = torch.zeros_like(items[index])
    return tuple(items)


def _wrap_step(model: torch.nn.Module, variant: str) -> torch.nn.Module:
    original_step = model.step

    if variant == "injection_without_shadow_memory":

        def step_without_shadow_memory(state: object) -> object:
            state = _zero_index(state, 22)
            new_state = original_step(state)
            return _zero_index(new_state, 22)

        model.step = step_without_shadow_memory  # type: ignore[method-assign]

    return model


def _variant_kwargs(name: str) -> dict[str, float]:
    if name == "full":
        return {}
    if name == "fixed_step":
        return dict(FIXED_STEP)
    if name == "nano_step":
        return dict(NANO_STEP)
    if name == "no_self_policy_drive":
        return {"self_policy_scale": 0.0}
    if name == "no_quarantine":
        return {"quarantine_scale": 0.0, "quarantine_retention": 0.0}
    if name == "no_dynamic_topology":
        return {
            "dynamic_topology_coupling_scale": 0.0,
            "self_topology_scale": 0.0,
            "hold_topology_scale": 0.0,
        }
    if name == "shadow_without_injection":
        return {"dynamic_topology_coupling_scale": 0.0}
    if name == "injection_without_shadow_memory":
        return {}
    if name == "pressure_policy":
        return {"pressure_policy_blend": 1.0}
    if name == "pressure_policy_no_self_drive":
        return {"pressure_policy_blend": 1.0, "self_policy_scale": 0.0}
    match = re.fullmatch(r"pressure_blend_([0-9p]+)_floor_([0-9p]+)", name)
    if match:
        return {
            "pressure_policy_blend": float(match.group(1).replace("p", ".")),
            "pressure_policy_floor": float(match.group(2).replace("p", ".")),
        }
    raise ValueError(f"unknown variant: {name}")


def _make_variant(name: str, hidden_size: int, device: str) -> torch.nn.Module:
    model = _make_substrate("demian_native_v7.4", hidden_size, _variant_kwargs(name))
    model = _wrap_step(model, name)
    return model.to(device=torch.device(device), dtype=torch.float32)


def _perturb_probe(
    model: torch.nn.Module,
    *,
    steps: int,
    seed: int,
    perturb_step: int,
    perturb_scale: float,
    perturb_mode: str,
    device: str,
) -> dict[str, Any]:
    runner = SelfLoopRunner(model, device=device)
    base_state = model.initial_state(1, runner.device)
    _, baseline_summary, baseline_final = runner.run(steps=steps, seed=seed, initial_state=base_state)
    _, summary, final_state = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
        perturb_step=perturb_step,
        perturb_scale=perturb_scale,
        perturb_mode=perturb_mode,
    )
    return {
        "baseline_mean_message_norm": float(baseline_summary.mean_message_norm),
        "rss_negation_final_cosine": float(torch.nn.functional.cosine_similarity(baseline_final, final_state, dim=0).item()),
        "rss_negation_peak_message_ratio": float(summary.max_message_norm / (baseline_summary.max_message_norm + 1e-10)),
    }


def _memory_probe(
    model: torch.nn.Module,
    *,
    steps: int,
    seed: int,
    initial_delta: float,
    device: str,
) -> dict[str, Any]:
    runner = SelfLoopRunner(model, device=device)
    base_state = model.initial_state(1, runner.device)
    alt_state = tuple(s + initial_delta * torch.randn_like(s) for s in base_state)
    _, _, clean_final = runner.run(steps=steps, seed=seed, initial_state=base_state)
    _, _, alt_final = runner.run(steps=steps, seed=seed, initial_state=alt_state)
    return {
        "memory_final_cosine": float(torch.nn.functional.cosine_similarity(clean_final, alt_final, dim=0).item()),
        "memory_final_l2_gap": float(torch.norm(clean_final - alt_final).item()),
    }


def _bottleneck_probe(
    model: torch.nn.Module,
    *,
    hidden_size: int,
    steps: int,
    seed: int,
    bottleneck_dim: int,
    bottleneck_interval: int,
    device: str,
) -> dict[str, Any]:
    runner = SelfLoopRunner(model, device=device)
    proj = _fixed_projection(hidden_size, bottleneck_dim, seed + 101)
    base_state = model.initial_state(1, runner.device)
    _, _, clean_final = runner.run(steps=steps, seed=seed, initial_state=base_state)

    with torch.no_grad():
        state = base_state
        codes: list[torch.Tensor] = []
        for step_idx in range(1, steps + 1):
            state = model.step(state)
            h = model.state_vector(state).view(-1).detach().float().cpu()
            if step_idx % bottleneck_interval == 0:
                code = _encode_state(h, proj)
                codes.append(code)
                recon = _decode_code(code, proj).to(runner.device, dtype=runner.dtype)
                recon = recon * (h.norm() + 1e-10)
                state = model.write_surface_state(state, recon)
        bottleneck_final = model.state_vector(state).view(-1).detach().float().cpu()

    stats = _code_stats(codes)
    return {
        "bottleneck_final_cosine": float(torch.nn.functional.cosine_similarity(clean_final, bottleneck_final, dim=0).item()),
        "bottleneck_reuse_ratio": float(stats.get("reuse_ratio", 0.0)),
        "bottleneck_unique_codes": float(stats.get("unique_codes", 0.0)),
        "bottleneck_code_entropy": float(stats.get("code_entropy", 0.0)),
    }


def _coupled_probe(
    model_factory: Callable[[], torch.nn.Module],
    *,
    hidden_size: int,
    steps: int,
    seed: int,
    coupling_dim: int,
    coupling_interval: int,
    coupling_strength: float,
    device: str,
) -> dict[str, Any]:
    model_a = model_factory()
    model_b = model_factory()
    runner_a = SelfLoopRunner(model_a, device=device)
    runner_b = SelfLoopRunner(model_b, device=device)
    proj = _fixed_projection(hidden_size, coupling_dim, seed + 211)
    state_a = model_a.initial_state(1, runner_a.device)
    state_b = model_b.initial_state(1, runner_b.device)
    cosines: list[float] = []

    with torch.no_grad():
        for step_idx in range(1, steps + 1):
            state_a = model_a.step(state_a)
            state_b = model_b.step(state_b)
            h_a = model_a.state_vector(state_a).view(-1).detach().float().cpu()
            h_b = model_b.state_vector(state_b).view(-1).detach().float().cpu()
            if step_idx % coupling_interval == 0:
                code_a = _encode_state(h_a, proj)
                code_b = _encode_state(h_b, proj)
                msg_a = _decode_code(code_a, proj).to(runner_a.device, dtype=runner_a.dtype)
                msg_b = _decode_code(code_b, proj).to(runner_b.device, dtype=runner_b.dtype)
                state_a = model_a.inject_coupling_message(state_a, msg_b, coupling_strength)
                state_b = model_b.inject_coupling_message(state_b, msg_a, coupling_strength)
                h_a = model_a.state_vector(state_a).view(-1).detach().float().cpu()
                h_b = model_b.state_vector(state_b).view(-1).detach().float().cpu()
            cosines.append(float(torch.nn.functional.cosine_similarity(h_a, h_b, dim=0).item()))

    return {
        "coupled_final_cosine": cosines[-1] if cosines else 0.0,
        "coupled_mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
    }


def _resume_probe(
    model_factory: Callable[[], torch.nn.Module],
    *,
    pause_steps: int,
    resume_steps: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    source = model_factory()
    runner_device = torch.device(device)
    pause_state = source.initial_state(1, runner_device)
    pause_state, _pre_vectors, _pre_metrics = _manual_step_trace(source, pause_state, pause_steps)
    pause_surface = source.state_vector(pause_state).detach().clone()
    pause_body = {name: value.detach().clone() for name, value in source.state_dict().items()}
    _uninterrupted_state, uninterrupted_vectors, uninterrupted_metrics = _manual_step_trace(
        source,
        _clone_substrate_state(pause_state),
        resume_steps,
    )

    def run_arm(*, restore_body: bool, restore_state: bool, restore_surface: bool) -> tuple[list[torch.Tensor], list[dict[str, float]]]:
        torch.manual_seed(seed)
        model = model_factory()
        if restore_body:
            model.load_state_dict(pause_body)
        if restore_state:
            arm_state = _clone_substrate_state(pause_state)
        else:
            arm_state = model.initial_state(1, runner_device)
        if restore_surface:
            arm_state = model.write_surface_state(arm_state, pause_surface)
        _state, vectors, metrics = _manual_step_trace(model, arm_state, resume_steps)
        return vectors, metrics

    capsule_vectors, capsule_metrics = run_arm(restore_body=True, restore_state=True, restore_surface=False)
    notebook_vectors, _notebook_metrics = run_arm(restore_body=False, restore_state=False, restore_surface=True)

    def final_cos(vectors: list[torch.Tensor]) -> float:
        if not uninterrupted_vectors or not vectors:
            return 0.0
        return float(torch.nn.functional.cosine_similarity(uninterrupted_vectors[-1], vectors[-1], dim=0).item())

    def mean_gap(vectors: list[torch.Tensor]) -> float:
        gaps = [
            float(torch.norm(a - b)) / np.sqrt(max(a.shape[0], 1))
            for a, b in zip(uninterrupted_vectors, vectors)
        ]
        return float(np.mean(gaps)) if gaps else 0.0

    capsule_gap = mean_gap(capsule_vectors)
    notebook_gap = mean_gap(notebook_vectors)
    return {
        "resume_capsule_final_cosine": final_cos(capsule_vectors),
        "resume_notebook_final_cosine": final_cos(notebook_vectors),
        "resume_capsule_mean_gap": capsule_gap,
        "resume_notebook_mean_gap": notebook_gap,
        "resume_trajectory_shape_ratio": float(notebook_gap / (capsule_gap + 1e-10)),
        "resume_route_metric_mean_abs_gap": _mean_abs_metric_gap(uninterrupted_metrics, capsule_metrics, ROUTE_NAMES),
    }


def _route_trace_probe(
    model_factory: Callable[[], torch.nn.Module],
    *,
    steps: int,
    seed: int,
    device: str,
    baseline_metrics: list[dict[str, float]] | None,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = model_factory()
    state = model.initial_state(1, torch.device(device))
    _state, _vectors, metrics = _manual_step_trace(model, state, steps)
    summary: dict[str, Any] = {}
    for name in ROUTE_NAMES:
        values = [float(row.get(name, 0.0)) for row in metrics]
        summary[f"{name}_mean"] = float(np.mean(values)) if values else 0.0
        summary[f"{name}_last"] = values[-1] if values else 0.0
    if baseline_metrics is not None:
        summary["route_metric_mean_abs_gap_vs_full"] = _mean_abs_metric_gap(baseline_metrics, metrics, ROUTE_NAMES)
    return summary, metrics


def run_variant(name: str, args: argparse.Namespace) -> dict[str, Any]:
    def factory() -> torch.nn.Module:
        return _make_variant(name, args.hidden_size, args.device)

    seed_rows: list[dict[str, Any]] = []
    for seed in _parse_ints(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        perturb = _perturb_probe(
            factory(),
            steps=args.steps,
            seed=seed,
            perturb_step=args.perturb_step,
            perturb_scale=args.rss_perturb_scale,
            perturb_mode="rss_negation",
            device=args.device,
        )
        memory = _memory_probe(
            factory(),
            steps=args.steps,
            seed=seed,
            initial_delta=args.initial_delta,
            device=args.device,
        )
        bottleneck = _bottleneck_probe(
            factory(),
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=seed,
            bottleneck_dim=args.bottleneck_dim,
            bottleneck_interval=args.bottleneck_interval,
            device=args.device,
        )
        coupled = _coupled_probe(
            factory,
            hidden_size=args.hidden_size,
            steps=args.steps,
            seed=seed,
            coupling_dim=args.coupling_dim,
            coupling_interval=args.coupling_interval,
            coupling_strength=args.coupling_strength,
            device=args.device,
        )
        resume = _resume_probe(
            factory,
            pause_steps=args.pause_steps,
            resume_steps=args.resume_steps,
            seed=seed,
            device=args.device,
        )
        baseline_route_metrics: list[dict[str, float]] | None = None
        if name != "full":
            def full_factory() -> torch.nn.Module:
                return _make_variant("full", args.hidden_size, args.device)

            _baseline_route_summary, baseline_route_metrics = _route_trace_probe(
                full_factory,
                steps=args.steps,
                seed=seed,
                device=args.device,
                baseline_metrics=None,
            )
        route_summary, _route_metrics = _route_trace_probe(
            factory,
            steps=args.steps,
            seed=seed,
            device=args.device,
            baseline_metrics=baseline_route_metrics,
        )
        seed_rows.append({
            "seed": seed,
            **perturb,
            **memory,
            **bottleneck,
            **coupled,
            **resume,
            **route_summary,
        })

    aggregate: dict[str, Any] = {}
    numeric_keys = sorted({
        key
        for row in seed_rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and key != "seed"
    })
    for key in numeric_keys:
        values = [float(row[key]) for row in seed_rows if key in row]
        aggregate[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
        aggregate[f"{key}_std"] = float(np.std(values)) if values else 0.0

    return {
        "variant": name,
        "substrate_kwargs": _variant_kwargs(name),
        "aggregate": aggregate,
        "seeds": seed_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        default="full,pressure_policy,pressure_policy_no_self_drive,fixed_step,nano_step,no_self_policy_drive,no_quarantine,no_dynamic_topology,shadow_without_injection,injection_without_shadow_memory",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seeds", default="94,95,96,97")
    parser.add_argument("--perturb-step", type=int, default=64)
    parser.add_argument("--rss-perturb-scale", type=float, default=0.5)
    parser.add_argument("--initial-delta", type=float, default=0.05)
    parser.add_argument("--bottleneck-dim", type=int, default=8)
    parser.add_argument("--bottleneck-interval", type=int, default=16)
    parser.add_argument("--coupling-dim", type=int, default=8)
    parser.add_argument("--coupling-interval", type=int, default=16)
    parser.add_argument("--coupling-strength", type=float, default=0.05)
    parser.add_argument("--pause-steps", type=int, default=64)
    parser.add_argument("--resume-steps", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="data/substrate_lab/v74_organ_ablation_20260429")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    rows = []
    for variant in variants:
        row = run_variant(variant, args)
        rows.append(row)
        agg = row["aggregate"]
        print(
            f"{variant}: "
            f"rss={agg.get('rss_negation_final_cosine_mean', 0.0):.4f} "
            f"mem={agg.get('memory_final_cosine_mean', 0.0):.4f} "
            f"reuse={agg.get('bottleneck_reuse_ratio_mean', 0.0):.4f} "
            f"couple={agg.get('coupled_final_cosine_mean', 0.0):+.4f} "
            f"resume={agg.get('resume_capsule_final_cosine_mean', 0.0):.6f} "
            f"topoinj={agg.get('v74_dynamic_topology_injection_norm_mean_mean', 0.0):.6g}"
        )

    payload = {
        "config": vars(args),
        "variants": rows,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
