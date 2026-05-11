"""Probe whether v7.4 ownership/viability quadrants separate policy gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import _make_substrate  # noqa: E402


GATE_NAMES = [
    "v74_preserve_gate_mean",
    "v74_adapt_gate_mean",
    "v74_recover_gate_mean",
    "v74_refuse_gate_mean",
    "v74_quarantine_gate_mean",
    "v74_integrate_gate_mean",
    "v74_hold_gate_mean",
]

METRIC_NAMES = [
    "v74_viability_mean",
    "v74_ownership_mean",
    "v74_tension_mean",
    "v74_integrate_pressure_mean",
    "v74_refuse_pressure_mean",
    "v74_recover_pressure_mean",
    "v74_reject_pressure_mean",
    "v74_hold_pressure_mean",
    "v74_pressure_entropy_mean",
    "v74_resolution_open_mean",
    "v74_dynamic_step_dt_mean",
    *GATE_NAMES,
]


def _unit_vector(hidden_size: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    vec = torch.randn(1, hidden_size, generator=generator, device=device)
    return vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-8)


def _set_surface_norm(
    model: torch.nn.Module,
    state: tuple[torch.Tensor, ...],
    target_rss_gain: float,
) -> tuple[torch.Tensor, ...]:
    """Set previous_surface to target v7.4 rss_gain approximately."""
    target_norm = model.rss_floor + target_rss_gain * (model.rss_target - model.rss_floor)
    surface = _unit_vector(model.hidden_size, 9001, state[0].device) * target_norm * math.sqrt(model.hidden_size)
    items = list(state)
    items[18] = surface
    return tuple(items)


def _make_quadrant_state(
    model: torch.nn.Module,
    seed: int,
    ownership_high: bool,
    viability_high: bool,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    state = model.initial_state(1, device)
    items = list(state)
    hidden = model.hidden_size

    owned = _unit_vector(hidden, seed + 11, device) * 1.5
    other = -owned
    support = _unit_vector(hidden, seed + 23, device) * 1.5
    noise = 0.02 * _unit_vector(hidden, seed + 37, device)

    # Indices from DemianNativeV71 state layout:
    # 11 ancestry, 12 active, 13 projection, 15 trajectory_memory,
    # 16 trajectory_valence, 17 trajectory_reuse, 18 previous_surface,
    # 19 resource_state.
    if ownership_high:
        items[12] = owned
        items[15] = noise
        items[17] = noise
        items[11] = owned
        items[16] = noise
    else:
        items[12] = owned
        items[15] = noise
        items[17] = noise
        items[11] = other
        items[16] = noise

    if viability_high:
        items[19] = torch.full_like(items[19], 3.0)
        items[13] = items[12].clone()
        items[15] = support
        items[17] = support.clone()
        state = tuple(items)
        state = _set_surface_norm(model, state, target_rss_gain=1.0)
        items = list(state)
    else:
        items[19] = torch.full_like(items[19], -3.0)
        items[13] = -items[12]
        items[15] = support
        items[17] = -support
        state = tuple(items)
        state = _set_surface_norm(model, state, target_rss_gain=0.0)
        items = list(state)

    # Reset v7.4-specific buffers so the first response is from the quadrant,
    # not from retained potential/quarantine.
    items[20] = torch.zeros_like(items[20])
    items[21] = torch.zeros_like(items[21])
    items[22] = torch.zeros_like(items[22])
    return tuple(items)


def _run_case(
    model: torch.nn.Module,
    quadrant: str,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    ownership_high = quadrant.startswith("high_ownership")
    viability_high = quadrant.endswith("high_viability")
    state = _make_quadrant_state(model, seed, ownership_high, viability_high, device)
    with torch.no_grad():
        _new_state = model.step(state)
    aux = dict(model.step_aux())
    return {
        "quadrant": quadrant,
        "seed": seed,
        **{name: float(aux.get(name, 0.0)) for name in METRIC_NAMES},
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_quadrant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_quadrant.setdefault(row["quadrant"], []).append(row)

    summary: dict[str, Any] = {}
    for quadrant, qrows in by_quadrant.items():
        qsummary: dict[str, float] = {}
        for name in METRIC_NAMES:
            values = [float(row[name]) for row in qrows]
            qsummary[f"{name}_mean"] = float(np.mean(values))
            qsummary[f"{name}_std"] = float(np.std(values))
        gate_means = {name: qsummary[f"{name}_mean"] for name in GATE_NAMES}
        qsummary["dominant_gate"] = max(gate_means, key=gate_means.get)  # type: ignore[assignment]
        qsummary["gate_spread"] = float(max(gate_means.values()) - min(gate_means.values()))
        summary[quadrant] = qsummary

    # A compact separation diagnostic: average dominant-gate margin against
    # uniform softmax. Low values mean the policy is mostly not choosing.
    uniform = 1.0 / len(GATE_NAMES)
    margins = [
        abs(float(qsummary[f"{qsummary['dominant_gate']}_mean"]) - uniform)
        for qsummary in summary.values()
    ]
    summary["_diagnostics"] = {
        "mean_dominant_margin_from_uniform": float(np.mean(margins)) if margins else 0.0,
        "unique_dominant_gates": sorted({str(qsummary["dominant_gate"]) for qsummary in summary.values()}),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seeds", default="94,95,96,97")
    parser.add_argument("--pressure-policy-blend", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="data/substrate_lab/v74_ownership_quadrants_20260429")
    return parser.parse_args()


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    quadrants = [
        "high_ownership_high_viability",
        "high_ownership_low_viability",
        "low_ownership_high_viability",
        "low_ownership_low_viability",
    ]
    rows: list[dict[str, Any]] = []
    for seed in _parse_ints(args.seeds):
        torch.manual_seed(seed)
        model = _make_substrate(
            "demian_native_v7.4",
            args.hidden_size,
            {"pressure_policy_blend": args.pressure_policy_blend},
        )
        model = model.to(device=device, dtype=torch.float32)
        for quadrant in quadrants:
            rows.append(_run_case(model, quadrant, seed, device))

    summary = _summarize(rows)
    payload = {
        "config": vars(args),
        "quadrants": summary,
        "rows": rows,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for quadrant in quadrants:
        q = summary[quadrant]
        print(
            f"{quadrant}: "
            f"own={q['v74_ownership_mean_mean']:.3f} "
            f"via={q['v74_viability_mean_mean']:.3f} "
            f"dom={q['dominant_gate']} "
            f"spread={q['gate_spread']:.4f} "
            f"hold={q['v74_hold_gate_mean_mean']:.3f} "
            f"int={q['v74_integrate_gate_mean_mean']:.3f} "
            f"rec={q['v74_recover_gate_mean_mean']:.3f} "
            f"ref={q['v74_refuse_gate_mean_mean']:.3f} "
            f"qua={q['v74_quarantine_gate_mean_mean']:.3f}"
        )
    print(f"diagnostics: {summary['_diagnostics']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
