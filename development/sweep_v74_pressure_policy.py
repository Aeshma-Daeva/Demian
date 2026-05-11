"""Sweep v7.4 pressure-policy blend/floor calibration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.ablate_native_v74_organs import run_variant  # noqa: E402
from development.probe_v74_ownership_viability_quadrants import (  # noqa: E402
    GATE_NAMES,
    _make_quadrant_state,
    _parse_ints,
    _summarize,
)
from development.substrate_lab import _make_substrate  # noqa: E402

import torch  # noqa: E402


def _parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _variant_name(blend: float, floor: float) -> str:
    return f"pressure_blend_{blend:g}_floor_{floor:g}".replace(".", "p")


def _quadrant_rows(
    *,
    hidden_size: int,
    seeds: list[int],
    device: str,
    blend: float,
    floor: float,
) -> list[dict[str, Any]]:
    quadrants = [
        "high_ownership_high_viability",
        "high_ownership_low_viability",
        "low_ownership_high_viability",
        "low_ownership_low_viability",
    ]
    rows: list[dict[str, Any]] = []
    torch_device = torch.device(device)
    for seed in seeds:
        torch.manual_seed(seed)
        model = _make_substrate(
            "demian_native_v7.4",
            hidden_size,
            {
                "pressure_policy_blend": blend,
                "pressure_policy_floor": floor,
            },
        )
        model = model.to(device=torch_device, dtype=torch.float32)
        for quadrant in quadrants:
            ownership_high = quadrant.startswith("high_ownership")
            viability_high = quadrant.endswith("high_viability")
            state = _make_quadrant_state(model, seed, ownership_high, viability_high, torch_device)
            with torch.no_grad():
                _new_state = model.step(state)
            aux = dict(model.step_aux())
            rows.append(
                {
                    "quadrant": quadrant,
                    "seed": seed,
                    **{name: float(aux.get(name, 0.0)) for name in [
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
                    ]},
                }
            )
    return rows


def _expected_gate_accuracy(summary: dict[str, Any]) -> float:
    expected = {
        "high_ownership_high_viability": "v74_integrate_gate_mean",
        "high_ownership_low_viability": "v74_recover_gate_mean",
        "low_ownership_high_viability": "v74_refuse_gate_mean",
        "low_ownership_low_viability": "v74_quarantine_gate_mean",
    }
    correct = 0
    total = 0
    for quadrant, gate in expected.items():
        if quadrant in summary:
            correct += int(summary[quadrant]["dominant_gate"] == gate)
            total += 1
    return correct / total if total else 0.0


def _score(row: dict[str, Any]) -> float:
    behavior = row["behavior"]["aggregate"]
    rss = behavior.get("rss_negation_final_cosine_mean", 0.0)
    memory = behavior.get("memory_final_cosine_mean", 0.0)
    reuse = behavior.get("bottleneck_reuse_ratio_mean", 0.0)
    resume = behavior.get("resume_capsule_final_cosine_mean", 0.0)
    gate_margin = row["quadrant_summary"]["_diagnostics"]["mean_dominant_margin_from_uniform"]
    gate_accuracy = row["expected_gate_accuracy"]
    memory_penalty = max(0.0, 0.75 - memory) * 3.0
    rss_penalty = max(0.0, 0.62 - rss) * 3.0
    resume_penalty = max(0.0, 0.978 - resume) * 10.0
    return (
        1.1 * rss
        + 1.0 * memory
        + 0.3 * reuse
        + 0.4 * gate_accuracy
        + 0.2 * min(gate_margin / 0.20, 1.0)
        - memory_penalty
        - rss_penalty
        - resume_penalty
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blends", default="0,0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--floors", default="0.005,0.01,0.03")
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
    parser.add_argument("--out-dir", default="data/substrate_lab/v74_pressure_policy_sweep_20260429")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    seeds = _parse_ints(args.seeds)
    for blend in _parse_floats(args.blends):
        for floor in _parse_floats(args.floors):
            variant_name = _variant_name(blend, floor)
            behavior_args = argparse.Namespace(**vars(args))
            behavior_args.variants = variant_name
            behavior = run_variant(variant_name, behavior_args)
            quadrant_rows = _quadrant_rows(
                hidden_size=args.hidden_size,
                seeds=seeds,
                device=args.device,
                blend=blend,
                floor=floor,
            )
            quadrant_summary = _summarize(quadrant_rows)
            row = {
                "blend": blend,
                "floor": floor,
                "variant": variant_name,
                "behavior": behavior,
                "quadrant_summary": quadrant_summary,
                "expected_gate_accuracy": _expected_gate_accuracy(quadrant_summary),
            }
            row["score"] = _score(row)
            rows.append(row)
            agg = behavior["aggregate"]
            diag = quadrant_summary["_diagnostics"]
            print(
                f"blend={blend:g} floor={floor:g} score={row['score']:.4f} "
                f"rss={agg.get('rss_negation_final_cosine_mean', 0.0):.4f} "
                f"mem={agg.get('memory_final_cosine_mean', 0.0):.4f} "
                f"resume={agg.get('resume_capsule_final_cosine_mean', 0.0):.6f} "
                f"gate_acc={row['expected_gate_accuracy']:.2f} "
                f"gate_margin={diag['mean_dominant_margin_from_uniform']:.4f}"
            )

    best = max(rows, key=lambda row: row["score"])
    payload = {
        "config": vars(args),
        "selection_rule": "score balances stress behavior, memory, resume, and quadrant gate separation",
        "recommended": {
            "blend": best["blend"],
            "floor": best["floor"],
            "variant": best["variant"],
            "score": best["score"],
            "expected_gate_accuracy": best["expected_gate_accuracy"],
            "behavior_aggregate": best["behavior"]["aggregate"],
            "quadrant_diagnostics": best["quadrant_summary"]["_diagnostics"],
        },
        "rows": rows,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"recommended blend={best['blend']:g} floor={best['floor']:g} score={best['score']:.4f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
