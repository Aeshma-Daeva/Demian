#!/usr/bin/env python3
"""Export 3D trajectories for the v9 message/carrier release-gate probe."""

from __future__ import annotations

import json
import os
import sys
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.probe_v9_message_carrier_strange import (
    HIDDEN_SIZE,
    PERTURB_SCALES,
    PERTURB_STEP,
    SEEDS,
    STEPS,
    ExperimentalV9MessageCarrier,
    accumulator_kwargs,
)
from development.substrates.legacy import SelfLoopRunner
from development.substrates.trajectory_export import (
    DEFAULT_ROUTE_METRICS,
    _clean_float,
    _compact_summary,
    _metrics_from_step,
    _project_records,
    _projection_feature_names,
)

OUT_DIR = Path("data/substrate_lab/v9_release_gate_trajectory_3d_20260509")
ROUTE_METRICS = (
    *DEFAULT_ROUTE_METRICS,
    "binding_active",
)


def tuned_strange_kwargs() -> dict[str, float]:
    return {
        "init_scale": 0.15,
        "state_gain": 1.4,
        "slow_decay": 0.88,
        "control_decay": 0.58,
        "slow_readout_scale": 0.2,
        "control_to_fast_scale": 0.5,
        "fast_to_slow_gate_bias": 0.0,
    }


def variant_kwargs() -> dict[str, dict[str, Any]]:
    base = {
        **accumulator_kwargs(tuned_strange_kwargs()),
        "binding_start_step": 32,
        "initial_message_scale": 0.0,
        "initial_carrier_scale": 0.0,
    }
    return {
        "bound_no_release": dict(base),
        "manual_release_medium": {
            **base,
            "release_step": 96,
            "release_duration": 2,
            "release_gain": 0.25,
        },
        "learned_release_neutral": {
            **base,
            "release_policy": "learned",
            "release_threshold": 0.0,
            "release_temperature": 1.0,
            "release_gain": 0.30,
        },
        "learned_release_high_threshold": {
            **base,
            "release_policy": "learned",
            "release_threshold": 0.25,
            "release_temperature": 0.75,
            "release_gain": 0.30,
        },
    }


def collect_records(
    *,
    seeds: tuple[int, ...] = SEEDS,
    perturb_scales: tuple[float, ...] = PERTURB_SCALES,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, kwargs in variant_kwargs().items():
        for seed in seeds:
            records.extend(run_case(label, kwargs, seed, "clean", None))
            for scale in perturb_scales:
                records.extend(run_case(label, kwargs, seed, "perturbed", scale))
    return records


def run_case(
    label: str,
    kwargs: dict[str, Any],
    seed: int,
    run_kind: str,
    perturb_scale: float | None,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    model = ExperimentalV9MessageCarrier(HIDDEN_SIZE, **kwargs)
    runner = SelfLoopRunner(model, device="cpu")
    trajectory, summary, _ = runner.run(
        steps=STEPS,
        seed=seed,
        perturb_step=PERTURB_STEP if perturb_scale is not None else None,
        perturb_scale=perturb_scale or 0.0,
    )
    rows = []
    for step in trajectory:
        step_dict = asdict(step)
        point_id = (
            f"{label}|seed={seed}|{run_kind}|"
            f"scale={'clean' if perturb_scale is None else perturb_scale}|step={step.step}"
        )
        rows.append(
            {
                "point_id": point_id,
                "substrate": label,
                "seed": seed,
                "run_kind": run_kind,
                "perturb_scale": perturb_scale,
                "step": int(step.step),
                "metrics": _metrics_from_step(step_dict, ROUTE_METRICS),
                "summary": asdict(summary),
                "is_perturb_step": run_kind == "perturbed" and step.step == PERTURB_STEP,
                "is_final_step": step.step == STEPS,
            }
        )
    return rows


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument(
        "--perturb-scales",
        default=",".join(str(scale) for scale in PERTURB_SCALES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    perturb_scales = parse_csv_floats(args.perturb_scales)
    records = collect_records(seeds=seeds, perturb_scales=perturb_scales)
    records.sort(key=record_sort_key)
    features = _projection_feature_names(ROUTE_METRICS)
    coords, projection = _project_records(records, features)

    points = []
    metrics = []
    events = []
    summaries: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for record, coord in zip(records, coords):
        point = {
            "point_id": record["point_id"],
            "x": _clean_float(coord[0]),
            "y": _clean_float(coord[1]),
            "z": _clean_float(coord[2]),
            "step": record["step"],
            "seed": record["seed"],
            "substrate": record["substrate"],
            "run_kind": record["run_kind"],
            "perturb_scale": record["perturb_scale"],
        }
        points.append(point)
        metrics.append({"point_id": record["point_id"], **record["metrics"]})
        if record["is_perturb_step"] or record["is_final_step"]:
            events.append(
                {
                    "event_kind": "perturbation" if record["is_perturb_step"] else "final_state",
                    **point,
                }
            )
        scale_key = "clean" if record["perturb_scale"] is None else str(record["perturb_scale"])
        summary_key = (record["substrate"], record["seed"], record["run_kind"], scale_key)
        summaries.setdefault(
            summary_key,
            {
                "substrate": record["substrate"],
                "seed": record["seed"],
                "run_kind": record["run_kind"],
                "perturb_scale": record["perturb_scale"],
                "summary": _compact_summary(record["summary"]),
            },
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trajectory_3d.json"
    out_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "substrates": list(variant_kwargs()),
                    "substrate_labels": {
                        key: "v9 message/carrier release probe" for key in variant_kwargs()
                    },
                    "seeds": list(seeds),
                    "steps": STEPS,
                    "hidden_size": HIDDEN_SIZE,
                    "projection_method": "deterministic_pca_on_metric_vectors",
                    "projection_features": features,
                    "perturb_step": PERTURB_STEP,
                    "perturb_scales": list(perturb_scales),
                    "perturb_mode": "noise",
                    "schema_version": 1,
                    "viewer_note": "release openness and strength are internal gate traces",
                },
                "projection": projection,
                "points": points,
                "metrics": metrics,
                "events": events,
                "summaries": {"runs": [summaries[key] for key in sorted(summaries)]},
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(out_path)


def record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    run_order = 0 if record["run_kind"] == "clean" else 1
    scale = -1.0 if record["perturb_scale"] is None else float(record["perturb_scale"])
    return (record["substrate"], record["seed"], run_order, scale, record["step"])


if __name__ == "__main__":
    main()
