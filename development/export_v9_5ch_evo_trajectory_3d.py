#!/usr/bin/env python3
"""Export v9-5ch evolutionary archive candidates to the 3D trajectory schema."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolve_v9_5ch_release import (
    evaluate_genome,
    motif_suite,
    run_with_motif,
)
from development.substrates.trajectory_export import (
    CORE_METRICS,
    DEFAULT_ROUTE_METRICS,
    _clean_float,
    _project_records,
    _projection_feature_names,
)

EXTRA_ROUTE_METRICS = (
    "carrier_state_norm",
    "message_signature_cosine",
    "carrier_signature_cosine",
    "fast_signature_cosine",
    "path_length",
    "net_displacement",
    "path_directness",
    "path_curvature_mean",
    "path_loop_area_proxy",
    "message_signature_tail",
    "carrier_signature_tail",
    "message_signature_peak",
    "carrier_signature_peak",
    "release_duty_cycle",
    "release_transfer",
    "release_pre_accumulation",
    "release_pre_event_pressure",
    "release_local_displacement",
    "release_local_delta_lift",
    "release_local_curvature_lift",
    "release_local_signature_push",
    "release_local_causality",
    "release_geometric_event",
    "phase_transition_score",
    "release_to_fast_norm",
    "release_to_slow_norm",
    "release_to_control_norm",
    "release_to_message_norm",
    "release_to_carrier_norm",
    "first_release_step",
    "release_step_std",
    "release_burstiness",
    "release_eligible_fraction",
    "release_early_fraction",
    "release_late_fraction",
    "pre_release_accumulation_slope",
    "pre_release_tail_accumulation",
    "pressure_at_release_mean",
    "pressure_pre_release_mean",
    "release_after_accumulation_score",
    "release_timing_score",
    "delayed_release_pressure",
    "internal_richness",
    "channel_separation",
    "boundedness",
    "geometric_coherence",
    "plastic_state_norm",
)
ROUTE_METRICS = tuple(dict.fromkeys((*DEFAULT_ROUTE_METRICS, *EXTRA_ROUTE_METRICS)))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def selected_candidates(archive_dir: Path, source_name: str, limit: int) -> list[dict[str, Any]]:
    source = load_json(archive_dir / source_name)
    rows = list(source.values()) if isinstance(source, dict) else list(source)
    rows.sort(key=lambda row: float(row.get("rank_score", 0.0)), reverse=True)
    return rows[:limit]


def collect_records(
    archive_dir: Path,
    source_name: str,
    limit: int,
    motif_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    config = load_json(archive_dir / "config.json")
    candidates = selected_candidates(archive_dir, source_name, limit)
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        genome = candidate["genome"]
        for seed in config["eval_seeds"][:1]:
            motifs = motif_suite(config["hidden_size"], seed)[:motif_limit]
            for scale in config["perturb_scales"][:1]:
                for motif in motifs:
                    run = run_with_motif(
                        genome,
                        hidden_size=config["hidden_size"],
                        steps=config["steps"],
                        seed=seed,
                        perturb_step=config["perturb_step"],
                        perturb_scale=scale,
                        motif=motif,
                        rank=config["rank"],
                        device=config.get("resolved_main_device", config["device"]),
                    )
                    records.extend(records_from_run(candidate, run, config))
    return records, config, candidates


def records_from_run(
    candidate: dict[str, Any],
    run: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    run_key = (
        f"{candidate['id']}|seed={run['seed']}|"
        f"{run['motif_family']}:{run['motif_index']}|scale={run['perturb_scale']}"
    )
    for step in run["trajectory"]:
        route_metrics = {
            key: _clean_float(step.get("route_metrics", {}).get(key, 0.0))
            for key in DEFAULT_ROUTE_METRICS
        }
        for key in EXTRA_ROUTE_METRICS:
            route_metrics[key] = _clean_float(
                step.get(key, run["metrics"].get(key, step.get("route_metrics", {}).get(key, 0.0)))
            )
        metrics = {
            key: _clean_float(step.get(key, 0.0))
            for key in CORE_METRICS
        }
        metrics["route_metrics"] = route_metrics
        point_id = f"{run_key}|step={step['step']}"
        rows.append(
            {
                "point_id": point_id,
                "candidate_id": candidate["id"],
                "substrate": candidate["id"],
                "seed": run["seed"],
                "run_kind": "perturbed",
                "perturb_scale": run["perturb_scale"],
                "perturb_channel": run.get("perturb_channel", "fast"),
                "perturb_mode": run.get("perturb_mode", "external"),
                "perturb_family": run["motif_family"],
                "motif_index": run["motif_index"],
                "regime_class": run["summary"]["regime_class"],
                "step": int(step["step"]),
                "metrics": metrics,
                "summary": run["summary"],
                "is_perturb_step": int(step["step"]) == int(config["perturb_step"]),
                "is_final_step": int(step["step"]) == int(config["steps"]),
            }
        )
    return rows


def export_payload(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    records.sort(key=record_sort_key)
    features = _projection_feature_names(ROUTE_METRICS)
    coords, projection = _project_records(records, features)
    points = []
    metrics = []
    events = []
    for record, coord in zip(records, coords):
        point = {
            "point_id": record["point_id"],
            "x": _clean_float(coord[0]),
            "y": _clean_float(coord[1]),
            "z": _clean_float(coord[2]),
            "step": record["step"],
            "seed": record["seed"],
            "substrate": record["substrate"],
            "candidate_id": record["candidate_id"],
            "run_kind": record["run_kind"],
            "perturb_scale": record["perturb_scale"],
            "perturb_channel": record.get("perturb_channel", "fast"),
            "perturb_mode": record.get("perturb_mode", "external"),
            "perturb_family": record["perturb_family"],
            "motif_index": record["motif_index"],
            "regime_class": record["regime_class"],
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
    return {
        "metadata": {
            "substrates": [candidate["id"] for candidate in candidates],
            "substrate_labels": {
                candidate["id"]: candidate["summary_label"] if "summary_label" in candidate else "v9-5ch evolved candidate"
                for candidate in candidates
            },
            "candidate_metrics": {
                candidate["id"]: candidate["metrics"] for candidate in candidates
            },
            "seeds": config["eval_seeds"],
            "steps": config["steps"],
            "hidden_size": config["hidden_size"],
            "projection_method": "deterministic_pca_on_metric_vectors",
            "projection_features": features,
            "perturb_step": config["perturb_step"],
            "perturb_scales": config["perturb_scales"],
            "schema_version": 2,
            "viewer_note": "paths show bounded machine morphology and release geometry, not return-to-clean recovery or target signature matching",
        },
        "projection": projection,
        "points": points,
        "metrics": metrics,
        "events": events,
        "summaries": {
            "runs": [
                {
                    "candidate_id": candidate["id"],
                    "metrics": candidate["metrics"],
                    "rank_score": candidate.get("rank_score", 0.0),
                }
                for candidate in candidates
            ]
        },
    }


def record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["candidate_id"],
        record["seed"],
        record["perturb_family"],
        record["motif_index"],
        float(record["perturb_scale"]),
        record["step"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export v9-5ch evolution archive to trajectory_3d.json")
    parser.add_argument("--archive-dir", default="data/evolution/v9_5ch_release_20260509")
    parser.add_argument("--source", default="archive.json")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--motif-limit", type=int, default=4)
    parser.add_argument("--out-dir", default="data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_dir = Path(args.archive_dir)
    records, config, candidates = collect_records(
        archive_dir,
        args.source,
        args.limit,
        args.motif_limit,
    )
    payload = export_payload(records, config, candidates)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trajectory_3d.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(out_path)


if __name__ == "__main__":
    main()
