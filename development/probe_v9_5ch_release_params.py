#!/usr/bin/env python3
"""Probe release scalar sensitivity for archived v9 five-channel candidates."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolve_v9_5ch_release import aggregate_runs, motif_suite, repair_genome, run_with_motif
from development.export_v9_5ch_evo_trajectory_3d import export_payload, records_from_run


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_archive_rows(path: Path) -> list[dict[str, Any]]:
    raw = load_json(path)
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    rows.sort(key=lambda row: float(row.get("rank_score", 0.0)), reverse=True)
    return rows


def select_candidate(rows: list[dict[str, Any]], candidate_id: str | None) -> dict[str, Any]:
    if candidate_id is None:
        return rows[0]
    for row in rows:
        if row.get("id") == candidate_id:
            return row
    raise SystemExit(f"candidate not found: {candidate_id}")


def scalar_variants(
    base_scalars: dict[str, float],
    threshold_deltas: list[float],
    temperature_scales: list[float],
    gain_scales: list[float],
) -> list[dict[str, Any]]:
    variants = []
    for threshold_delta in threshold_deltas:
        for temperature_scale in temperature_scales:
            for gain_scale in gain_scales:
                scalars = dict(base_scalars)
                scalars["release_threshold"] += threshold_delta
                scalars["release_temperature"] *= temperature_scale
                scalars["release_gain"] *= gain_scale
                label = (
                    f"thr_{threshold_delta:+.3f}"
                    f"__temp_x{temperature_scale:.3f}"
                    f"__gain_x{gain_scale:.3f}"
                )
                variants.append(
                    {
                        "label": safe_id(label),
                        "scalar_overrides": {
                            "release_threshold": scalars["release_threshold"],
                            "release_temperature": scalars["release_temperature"],
                            "release_gain": scalars["release_gain"],
                        },
                        "scalars": scalars,
                    }
                )
    return variants


def safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def genome_with_scalars(candidate: dict[str, Any], scalars: dict[str, float], config: dict[str, Any]) -> dict[str, Any]:
    genome = copy.deepcopy(candidate["genome"])
    genome["scalars"].update(scalars)
    return repair_genome(genome, int(config["hidden_size"]), int(config["rank"]))


def run_variant(
    candidate: dict[str, Any],
    variant: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    genome = genome_with_scalars(candidate, variant["scalars"], config)
    runs = []
    records = []
    seeds = config["eval_seeds"][: args.seed_limit]
    scales = config["perturb_scales"][: args.scale_limit]
    for seed in seeds:
        motifs = motif_suite(int(config["hidden_size"]), int(seed))[: args.motif_limit]
        for scale in scales:
            for motif in motifs:
                run = run_with_motif(
                    genome,
                    hidden_size=int(config["hidden_size"]),
                    steps=int(config["steps"]),
                    seed=int(seed),
                    perturb_step=int(config["perturb_step"]),
                    perturb_scale=float(scale),
                    motif=motif,
                    rank=int(config["rank"]),
                    device=str(config["device"]),
                )
                runs.append(run)
                variant_candidate = {
                    "id": f"{candidate['id']}__{variant['label']}",
                    "metrics": {},
                    "rank_score": 0.0,
                    "summary_label": (
                        f"{candidate['id']} release probe "
                        f"T={variant['scalar_overrides']['release_threshold']:.3f} "
                        f"tau={variant['scalar_overrides']['release_temperature']:.3f} "
                        f"gain={variant['scalar_overrides']['release_gain']:.3f}"
                    ),
                }
                records.extend(records_from_run(variant_candidate, run, config))
    aggregate = aggregate_runs([{key: value for key, value in run.items() if key != "trajectory"} for run in runs])
    aggregate.update(release_probe_summary(runs))
    summary = {
        "id": f"{candidate['id']}__{variant['label']}",
        "source_candidate_id": candidate["id"],
        "scalar_overrides": variant["scalar_overrides"],
        "metrics": aggregate,
        "rank_score": 0.0,
        "summary_label": "v9-5ch release scalar probe",
        "run_count": len(runs),
    }
    return summary, records


def release_probe_summary(runs: list[dict[str, Any]]) -> dict[str, float]:
    metrics = [run["metrics"] for run in runs]
    release_steps = [
        row
        for run in runs
        for row in run["trajectory"]
        if float(row.get("route_metrics", {}).get("release_strength_mean", 0.0)) > 0.0
    ]
    return {
        "release_strength_mean": mean_or_zero(m["release_strength_mean"] for m in metrics),
        "release_transfer_mean": mean_or_zero(m["release_transfer"] for m in metrics),
        "release_transfer_abs_mean": mean_or_zero(abs(m["release_transfer"]) for m in metrics),
        "release_step_count": float(len(release_steps)),
        "release_bias_norm_mean": mean_or_zero(
            row.get("route_metrics", {}).get("release_bias_norm", 0.0)
            for row in release_steps
        ),
    }


def mean_or_zero(values: Any) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def parse_float_list(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return default
    parsed = []
    for value in values:
        parsed.extend(float(part) for part in value.split(",") if part)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe v9-5ch release threshold/temperature/gain sensitivity")
    parser.add_argument("--archive-dir", default="data/evolution/v9_5ch_release_20260509_full")
    parser.add_argument("--source", default="archive.json")
    parser.add_argument("--candidate-id")
    parser.add_argument("--threshold-delta", action="append")
    parser.add_argument("--temperature-scale", action="append")
    parser.add_argument("--gain-scale", action="append")
    parser.add_argument("--seed-limit", type=int, default=1)
    parser.add_argument("--motif-limit", type=int, default=4)
    parser.add_argument("--scale-limit", type=int, default=1)
    parser.add_argument("--out-dir", default="data/substrate_lab/v9_5ch_release_param_probe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_dir = Path(args.archive_dir)
    config = load_json(archive_dir / "config.json")
    candidate = select_candidate(load_archive_rows(archive_dir / args.source), args.candidate_id)
    base_scalars = candidate["genome"]["scalars"]
    variants = scalar_variants(
        base_scalars,
        parse_float_list(args.threshold_delta, [-0.14, 0.0, 0.14]),
        parse_float_list(args.temperature_scale, [0.7, 1.0, 1.35]),
        parse_float_list(args.gain_scale, [0.65, 1.0, 1.45]),
    )
    summaries = []
    records = []
    for variant in variants:
        summary, variant_records = run_variant(candidate, variant, config, args)
        summaries.append(summary)
        records.extend(variant_records)

    summaries.sort(key=lambda row: float(row["metrics"].get("release_effectiveness", 0.0)), reverse=True)
    payload = export_payload(records, config, summaries)
    payload["metadata"]["viewer_note"] = "release parameter probe; variants clone one archived candidate and alter release scalars only"
    payload["metadata"]["source_candidate_id"] = candidate["id"]
    payload["metadata"]["probe_axes"] = ["release_threshold", "release_temperature", "release_gain"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "source_candidate": {
            "id": candidate["id"],
            "rank_score": candidate.get("rank_score", 0.0),
            "metrics": candidate.get("metrics", {}),
            "release_scalars": {
                key: base_scalars[key]
                for key in ("release_threshold", "release_temperature", "release_gain")
            },
        },
        "variant_count": len(summaries),
        "run_shape": {
            "seed_limit": args.seed_limit,
            "motif_limit": args.motif_limit,
            "scale_limit": args.scale_limit,
        },
        "variants": summaries,
    }, indent=2, allow_nan=False) + "\n")
    (out_dir / "trajectory_3d.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(out_dir / "summary.json")
    print(out_dir / "trajectory_3d.json")


if __name__ == "__main__":
    main()
