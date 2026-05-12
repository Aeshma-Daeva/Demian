#!/usr/bin/env python3
"""Paired causal-release diagnostics for Demian v1 release routing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from statistics import median
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolution.scoring import rank_components, scalar_rank
from development.evolve_v9_5ch_release import (
    CHANNELS,
    default_genome,
    motif_suite,
    paired_causal_run,
    random_genome,
)

V93_ARCHIVES = tuple(
    Path(f"data/evolution/v9_3_cap_calibrated_island_{idx}_20260510")
    for idx in range(1, 5)
)
V100_ARCHIVES = tuple(
    Path(f"data/evolution/v10_0_frozen_evolution_island_{idx}_20260510")
    for idx in range(1, 5)
)

CAUSAL_FIELDS = (
    "release_causal_divergence",
    "release_causal_anchor_count",
    *(f"release_causal_divergence_{channel}" for channel in CHANNELS),
)
CSV_FIELDS = (
    "label",
    "candidate_id",
    "ablation",
    "seed",
    "perturb_scale",
    "perturb_channel",
    "perturb_mode",
    "motif_family",
    "motif_index",
    "rank_score",
    "release_duty_cycle",
    "release_strength_mean",
    "release_geometric_event",
    *CAUSAL_FIELDS,
    "phase_transition_score",
    "internal_richness",
    "channel_separation",
    "mathematical_curiosity",
    "geometric_coherence",
    "regime_bonus",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_archive_rows(archive_dir: Path) -> list[dict[str, Any]]:
    archive = load_json(archive_dir / "archive.json")
    rows = list(archive.values()) if isinstance(archive, dict) else list(archive)
    return sorted(rows, key=lambda row: float(row.get("rank_score", 0.0)), reverse=True)


def archive_candidates(archive_dirs: Iterable[Path], top_n: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for archive_dir in archive_dirs:
        config = load_json(archive_dir / "config.json")
        for row in load_archive_rows(archive_dir)[:top_n]:
            candidates.append(
                {
                    "label": archive_dir.name,
                    "candidate_id": row["id"],
                    "genome": row["genome"],
                    "config": config,
                }
            )
    return candidates


def run_rows_for_candidate(
    candidate: dict[str, Any],
    *,
    seeds: list[int],
    motif_limit: int,
    perturb_scales: list[float],
    perturb_channels: list[str],
    perturb_modes: list[str],
    device: str,
) -> list[dict[str, Any]]:
    config = candidate["config"]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        motifs = motif_suite(int(config["hidden_size"]), seed)[:motif_limit]
        for scale in perturb_scales:
            for channel in perturb_channels:
                for mode in perturb_modes:
                    for motif in motifs:
                        paired = paired_causal_run(
                            candidate["genome"],
                            hidden_size=int(config["hidden_size"]),
                            steps=int(config["steps"]),
                            seed=seed,
                            perturb_step=int(config["perturb_step"]),
                            perturb_scale=scale,
                            motif=motif,
                            rank=int(config["rank"]),
                            device=device,
                            perturb_channel=channel,
                            perturb_mode=mode,
                        )
                        for run in paired["runs"]:
                            rows.append(flatten_run(candidate, run))
    return rows


def flatten_run(candidate: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    metrics = run["metrics"]
    rank_metrics = {
        "internal_richness": metrics.get("internal_richness", 0.0),
        "channel_separation": metrics.get("channel_separation", 0.0),
        "release_causal_divergence": metrics.get("release_causal_divergence", 0.0),
        "phase_transition_score": metrics.get("phase_transition_score", 0.0),
        "mathematical_curiosity": metrics.get("mathematical_curiosity", 0.0),
        "geometric_coherence": metrics.get("geometric_coherence", 0.0),
        "regime_bonus": metrics.get("regime_bonus", 0.0),
        "release_duty_cycle": metrics.get("release_duty_cycle", 0.0),
    }
    row = {
        "label": candidate["label"],
        "candidate_id": candidate["candidate_id"],
        "ablation": run["ablation"],
        "seed": run["seed"],
        "perturb_scale": run["perturb_scale"],
        "perturb_channel": run.get("perturb_channel", "fast"),
        "perturb_mode": run.get("perturb_mode", "external"),
        "motif_family": run["motif_family"],
        "motif_index": run["motif_index"],
        "rank_score": scalar_rank({"metrics": rank_metrics}),
    }
    row.update({field: metrics.get(field, "") for field in CSV_FIELDS if field not in row})
    row["rank_components"] = json.dumps(rank_components({"metrics": rank_metrics}), sort_keys=True)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (*CSV_FIELDS, "rank_components")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_archive_audit_csv(
    path: Path,
    candidates: list[dict[str, Any]],
    *,
    seeds: list[int],
    motif_limit: int,
    perturb_scales: list[float],
    perturb_channels: list[str],
    perturb_modes: list[str],
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.extend(
            run_rows_for_candidate(
                candidate,
                seeds=seeds,
                motif_limit=motif_limit,
                perturb_scales=perturb_scales,
                perturb_channels=perturb_channels,
                perturb_modes=perturb_modes,
                device=device,
            )
        )
    write_csv(path, rows)
    return rows


def sampled_manual_candidates(
    *,
    count: int,
    hidden_size: int,
    rank: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    route_mixes = (
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (0.2, 0.2, 0.2, 0.2, 0.2),
        (0.0, 0.3, 0.2, 0.3, 0.2),
        (0.1, 0.1, 0.1, 0.35, 0.35),
        (0.15, 0.35, 0.2, 0.15, 0.15),
    )
    candidates: list[dict[str, Any]] = []
    for idx in range(count):
        genome = random_genome(rng, hidden_size, rank) if idx else default_genome(hidden_size, rank)
        scalars = genome["scalars"]
        scalars["release_gain"] = rng.uniform(0.04, 0.46)
        scalars["release_threshold"] = rng.uniform(-0.40, 0.55)
        scalars["release_temperature"] = rng.uniform(0.35, 1.80)
        mix = rng.choice(route_mixes)
        for channel, value in zip(CHANNELS, mix):
            scalars[f"release_to_{channel}_scale"] = value
        candidates.append(
            {
                "label": "manual_causal_sweep",
                "candidate_id": f"manual_{idx:03d}",
                "genome": genome,
                "config": {
                    "hidden_size": hidden_size,
                    "steps": 128,
                    "perturb_step": 64,
                    "rank": rank,
                },
            }
        )
    return candidates


def paired_no_release_drift(rows: list[dict[str, Any]]) -> float:
    values = [
        float(row.get("release_causal_divergence", 0.0))
        for row in rows
        if row.get("ablation") == "original"
        and float(row.get("release_duty_cycle", 0.0) or 0.0) <= 0.0
    ]
    return float(median(values)) if values else 0.0


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("archive-audit")
    audit.add_argument("--out", default="data/diagnostics/causal_release_audit_20260511.csv")
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--top-n", type=int, default=1)
    audit.add_argument("--seeds", type=parse_ints, default=parse_ints("94,95,96,97,98,99,100,101,102"))
    audit.add_argument("--motif-limit", type=int, default=1)
    audit.add_argument("--perturb-scales", type=parse_floats, default=parse_floats("0.35"))
    audit.add_argument("--perturb-channels", type=parse_strings, default=parse_strings("fast"))
    audit.add_argument("--perturb-modes", type=parse_strings, default=parse_strings("external"))

    sweep = sub.add_parser("manual-sweep")
    sweep.add_argument("--out", default="data/diagnostics/manual_causal_sweep_20260511.csv")
    sweep.add_argument("--device", default="cpu")
    sweep.add_argument("--count", type=int, default=50)
    sweep.add_argument("--sample-seed", type=int, default=20260511)
    sweep.add_argument("--hidden-size", type=int, default=32)
    sweep.add_argument("--rank", type=int, default=2)
    sweep.add_argument("--seeds", type=parse_ints, default=parse_ints("94"))
    sweep.add_argument("--motif-limit", type=int, default=1)
    sweep.add_argument("--perturb-scales", type=parse_floats, default=parse_floats("0.35"))
    sweep.add_argument("--perturb-channels", type=parse_strings, default=parse_strings("fast"))
    sweep.add_argument("--perturb-modes", type=parse_strings, default=parse_strings("external"))
    args = parser.parse_args()

    if args.command == "archive-audit":
        candidates = archive_candidates((*V93_ARCHIVES, *V100_ARCHIVES), args.top_n)
    else:
        candidates = sampled_manual_candidates(
            count=args.count,
            hidden_size=args.hidden_size,
            rank=args.rank,
            seed=args.sample_seed,
        )
    rows = write_archive_audit_csv(
        Path(args.out),
        candidates,
        seeds=args.seeds,
        motif_limit=args.motif_limit,
        perturb_scales=args.perturb_scales,
        perturb_channels=args.perturb_channels,
        perturb_modes=args.perturb_modes,
        device=args.device,
    )
    drift = paired_no_release_drift(rows)
    threshold = max(0.01, 2.0 * drift)
    print(f"wrote {args.out}")
    print(f"paired_no_release_drift_median={drift:.8f}")
    print(f"initial_divergence_threshold={threshold:.8f}")


if __name__ == "__main__":
    main()
