#!/usr/bin/env python3
"""Held-out cross-evaluation for v10.0 frozen evolution candidates."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolution.scoring import rank_components, scalar_rank
from development.evolve_v9_5ch_release import (
    aggregate_runs,
    motif_suite,
    paired_causal_run,
    run_with_motif,
)

DEFAULT_CANDIDATES = (
    "island_1:data/evolution/v10_0_frozen_evolution_island_1_20260510:gen018_candidate003",
    "island_2:data/evolution/v10_0_frozen_evolution_island_2_20260510:gen017_candidate003",
    "island_3:data/evolution/v10_0_frozen_evolution_island_3_20260510:gen018_candidate002",
    "island_4:data/evolution/v10_0_frozen_evolution_island_4_20260510:gen010_candidate006",
)

SUMMARY_FIELDS = (
    "label",
    "island",
    "candidate_id",
    "ablation",
    "run_count",
    "rank_score",
    "internal_richness",
    "channel_separation",
    "release_duty_cycle",
    "release_strength_mean",
    "release_geometric_event",
    "release_causal_divergence",
    "release_causal_divergence_fast",
    "release_causal_divergence_slow",
    "release_causal_divergence_control",
    "release_causal_divergence_message",
    "release_causal_divergence_carrier",
    "phase_transition_score",
    "release_local_displacement",
    "release_local_delta_lift",
    "release_pre_event_pressure",
    "pressure_at_release_mean",
    "geometric_coherence",
    "mathematical_curiosity",
    "boundedness",
    "regime_bonus",
)

RUN_FIELDS = (
    "label",
    "island",
    "candidate_id",
    "ablation",
    "seed",
    "perturb_scale",
    "perturb_channel",
    "perturb_mode",
    "motif_family",
    "motif_index",
    "regime_class",
    "release_duty_cycle",
    "release_strength_mean",
    "release_geometric_event",
    "release_causal_divergence",
    "phase_transition_score",
    "release_local_displacement",
    "release_local_delta_lift",
    "channel_separation",
    "internal_richness",
    "geometric_coherence",
    "boundedness",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_candidate_spec(spec: str) -> dict[str, Any]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"candidate spec must be label:archive_dir:candidate_id, got {spec!r}")
    label, archive_dir, candidate_id = parts
    return {
        "label": label,
        "archive_dir": Path(archive_dir),
        "candidate_id": candidate_id,
    }


def find_candidate(archive_dir: Path, candidate_id: str) -> dict[str, Any]:
    archive = load_json(archive_dir / "archive.json")
    rows = list(archive.values()) if isinstance(archive, dict) else list(archive)
    for row in rows:
        if row.get("id") == candidate_id:
            return row
    raise KeyError(f"{candidate_id} not found in {archive_dir / 'archive.json'}")


def ablated_genome(genome: dict[str, Any], ablation: str) -> dict[str, Any]:
    clone = copy.deepcopy(genome)
    scalars = clone.setdefault("scalars", {})
    if ablation == "original":
        return clone
    if ablation == "release_gain_zero":
        scalars["release_gain"] = 0.0
        return clone
    if ablation == "release_routes_disabled":
        return clone
    raise ValueError(f"unknown ablation: {ablation}")


def candidate_runs(
    candidate: dict[str, Any],
    config: dict[str, Any],
    *,
    ablation: str,
    seeds: list[int],
    perturb_scales: list[float],
    perturb_channels: list[str],
    perturb_modes: list[str],
    motif_limit: int,
    device: str,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    genome = ablated_genome(candidate["genome"], ablation)
    for seed in seeds:
        motifs = motif_suite(int(config["hidden_size"]), seed)[:motif_limit]
        for scale in perturb_scales:
            for channel in perturb_channels:
                for mode in perturb_modes:
                    for motif in motifs:
                        if ablation == "release_routes_disabled":
                            run = run_with_motif(
                                genome,
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
                                release_routes_disabled=True,
                            )
                        elif ablation == "original":
                            paired = paired_causal_run(
                                genome,
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
                            run = next(item for item in paired["runs"] if item["ablation"] == "original")
                        else:
                            run = run_with_motif(
                                genome,
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
                        run.pop("trajectory", None)
                        runs.append(run)
    return runs


def summarize_runs(
    *,
    label: str,
    island: str,
    candidate_id: str,
    ablation: str,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = aggregate_runs(runs)
    if "release_strength_mean" not in aggregate:
        aggregate["release_strength_mean"] = mean(
            float(run["metrics"].get("release_strength_mean", 0.0)) for run in runs
        )
    row = {"metrics": aggregate}
    return {
        "label": label,
        "island": island,
        "candidate_id": candidate_id,
        "ablation": ablation,
        "run_count": len(runs),
        "rank_score": scalar_rank(row),
        "rank_components": rank_components(row),
        "metrics": aggregate,
    }


def flatten_summary(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    out = {field: row.get(field, metrics.get(field, "")) for field in SUMMARY_FIELDS}
    return out


def flatten_run(
    *,
    label: str,
    island: str,
    candidate_id: str,
    ablation: str,
    run: dict[str, Any],
) -> dict[str, Any]:
    metrics = run["metrics"]
    summary = run["summary"]
    base = {
        "label": label,
        "island": island,
        "candidate_id": candidate_id,
        "ablation": ablation,
        "seed": run["seed"],
        "perturb_scale": run["perturb_scale"],
        "perturb_channel": run.get("perturb_channel", "fast"),
        "perturb_mode": run.get("perturb_mode", "external"),
        "motif_family": run["motif_family"],
        "motif_index": run["motif_index"],
        "regime_class": summary["regime_class"],
    }
    for field in RUN_FIELDS:
        base.setdefault(field, metrics.get(field, ""))
    return base


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    candidate_specs = [parse_candidate_spec(spec) for spec in args.candidates]
    summaries: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for spec in candidate_specs:
        config = load_json(spec["archive_dir"] / "config.json")
        candidate = find_candidate(spec["archive_dir"], spec["candidate_id"])
        candidates.append(
            {
                "label": spec["label"],
                "archive_dir": str(spec["archive_dir"]),
                "candidate_id": candidate["id"],
                "source_rank_score": candidate.get("rank_score"),
                "source_metrics": candidate.get("metrics", {}),
            }
        )
        for ablation in args.ablations:
            runs = candidate_runs(
                candidate,
                config,
                ablation=ablation,
                seeds=args.seeds,
                perturb_scales=args.perturb_scales,
                perturb_channels=args.perturb_channels,
                perturb_modes=args.perturb_modes,
                motif_limit=args.motif_limit,
                device=args.device,
            )
            summaries.append(
                summarize_runs(
                    label=spec["label"],
                    island=spec["label"],
                    candidate_id=candidate["id"],
                    ablation=ablation,
                    runs=runs,
                )
            )
            run_rows.extend(
                flatten_run(
                    label=spec["label"],
                    island=spec["label"],
                    candidate_id=candidate["id"],
                    ablation=ablation,
                    run=run,
                )
                for run in runs
            )
            print(
                "{label} {candidate} {ablation}: runs={runs} rank={rank:.4f} duty={duty:.4f} event={event:.4f}".format(
                    label=spec["label"],
                    candidate=candidate["id"],
                    ablation=ablation,
                    runs=len(runs),
                    rank=summaries[-1]["rank_score"],
                    duty=summaries[-1]["metrics"]["release_duty_cycle"],
                    event=summaries[-1]["metrics"]["release_geometric_event"],
                )
            )
    return {
        "experiment": "v10.0-frozen-heldout-cross-eval",
        "source_experiment": "v10.0-frozen-evolution",
        "device": args.device,
        "seeds": args.seeds,
        "perturb_scales": args.perturb_scales,
        "perturb_channels": args.perturb_channels,
        "perturb_modes": args.perturb_modes,
        "motif_limit": args.motif_limit,
        "ablations": args.ablations,
        "candidates": candidates,
        "summaries": summaries,
        "run_rows": run_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/substrate_lab/v10_frozen_cross_eval_20260511")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("95,96,97,98,99,100,101,102"))
    parser.add_argument("--perturb-scales", type=parse_csv_floats, default=parse_csv_floats("0.25,0.35,0.50"))
    parser.add_argument("--perturb-channels", type=parse_csv_strings, default=parse_csv_strings("fast,carrier,message"))
    parser.add_argument("--perturb-modes", type=parse_csv_strings, default=parse_csv_strings("external"))
    parser.add_argument("--motif-limit", type=int, default=2)
    parser.add_argument("--ablations", type=parse_csv_strings, default=parse_csv_strings("original,release_routes_disabled,release_gain_zero"))
    parser.add_argument("--candidates", nargs="*", default=list(DEFAULT_CANDIDATES))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    summary_rows = [flatten_summary(row) for row in payload["summaries"]]
    write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(out_dir / "runs.csv", RUN_FIELDS, payload["run_rows"])
    json_payload = dict(payload)
    json_payload.pop("run_rows")
    (out_dir / "summary.json").write_text(json.dumps(json_payload, indent=2, sort_keys=True))
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'runs.csv'}")


if __name__ == "__main__":
    main()
