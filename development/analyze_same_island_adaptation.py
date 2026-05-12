#!/usr/bin/env python3
"""Analyze same-island Track B phenotype adaptation from archives and confirmations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolution.scoring import scalar_rank


DEFAULT_ARCHIVE_ROOT = Path(
    "data/evolution/track_b_discovery_20260513/track_b_rank_landscape_sweep_combined_discovery_2"
)
DEFAULT_DIAGNOSTICS_ROOTS = (
    Path("data/diagnostics/track_b_same_island_adaptation_20260513"),
    Path("data/diagnostics/track_b_discovery_20260513"),
)
DEFAULT_OUTPUT_CSV = Path("data/diagnostics/track_b_same_island_adaptation_20260513/same_island_table.csv")
DEFAULT_ANCESTORS = ("gen000_candidate005", "gen000_candidate000")
DEFAULT_REFERENCE_CANDIDATE = "gen008_candidate015"
DEFAULT_INITIAL_CONFIRMATION_IDS = (
    "gen007_candidate010",
    "gen008_candidate015",
    "gen012_candidate013",
    "gen010_candidate013",
    "gen009_candidate009",
    "gen010_candidate012",
    "gen015_candidate002",
    "gen014_candidate001",
)
DEFAULT_CROSS_ISLAND_CANDIDATES = (
    Path(
        "data/evolution/track_b_discovery_20260513/"
        "track_b_rank_landscape_sweep_combined_discovery_1/candidates/gen012_candidate006.json"
    ),
)
CHANNELS = ("fast", "slow", "control", "message", "carrier")
CSV_FIELDS = (
    "archive_id",
    "candidate_id",
    "island",
    "generation",
    "parent_ids",
    "ancestor_ids",
    "scalar_distance",
    "rank_score",
    "release_geometric_event",
    "phase_transition_score",
    "release_duty_cycle",
    "discovery_dominant_regime",
    "discovery_regimes",
    "summary_path",
    "confirmation_phenotype",
    "best_resume_channel",
    "channel_necessity_order",
    "heldout_dominant_regime",
    "carrier_family_accuracy",
    "carrier_step_score",
    "phenotypes_by_step",
    "adaptation_label",
    "candidate_path",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def generation_from_id(candidate_id: str) -> int:
    if candidate_id.startswith("gen") and "_candidate" in candidate_id:
        return int(candidate_id[3:6])
    return -1


def dominant_regime(regimes: dict[str, Any]) -> str:
    if regimes:
        return str(max(regimes, key=regimes.get))
    return "unknown"


def scalar_signature(candidate: dict[str, Any]) -> dict[str, float]:
    scalars = candidate.get("genome", {}).get("scalars", {})
    return {str(key): float(value) for key, value in scalars.items()}


def scalar_distance(candidate: dict[str, Any], reference: dict[str, Any]) -> float:
    left = scalar_signature(candidate)
    right = scalar_signature(reference)
    common = sorted(set(left) & set(right))
    if not common:
        return math.inf
    return math.sqrt(sum((left[key] - right[key]) ** 2 for key in common))


def candidate_record(path: Path, archive_root: Path, *, island: str, reference: dict[str, Any]) -> dict[str, Any]:
    candidate = load_json(path)
    metrics = candidate.get("metrics", {})
    regimes = metrics.get("regimes", {})
    candidate_id = str(candidate.get("id", path.stem))
    return {
        "archive_id": archive_root.name,
        "candidate_id": candidate_id,
        "candidate": candidate,
        "candidate_path": path,
        "island": island,
        "generation": generation_from_id(candidate_id),
        "parent_ids": [str(item) for item in candidate.get("parent_ids", [])],
        "ancestor_ids": [str(item) for item in candidate.get("ancestor_ids", [])],
        "scalar_distance": scalar_distance(candidate, reference),
        "rank_score": scalar_rank(candidate),
        "release_geometric_event": float(metrics.get("release_geometric_event", 0.0)),
        "phase_transition_score": float(metrics.get("phase_transition_score", 0.0)),
        "release_duty_cycle": float(metrics.get("release_duty_cycle", 0.0)),
        "discovery_dominant_regime": dominant_regime(regimes),
        "discovery_regimes": regimes,
    }


def load_archive_candidates(
    archive_root: Path,
    *,
    ancestors: set[str],
    reference_candidate_id: str,
) -> list[dict[str, Any]]:
    reference_path = archive_root / "candidates" / f"{reference_candidate_id}.json"
    reference = load_json(reference_path)
    records = []
    for path in sorted((archive_root / "candidates").glob("*.json")):
        candidate = load_json(path)
        candidate_ancestors = {str(item) for item in candidate.get("ancestor_ids", [])}
        if not candidate_ancestors & ancestors and str(candidate.get("id", path.stem)) != reference_candidate_id:
            continue
        records.append(candidate_record(path, archive_root, island=archive_root.name, reference=reference))
    return records


def load_extra_candidates(paths: Iterable[Path], *, reference: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        archive_root = path.parent.parent
        records.append(candidate_record(path, archive_root, island=archive_root.name, reference=reference))
    return records


def summary_matches_candidate(summary: dict[str, Any], candidate_path: Path) -> bool:
    config = summary.get("config", {})
    recorded = config.get("candidate_path")
    if not recorded:
        return True
    return Path(str(recorded)).as_posix() == candidate_path.as_posix()


def summary_paths_for(record: dict[str, Any], diagnostics_roots: list[Path]) -> list[Path]:
    archive_id = str(record["archive_id"])
    candidate_id = str(record["candidate_id"])
    paths = []
    for root in diagnostics_roots:
        paths.extend(sorted((root / archive_id / candidate_id).glob("perturb_step_*/summary.json")))
        paths.extend(sorted((root / candidate_id).glob("perturb_step_*/summary.json")))
    return paths


def perturb_step_from_summary_path(path: Path) -> int:
    label = path.parent.name
    if label.startswith("perturb_step_"):
        return int(label.removeprefix("perturb_step_"))
    return -1


def load_matching_summaries(record: dict[str, Any], diagnostics_roots: list[Path]) -> dict[int, tuple[Path, dict[str, Any]]]:
    summaries = {}
    for path in summary_paths_for(record, diagnostics_roots):
        summary = load_json(path)
        if not summary_matches_candidate(summary, Path(record["candidate_path"])):
            continue
        summaries[perturb_step_from_summary_path(path)] = (path, summary)
    return summaries


def row_by_key(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get(key, "")) == value:
            return row
    return None


def probe_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    probes = summary.get("probe_summary", [])
    if isinstance(probes, dict):
        probes = probes.get("probes", [])
    return [row for row in probes if isinstance(row, dict)]


def carrier_probe_scores(summary: dict[str, Any]) -> tuple[float, float]:
    family = 0.0
    step = 0.0
    for row in probe_rows(summary):
        if str(row.get("channel")) != "carrier":
            continue
        target = str(row.get("target"))
        metric = str(row.get("metric"))
        score = float(row.get("score", 0.0))
        if target == "family" and metric == "accuracy":
            family = max(family, score)
        if target == "step":
            step = max(step, score)
    return family, step


def heldout_dominant_regime(summary: dict[str, Any]) -> str:
    regimes = Counter(
        str(row.get("dominant_regime"))
        for row in summary.get("ablation_summary", [])
        if row.get("dominant_regime")
    )
    if not regimes:
        return "unknown"
    return regimes.most_common(1)[0][0]


def bounded_strange_confirmed(summary: dict[str, Any]) -> bool:
    for row in summary.get("ablation_summary", []):
        if row.get("dominant_regime") == "bounded_strange" and float(row.get("boundedness_mean", 0.0)) >= 0.95:
            return True
    return False


def classify_summary(summary: dict[str, Any]) -> dict[str, Any]:
    flags = summary.get("capsule_result_flags", {})
    best_resume = str(flags.get("best_single_channel_resume", ""))
    necessity_order = [str(item) for item in summary.get("channel_necessity_order", [])]
    carrier_family, carrier_step = carrier_probe_scores(summary)
    dominant = heldout_dominant_regime(summary)
    phenotypes = []
    if best_resume == "message" and necessity_order[:1] == ["message"]:
        phenotypes.append("message_capsule")
    if best_resume != "carrier" and (carrier_family >= 0.85 or carrier_step >= 0.90):
        phenotypes.append("carrier_support")
    if dominant == "surface_fixed_accumulating":
        phenotypes.append("surface_accumulator")
    if bounded_strange_confirmed(summary):
        phenotypes.append("unstable_strange")
    return {
        "phenotypes": phenotypes,
        "best_resume_channel": best_resume,
        "channel_necessity_order": necessity_order,
        "heldout_dominant_regime": dominant,
        "carrier_family_accuracy": carrier_family,
        "carrier_step_score": carrier_step,
    }


def adaptation_label(classifications_by_step: dict[int, dict[str, Any]]) -> str:
    if not classifications_by_step:
        return "unconfirmed"
    labels = {
        tuple(classifications_by_step[step]["phenotypes"])
        for step in sorted(classifications_by_step)
    }
    if len(labels) > 1:
        return "dynamic_adaptation_candidate"
    return "stable_confirmed_phenotype"


def candidate_output_dir(record: dict[str, Any], diagnostics_root: Path, perturb_step: int) -> Path:
    return (
        diagnostics_root
        / str(record["archive_id"])
        / str(record["candidate_id"])
        / f"perturb_step_{perturb_step:03d}"
    )


def confirmation_command(record: dict[str, Any], diagnostics_root: Path, perturb_step: int) -> list[str]:
    return [
        "venv/bin/python",
        "development/gate_state_propagation_characterization.py",
        "--candidate-path",
        Path(record["candidate_path"]).as_posix(),
        "--archive-root",
        Path(record["candidate_path"]).parent.parent.as_posix(),
        "--output-dir",
        candidate_output_dir(record, diagnostics_root, perturb_step).as_posix(),
        "--seeds",
        "96,97,98,99,100,101,102",
        "--perturb-scales",
        "0.2,0.35,0.7",
        "--perturb-step",
        str(perturb_step),
        "--capsule-pause-step",
        str(perturb_step),
        "--device",
        "cpu",
    ]


def enrich_record(record: dict[str, Any], diagnostics_roots: list[Path]) -> dict[str, Any]:
    summaries = load_matching_summaries(record, diagnostics_roots)
    classifications = {
        step: classify_summary(summary)
        for step, (_, summary) in sorted(summaries.items())
    }
    step64 = classifications.get(64)
    step64_path = summaries.get(64, ("", {}))[0] if 64 in summaries else ""
    phenotype = step64 or (classifications[sorted(classifications)[0]] if classifications else {})
    return {
        **record,
        "summary_path": Path(step64_path).as_posix() if step64_path else "",
        "confirmation_phenotype": "|".join(phenotype.get("phenotypes", [])),
        "best_resume_channel": phenotype.get("best_resume_channel", ""),
        "channel_necessity_order": ",".join(phenotype.get("channel_necessity_order", [])),
        "heldout_dominant_regime": phenotype.get("heldout_dominant_regime", ""),
        "carrier_family_accuracy": phenotype.get("carrier_family_accuracy", ""),
        "carrier_step_score": phenotype.get("carrier_step_score", ""),
        "phenotypes_by_step": json.dumps(
            {step: data["phenotypes"] for step, data in sorted(classifications.items())},
            sort_keys=True,
        ),
        "adaptation_label": adaptation_label(classifications),
    }


def format_csv_row(record: dict[str, Any]) -> dict[str, str]:
    return {
        "archive_id": str(record["archive_id"]),
        "candidate_id": str(record["candidate_id"]),
        "island": str(record["island"]),
        "generation": str(record["generation"]),
        "parent_ids": ",".join(record["parent_ids"]),
        "ancestor_ids": ",".join(record["ancestor_ids"]),
        "scalar_distance": f"{float(record['scalar_distance']):.9g}",
        "rank_score": f"{float(record['rank_score']):.9g}",
        "release_geometric_event": f"{float(record['release_geometric_event']):.9g}",
        "phase_transition_score": f"{float(record['phase_transition_score']):.9g}",
        "release_duty_cycle": f"{float(record['release_duty_cycle']):.9g}",
        "discovery_dominant_regime": str(record["discovery_dominant_regime"]),
        "discovery_regimes": json.dumps(record["discovery_regimes"], sort_keys=True),
        "summary_path": str(record["summary_path"]),
        "confirmation_phenotype": str(record["confirmation_phenotype"]),
        "best_resume_channel": str(record["best_resume_channel"]),
        "channel_necessity_order": str(record["channel_necessity_order"]),
        "heldout_dominant_regime": str(record["heldout_dominant_regime"]),
        "carrier_family_accuracy": str(record["carrier_family_accuracy"]),
        "carrier_step_score": str(record["carrier_step_score"]),
        "phenotypes_by_step": str(record["phenotypes_by_step"]),
        "adaptation_label": str(record["adaptation_label"]),
        "candidate_path": Path(record["candidate_path"]).as_posix(),
    }


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_csv_row(row))


def select_initial_records(rows: list[dict[str, Any]], candidate_ids: set[str]) -> list[dict[str, Any]]:
    by_id = {str(row["candidate_id"]): row for row in rows}
    return [by_id[candidate_id] for candidate_id in sorted(candidate_ids) if candidate_id in by_id]


def build_rows(
    archive_root: Path,
    *,
    ancestors: set[str],
    reference_candidate_id: str,
    diagnostics_roots: list[Path],
    extra_candidate_paths: list[Path],
) -> list[dict[str, Any]]:
    records = load_archive_candidates(
        archive_root,
        ancestors=ancestors,
        reference_candidate_id=reference_candidate_id,
    )
    reference = load_json(archive_root / "candidates" / f"{reference_candidate_id}.json")
    records.extend(load_extra_candidates(extra_candidate_paths, reference=reference))
    enriched = [enrich_record(record, diagnostics_roots) for record in records]
    return sorted(
        enriched,
        key=lambda row: (
            str(row["archive_id"]) != archive_root.name,
            -float(row["rank_score"]),
            float(row["scalar_distance"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--diagnostics-root", action="append", type=Path)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--reference-candidate", default=DEFAULT_REFERENCE_CANDIDATE)
    parser.add_argument("--ancestors", default=",".join(DEFAULT_ANCESTORS))
    parser.add_argument("--extra-candidate", action="append", type=Path)
    parser.add_argument("--initial-candidate-ids", default=",".join(DEFAULT_INITIAL_CONFIRMATION_IDS))
    parser.add_argument("--same-island-diagnostics-root", type=Path, default=DEFAULT_DIAGNOSTICS_ROOTS[0])
    parser.add_argument("--emit-missing-commands", action="store_true")
    parser.add_argument("--perturb-steps", default="64")
    args = parser.parse_args()

    diagnostics_roots = args.diagnostics_root or list(DEFAULT_DIAGNOSTICS_ROOTS)
    extra_candidates = args.extra_candidate or list(DEFAULT_CROSS_ISLAND_CANDIDATES)
    rows = build_rows(
        args.archive_root,
        ancestors=set(parse_csv_list(args.ancestors)),
        reference_candidate_id=args.reference_candidate,
        diagnostics_roots=diagnostics_roots,
        extra_candidate_paths=extra_candidates,
    )
    write_table(args.output_csv, rows)
    print(f"wrote {args.output_csv}")
    print(f"rows={len(rows)}")

    if args.emit_missing_commands:
        selected = select_initial_records(rows, set(parse_csv_list(args.initial_candidate_ids)))
        perturb_steps = [int(step) for step in parse_csv_list(args.perturb_steps)]
        for record in selected:
            existing = load_matching_summaries(record, diagnostics_roots)
            for perturb_step in perturb_steps:
                if perturb_step in existing:
                    continue
                print(" ".join(confirmation_command(record, args.same_island_diagnostics_root, perturb_step)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
