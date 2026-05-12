#!/usr/bin/env python3
"""Build, select, and optionally run the Track B discovery campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolution.scoring import (
    CAUSAL_MODE_GATE_STATE,
    NATIVE_EMERGENCE_RANK_MODE,
    NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
    NATIVE_OBJECTIVE_GATE_STATE,
    NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY,
    NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
    scalar_rank,
)


DISCOVERY_LABELS = (
    "track_b_seed_motif_timing_sweep",
    "track_b_rank_landscape_sweep",
    "track_b_long_horizon_sweep",
    "track_b_gate_state_probe",
)
RANK_LANDSCAPE_OBJECTIVES = (
    NATIVE_OBJECTIVE_GATE_STATE,
    NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
    NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY,
    NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
)
CONFIRMATION_OBJECTIVE_QUOTAS = {
    NATIVE_OBJECTIVE_GATE_STATE: 2,
    NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY: 2,
    NATIVE_OBJECTIVE_MORPHOLOGY_ONLY: 2,
    NATIVE_OBJECTIVE_COMBINED_DISCOVERY: 3,
}
CONFIRMATION_REGIME_QUOTAS = {
    "bounded_strange": 2,
    "surface_fixed_accumulating": 2,
}
CONFIRMATION_TARGET_COUNT = 12
PRESETS = ("smoke", "efficient-discovery", "confirmation", "long-horizon", "full-original")
DEFAULT_WORKERS = 6
DEFAULT_TORCH_THREADS = 1
DEFAULT_OUT_ROOT = Path("data/evolution/track_b_discovery_20260513")
DEFAULT_PLAN_PATH = Path("data/evolution/track_b_discovery_20260513_plan.json")
CONFIRMATION_ROOT = Path("data/diagnostics/track_b_discovery_20260513")


@dataclass(frozen=True)
class DiscoveryRun:
    experiment_label: str
    output_id: str
    seed: int
    eval_seeds: tuple[int, ...]
    perturb_step: int
    perturb_channels: tuple[str, ...]
    generations: int
    population: int
    native_objective: str
    perturb_scales: tuple[float, ...] = (0.35,)
    causal_mode: str = CAUSAL_MODE_GATE_STATE
    paired_causal: bool = False
    workers: int = DEFAULT_WORKERS
    torch_threads: int = DEFAULT_TORCH_THREADS
    motif_policy: str = "full"
    selection_policy: str = "native_diversity_funnel"

    def command(self, root: Path) -> list[str]:
        out_dir = root / self.output_id
        command = [
            sys.executable,
            "development/evolve_v9_5ch_release.py",
            "--population",
            str(self.population),
            "--generations",
            str(self.generations),
            "--seed",
            str(self.seed),
            "--eval-seeds",
            ",".join(str(seed) for seed in self.eval_seeds),
            "--steps",
            "128",
            "--perturb-step",
            str(self.perturb_step),
            "--perturb-scales",
            ",".join(format_float(value) for value in self.perturb_scales),
            "--perturb-channels",
            ",".join(self.perturb_channels),
            "--rank-mode",
            NATIVE_EMERGENCE_RANK_MODE,
            "--causal-mode",
            self.causal_mode,
            "--native-objective",
            self.native_objective,
            "--reproduction-mode",
            "native",
            "--no-default-seed",
            "--no-elitism",
            "--no-random-injection",
            "--device",
            "cpu",
            "--workers",
            str(self.workers),
            "--torch-threads",
            str(self.torch_threads),
            "--experiment-version",
            self.experiment_label,
            "--out-dir",
            str(out_dir),
        ]
        if self.paired_causal:
            command.append("--paired-causal")
        return command


def format_float(value: float) -> str:
    return f"{float(value):g}"


def paired_eval_seeds(run_index: int) -> tuple[int, int]:
    values = list(range(94, 102))
    start = (2 * run_index) % len(values)
    return values[start], values[(start + 1) % len(values)]


def smoke_runs() -> list[DiscoveryRun]:
    runs = []
    for idx, objective in enumerate(RANK_LANDSCAPE_OBJECTIVES):
        runs.append(
            DiscoveryRun(
                experiment_label="track_b_seed_motif_timing_sweep",
                output_id=f"track_b_smoke_{objective}",
                seed=2026051301 + idx,
                eval_seeds=(94,),
                perturb_step=64,
                perturb_channels=("fast",),
                generations=1,
                population=8,
                native_objective=objective,
                perturb_scales=(0.35,),
            )
        )
    return runs


def short_diversity_runs(*, efficient: bool = True) -> list[DiscoveryRun]:
    steps = (32, 48, 64, 80)
    channels = ("fast", "message", "carrier", "all")
    objectives = (
        NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
        NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY,
        NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
        NATIVE_OBJECTIVE_GATE_STATE,
    )
    runs = []
    for idx in range(12):
        seed = 2026051301 + idx
        runs.append(
            DiscoveryRun(
                experiment_label="track_b_seed_motif_timing_sweep",
                output_id=f"track_b_seed_motif_timing_sweep_{idx + 1:02d}",
                seed=seed,
                eval_seeds=paired_eval_seeds(idx),
                perturb_step=steps[idx % len(steps)],
                perturb_channels=(channels[idx % len(channels)],),
                generations=12 if efficient else 20,
                population=16,
                native_objective=objectives[idx % len(objectives)] if efficient else NATIVE_OBJECTIVE_GATE_STATE,
                perturb_scales=(0.35,) if efficient else (0.35, 0.7),
                paired_causal=not efficient,
                selection_policy="top_rank_plus_lineage_diversity",
            )
        )
    return runs


def rank_landscape_runs(*, efficient: bool = True) -> list[DiscoveryRun]:
    replicas = 2 if efficient else 3
    generations = 16 if efficient else 20
    runs = []
    for objective in RANK_LANDSCAPE_OBJECTIVES:
        for replica in range(replicas):
            idx = len(runs)
            runs.append(
                DiscoveryRun(
                    experiment_label="track_b_rank_landscape_sweep",
                    output_id=f"track_b_rank_landscape_sweep_{objective}_{replica + 1}",
                    seed=2026051401 + idx,
                    eval_seeds=(94, 95),
                    perturb_step=64,
                    perturb_channels=("fast",),
                    generations=generations,
                    population=16,
                    native_objective=objective,
                    perturb_scales=(0.35, 0.7),
                    paired_causal=not efficient,
                )
            )
    return runs


def long_horizon_runs(
    objectives: tuple[str, ...] = (NATIVE_OBJECTIVE_GATE_STATE, NATIVE_OBJECTIVE_MORPHOLOGY_ONLY),
    *,
    generations: int = 50,
) -> list[DiscoveryRun]:
    runs = []
    for objective in objectives:
        for replica in range(3):
            runs.append(
                DiscoveryRun(
                    experiment_label="track_b_long_horizon_sweep",
                    output_id=f"track_b_long_horizon_sweep_{objective}_g{generations}_{replica + 1}",
                    seed=2026051501 + len(runs),
                    eval_seeds=(94, 95),
                    perturb_step=64,
                    perturb_channels=("fast",),
                    generations=generations,
                    population=16,
                    native_objective=objective,
                    perturb_scales=(0.35, 0.7),
                    paired_causal=False,
                    selection_policy="confirmed_non_duplicate_mechanism",
                )
            )
    return runs


def discovery_matrix(preset: str = "efficient-discovery", *, include_long_horizon: bool = False) -> list[DiscoveryRun]:
    if preset == "smoke":
        return smoke_runs()
    if preset == "efficient-discovery":
        return short_diversity_runs(efficient=True) + rank_landscape_runs(efficient=True)
    if preset == "long-horizon":
        return long_horizon_runs()
    if preset == "full-original":
        runs = short_diversity_runs(efficient=False) + rank_landscape_runs(efficient=False)
        return runs + long_horizon_runs() if include_long_horizon else runs
    if preset == "confirmation":
        return []
    raise ValueError(f"unknown preset: {preset}")


def matrix_summary(runs: list[DiscoveryRun], root: Path, *, preset: str = "efficient-discovery") -> dict[str, Any]:
    return {
        "preset": preset,
        "labels": list(DISCOVERY_LABELS),
        "run_count": len(runs),
        "output_ids": [run.output_id for run in runs],
        "cpu_reference": {"workers": DEFAULT_WORKERS, "torch_threads": DEFAULT_TORCH_THREADS, "device": "cpu"},
        "runs": [
            {
                **asdict(run),
                "eval_seeds": list(run.eval_seeds),
                "perturb_channels": list(run.perturb_channels),
                "perturb_scales": list(run.perturb_scales),
                "command": run.command(root),
            }
            for run in runs
        ],
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def lineage_key(candidate: dict[str, Any]) -> str:
    ancestors = candidate.get("ancestor_ids") or []
    if ancestors:
        return str(ancestors[0])
    parents = candidate.get("parent_ids") or []
    if parents:
        return str(parents[0])
    return str(candidate.get("id", "unknown"))


def dominant_regime(candidate: dict[str, Any]) -> str:
    regimes = candidate.get("metrics", {}).get("regimes", {})
    if isinstance(regimes, dict) and regimes:
        return str(max(regimes, key=regimes.get))
    runs = candidate.get("runs", [])
    if runs:
        return str(runs[0].get("summary", {}).get("regime_class", "unknown"))
    return "unknown"


def scalar_signature(candidate: dict[str, Any], *, precision: int = 3) -> tuple[float, ...]:
    scalars = candidate.get("genome", {}).get("scalars", {})
    return tuple(round(float(scalars[key]), precision) for key in sorted(scalars))


def candidate_record(path: Path) -> dict[str, Any]:
    candidate = load_json(path)
    archive_root = path.parent.parent
    config_path = archive_root / "config.json"
    archive_config = load_json(config_path) if config_path.exists() else {}
    candidate["_candidate_path"] = path.as_posix()
    candidate["_archive_config"] = archive_config
    candidate["_native_rank"] = scalar_rank(candidate)
    candidate["_lineage_key"] = lineage_key(candidate)
    candidate["_dominant_regime"] = dominant_regime(candidate)
    candidate["_scalar_signature"] = scalar_signature(candidate)
    return candidate


def load_candidates(archive_roots: Iterable[Path]) -> list[dict[str, Any]]:
    candidates = []
    for root in archive_roots:
        for path in sorted((root / "candidates").glob("*.json")):
            candidates.append(candidate_record(path))
    return candidates


def morphology_rank(candidate: dict[str, Any]) -> float:
    return scalar_rank(
        {
            **candidate,
            "rank_mode": NATIVE_EMERGENCE_RANK_MODE,
            "native_objective": NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
        }
    )


def candidate_regimes(candidate: dict[str, Any]) -> dict[str, int]:
    regimes = candidate.get("metrics", {}).get("regimes", {})
    return {str(key): int(value) for key, value in regimes.items()}


def has_regime(candidate: dict[str, Any], regime: str) -> bool:
    return candidate_regimes(candidate).get(regime, 0) > 0


def is_mixed_regime(candidate: dict[str, Any]) -> bool:
    return sum(1 for value in candidate_regimes(candidate).values() if value > 0) > 1


def objective_ranked(candidates: list[dict[str, Any]], objective: str) -> list[dict[str, Any]]:
    matches = [
        candidate
        for candidate in candidates
        if str(candidate.get("native_objective", NATIVE_OBJECTIVE_GATE_STATE)) == objective
    ]
    if objective == NATIVE_OBJECTIVE_COMBINED_DISCOVERY:
        return sorted(matches, key=lambda row: (is_mixed_regime(row), float(row["_native_rank"])), reverse=True)
    return sorted(matches, key=lambda row: float(row["_native_rank"]), reverse=True)


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    per_bucket: int = 3,
    lineage_limit: int = 5,
    target_count: int = CONFIRMATION_TARGET_COUNT,
    objective_quotas: dict[str, int] | None = None,
    regime_quotas: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    objective_quotas = objective_quotas or CONFIRMATION_OBJECTIVE_QUOTAS
    regime_quotas = regime_quotas or CONFIRMATION_REGIME_QUOTAS
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    seen_paths: set[str] = set()

    def add(candidate: dict[str, Any], reason: str) -> bool:
        if candidate["_candidate_path"] in seen_paths:
            return False
        duplicate_key = (candidate["_dominant_regime"], candidate["_scalar_signature"])
        if duplicate_key in seen_keys:
            rejected.append(
                {
                    "candidate_id": candidate.get("id"),
                    "candidate_path": candidate["_candidate_path"],
                    "duplicate_reason": "same dominant regime and rounded scalar signature",
                    "selection_reason": reason,
                }
            )
            return False
        seen_keys.add(duplicate_key)
        seen_paths.add(candidate["_candidate_path"])
        selected.append({**candidate, "_selection_reason": reason})
        return True

    def add_until_quota(candidates_to_add: list[dict[str, Any]], count: int, reason: str) -> None:
        added = 0
        for candidate in candidates_to_add:
            if add(candidate, reason):
                added += 1
            if added >= count:
                break

    ranked = sorted(candidates, key=lambda row: float(row["_native_rank"]), reverse=True)

    for objective, quota in objective_quotas.items():
        add_until_quota(objective_ranked(candidates, objective), quota, f"objective_quota_{objective}")

    for regime, quota in regime_quotas.items():
        current = sum(1 for candidate in selected if has_regime(candidate, regime))
        if current >= quota:
            continue
        ranked_for_regime = sorted(
            [candidate for candidate in candidates if has_regime(candidate, regime)],
            key=lambda row: float(row["_native_rank"]),
            reverse=True,
        )
        add_until_quota(ranked_for_regime, quota - current, f"regime_quota_{regime}")

    morphology_ranked = sorted(candidates, key=morphology_rank, reverse=True)
    for candidate in ranked[:per_bucket]:
        if len(selected) >= target_count:
            break
        add(candidate, "global_fill_native_rank")

    for candidate in morphology_ranked[:per_bucket]:
        if len(selected) >= target_count:
            break
        add(candidate, "global_fill_morphology_rank")

    seen_lineages = {row["_lineage_key"] for row in selected}
    for candidate in ranked:
        if len(selected) >= target_count:
            break
        if candidate["_lineage_key"] in seen_lineages:
            continue
        add(candidate, "global_fill_distinct_lineage")
        seen_lineages.add(candidate["_lineage_key"])
        distinct_lineage_count = sum(
            1 for row in selected if row["_selection_reason"] == "global_fill_distinct_lineage"
        )
        if distinct_lineage_count >= lineage_limit:
            break

    return [
        {
            "candidate_id": str(candidate.get("id", "")),
            "candidate_path": candidate["_candidate_path"],
            "native_rank": float(candidate["_native_rank"]),
            "morphology_rank": morphology_rank(candidate),
            "native_objective": str(candidate.get("native_objective", NATIVE_OBJECTIVE_GATE_STATE)),
            "rank_mode": str(candidate.get("rank_mode", NATIVE_EMERGENCE_RANK_MODE)),
            "causal_mode": str(candidate.get("causal_mode", CAUSAL_MODE_GATE_STATE)),
            "lineage_key": candidate["_lineage_key"],
            "dominant_regime": candidate["_dominant_regime"],
            "regimes": candidate_regimes(candidate),
            "selection_reason": candidate["_selection_reason"],
            "discovery_perturb_step": discovery_perturb_step(candidate),
            "confirmation_commands": confirmation_commands(candidate),
        }
        for candidate in selected
    ]


def discovery_perturb_step(candidate: dict[str, Any]) -> int:
    runs = candidate.get("runs", [])
    if runs:
        step = runs[0].get("metrics", {}).get("perturb_step")
        if step is not None:
            return int(step)
    archive_step = candidate.get("_archive_config", {}).get("perturb_step")
    if archive_step is not None:
        return int(archive_step)
    return 64


def confirmation_commands(candidate: dict[str, Any]) -> list[list[str]]:
    candidate_path = Path(candidate["_candidate_path"])
    archive_root = candidate_path.parent.parent
    candidate_id = str(candidate.get("id", candidate_path.stem))
    steps = sorted({discovery_perturb_step(candidate), 64})
    commands = []
    for step in steps:
        output_dir = CONFIRMATION_ROOT / candidate_id / f"perturb_step_{step:03d}"
        commands.append(
            [
                sys.executable,
                "development/gate_state_propagation_characterization.py",
                "--candidate-path",
                candidate_path.as_posix(),
                "--archive-root",
                archive_root.as_posix(),
                "--output-dir",
                output_dir.as_posix(),
                "--seeds",
                "96,97,98,99,100,101,102",
                "--perturb-scales",
                "0.2,0.35,0.7",
                "--perturb-step",
                str(step),
                "--capsule-pause-step",
                str(step),
                "--device",
                "cpu",
            ]
        )
    return commands


def selection_summary(archive_roots: list[Path]) -> dict[str, Any]:
    candidates = load_candidates(archive_roots)
    selected = select_candidates(candidates)
    return {
        "preset": "confirmation",
        "archive_roots": [path.as_posix() for path in archive_roots],
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selection_policy": "objective_and_regime_quotas_plus_ranked_fill",
        "objective_quotas": dict(CONFIRMATION_OBJECTIVE_QUOTAS),
        "regime_quotas": dict(CONFIRMATION_REGIME_QUOTAS),
        "target_count": CONFIRMATION_TARGET_COUNT,
        "selected": selected,
    }


def parse_archive_roots(values: list[str]) -> list[Path]:
    roots = []
    for value in values:
        roots.extend(Path(item) for item in value.split(",") if item.strip())
    return roots


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="efficient-discovery")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--include-long-horizon", action="store_true")
    parser.add_argument("--write-plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument(
        "--archive-root",
        action="append",
        default=[],
        help="Archive root(s) for confirmation selection.",
    )
    parser.add_argument("--run", action="store_true", help="Execute the generated commands sequentially.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.preset == "confirmation":
        archive_roots = parse_archive_roots(args.archive_root)
        if not archive_roots:
            raise SystemExit("--preset confirmation requires at least one --archive-root")
        summary = selection_summary(archive_roots)
        commands = [
            command
            for selected in summary["selected"]
            for command in selected["confirmation_commands"]
        ]
    else:
        runs = discovery_matrix(args.preset, include_long_horizon=args.include_long_horizon)
        summary = matrix_summary(runs, args.out_root, preset=args.preset)
        commands = [run.command(args.out_root) for run in runs]

    args.write_plan.parent.mkdir(parents=True, exist_ok=True)
    args.write_plan.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.write_plan}")
    if args.run:
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
