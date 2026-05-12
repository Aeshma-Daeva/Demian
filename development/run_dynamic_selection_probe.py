#!/usr/bin/env python3
"""Plan and inspect Dynamic Selection Probe checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolution.scoring import scalar_rank


DEFAULT_ARCHIVE_ROOT = Path("data/evolution/dynamic_selection_probe_20260513")
DEFAULT_DIAGNOSTICS_ROOT = Path("data/diagnostics/dynamic_selection_probe_20260513")
DEFAULT_PLAN_PATH = Path("data/evolution/dynamic_selection_probe_20260513_checkpoint_plan.json")
CHECKPOINT_TOP_N = {19: 3, 39: 3, 59: 5}
HELD_OUT_SEEDS = (96, 97, 98)
CONFIRMATION_SCALES = (0.2, 0.35, 0.7)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_paths_for_generation(archive_root: Path, generation: int) -> list[Path]:
    return sorted((archive_root / "candidates").glob(f"gen{generation:03d}_candidate*.json"))


def load_generation_candidates(archive_root: Path, generation: int) -> list[dict[str, Any]]:
    candidates = []
    for path in candidate_paths_for_generation(archive_root, generation):
        candidate = load_json(path)
        candidate["_candidate_path"] = path.as_posix()
        candidates.append(candidate)
    return candidates


def select_checkpoint_candidates(
    archive_root: Path,
    *,
    checkpoints: dict[int, int] = CHECKPOINT_TOP_N,
) -> list[dict[str, Any]]:
    selected = []
    for generation, top_n in checkpoints.items():
        candidates = load_generation_candidates(archive_root, generation)
        ranked = sorted(candidates, key=scalar_rank, reverse=True)
        for rank_index, candidate in enumerate(ranked[:top_n], start=1):
            selected.append(
                {
                    "checkpoint_generation": generation,
                    "checkpoint_rank": rank_index,
                    "candidate_id": str(candidate.get("id", "")),
                    "candidate_path": candidate["_candidate_path"],
                    "rank_score": scalar_rank(candidate),
                    "rank_components": candidate.get("rank_components", {}),
                    "metrics": candidate.get("metrics", {}),
                    "parent_ids": candidate.get("parent_ids", []),
                    "ancestor_ids": candidate.get("ancestor_ids", []),
                    "confirmation_commands": confirmation_commands(
                        candidate,
                        diagnostics_root=DEFAULT_DIAGNOSTICS_ROOT,
                    ),
                }
            )
    return selected


def confirmation_commands(
    candidate: dict[str, Any],
    *,
    diagnostics_root: Path,
) -> list[list[str]]:
    candidate_path = Path(candidate["_candidate_path"])
    candidate_id = str(candidate.get("id", candidate_path.stem))
    generation = int(candidate.get("generation", 0))
    output_dir = diagnostics_root / f"gen{generation:03d}" / candidate_id / "perturb_step_064"
    return [
        [
            "venv/bin/python",
            "development/gate_state_propagation_characterization.py",
            "--candidate-path",
            candidate_path.as_posix(),
            "--archive-root",
            candidate_path.parent.parent.as_posix(),
            "--output-dir",
            output_dir.as_posix(),
            "--seeds",
            ",".join(str(seed) for seed in HELD_OUT_SEEDS),
            "--perturb-scales",
            ",".join(f"{scale:g}" for scale in CONFIRMATION_SCALES),
            "--perturb-step",
            "64",
            "--capsule-pause-step",
            "64",
            "--device",
            "cpu",
        ]
    ]


def checkpoint_plan(archive_root: Path, diagnostics_root: Path) -> dict[str, Any]:
    selected = select_checkpoint_candidates(archive_root)
    for row in selected:
        row["confirmation_commands"] = confirmation_commands(
            {
                **row,
                "id": row["candidate_id"],
                "generation": row["checkpoint_generation"],
                "_candidate_path": row["candidate_path"],
            },
            diagnostics_root=diagnostics_root,
        )
    return {
        "experiment": "dynamic_selection_probe",
        "archive_root": archive_root.as_posix(),
        "diagnostics_root": diagnostics_root.as_posix(),
        "checkpoints": dict(CHECKPOINT_TOP_N),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "confirmation_scales": list(CONFIRMATION_SCALES),
        "selected": selected,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--diagnostics-root", type=Path, default=DEFAULT_DIAGNOSTICS_ROOT)
    parser.add_argument("--write-plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--emit-commands", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    plan = checkpoint_plan(args.archive_root, args.diagnostics_root)
    args.write_plan.parent.mkdir(parents=True, exist_ok=True)
    args.write_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.write_plan}")
    if args.emit_commands:
        for selected in plan["selected"]:
            for command in selected["confirmation_commands"]:
                print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
