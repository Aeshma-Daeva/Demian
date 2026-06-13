#!/usr/bin/env python3
"""Adversarial truth-finding campaign for gate-state propagation claims."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolution.config import CHANNELS
from development.evolve_v9_5ch_release import default_genome, random_genome
from development.gate_state_control_nulls import (
    CandidateGateRow,
    CandidateSpec,
    apply_strict_profile_gate,
    calibrated_strict_thresholds,
    pass_rate_comparisons,
    positive_candidates_from_replication_summary,
    row_for_summary,
)
from development.gate_state_propagation_characterization import (
    CharacterizationConfig,
    MotifSpec,
    continue_resume,
    motif_from_spec,
    normalized_l2,
    parse_csv_floats,
    parse_csv_ints,
    parse_motifs,
    run_characterization,
    run_to_pause_and_baseline,
    write_csv,
    write_json,
)
from development.substrate_lab import perturbation_pair

DEFAULT_REPLICATION_SUMMARY = Path("data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json")
DEFAULT_OUTPUT_DIR = Path(f"data/diagnostics/gate_state_truth_campaign_{date.today().strftime('%Y%m%d')}")

CampaignFamily = Literal["track_b_positive", "random_unselected", "default_unselected", "default_jittered", "baseline"]
SplitName = Literal["calibration", "test", "baseline"]


class FlexibleSchema(BaseModel):
    """Validate known fields while preserving campaign metadata."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class CampaignConfig(FlexibleSchema):
    replication_summary_path: Path = DEFAULT_REPLICATION_SUMMARY
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seeds: list[int] = Field(default_factory=lambda: [96, 97, 98])
    perturb_scales: list[float] = Field(default_factory=lambda: [0.35])
    motifs: list[MotifSpec] = Field(default_factory=lambda: [MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)])
    steps: NonNegativeInt = 128
    perturb_step: NonNegativeInt = 64
    capsule_pause_step: NonNegativeInt = 64
    hidden_size: NonNegativeInt = 32
    rank: NonNegativeInt = 2
    track_b_count: NonNegativeInt = 30
    random_control_count: NonNegativeInt = 1000
    default_jitter_control_count: NonNegativeInt = 100
    calibration_track_b_count: NonNegativeInt = 10
    calibration_random_count: NonNegativeInt = 300
    calibration_default_jitter_count: NonNegativeInt = 30
    random_seed: int = 5152026
    default_jitter_scale: float = 0.015
    device: str = "cpu"
    baseline_substrates: list[str] = Field(
        default_factory=lambda: ["rnn", "gru", "lstm", "diag_ssm", "demian_native_v8", "demian_native_v9"]
    )
    baseline_seeds: list[int] = Field(default_factory=lambda: [96, 97, 98])
    baseline_perturb_scale: float = 0.03
    surgery_pause_steps: list[int] = Field(default_factory=list)
    surgery_splits: list[SplitName] = Field(default_factory=lambda: ["test"])

    @model_validator(mode="after")
    def split_counts_fit(self) -> "CampaignConfig":
        if self.calibration_track_b_count > self.track_b_count:
            raise ValueError("calibration_track_b_count must be <= track_b_count")
        if self.calibration_random_count > self.random_control_count:
            raise ValueError("calibration_random_count must be <= random_control_count")
        if self.calibration_default_jitter_count > self.default_jitter_control_count:
            raise ValueError("calibration_default_jitter_count must be <= default_jitter_control_count")
        if self.perturb_step > self.steps:
            raise ValueError("perturb_step must be <= steps")
        if self.capsule_pause_step > self.steps:
            raise ValueError("capsule_pause_step must be <= steps")
        if any(step <= 0 or step >= self.steps for step in self.surgery_pause_steps):
            raise ValueError("surgery_pause_steps must be within [1, steps - 1]")
        return self


class CampaignCandidate(FlexibleSchema):
    candidate_id: str
    candidate_path: Path
    family: CampaignFamily
    split: SplitName


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_candidate(path: Path, *, candidate_id: str, genome: dict[str, Any], family: CampaignFamily, seed: int | None) -> None:
    payload = {
        "id": candidate_id,
        "generation": 0,
        "index": 0,
        "rank_mode": "truth_campaign",
        "control_family": family,
        "control_seed": seed,
        "genome": genome,
        "metrics": {},
        "reproduction_kind": "truth_campaign_control",
        "parent_ids": [],
        "ancestor_ids": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jitter_numeric_tree(value: Any, rng: random.Random, scale: float) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value) + rng.gauss(0.0, scale)
    if isinstance(value, list):
        return [jitter_numeric_tree(item, rng, scale) for item in value]
    if isinstance(value, dict):
        return {key: jitter_numeric_tree(item, rng, scale) for key, item in value.items()}
    return value


def split_name(index: int, calibration_count: int) -> SplitName:
    return "calibration" if index < calibration_count else "test"


def build_campaign_candidates(config: CampaignConfig) -> list[CampaignCandidate]:
    positives = positive_candidates_from_replication_summary(config.replication_summary_path)[: int(config.track_b_count)]
    candidates = [
        CampaignCandidate(
            candidate_id=candidate.candidate_id,
            candidate_path=candidate.candidate_path,
            family="track_b_positive",
            split=split_name(idx, int(config.calibration_track_b_count)),
        )
        for idx, candidate in enumerate(positives)
    ]

    rng = random.Random(int(config.random_seed))
    control_dir = config.output_dir / "control_candidates"
    for idx in range(int(config.random_control_count)):
        seed = rng.randrange(1_000_000_000)
        candidate_id = f"random_unselected_control_{idx:04d}"
        path = control_dir / f"{candidate_id}.json"
        write_candidate(
            path,
            candidate_id=candidate_id,
            genome=random_genome(random.Random(seed), int(config.hidden_size), int(config.rank)),
            family="random_unselected",
            seed=seed,
        )
        candidates.append(
            CampaignCandidate(
                candidate_id=candidate_id,
                candidate_path=path,
                family="random_unselected",
                split=split_name(idx, int(config.calibration_random_count)),
            )
        )

    default = default_genome(int(config.hidden_size), int(config.rank))
    for idx in range(int(config.default_jitter_control_count)):
        seed = rng.randrange(1_000_000_000)
        local_rng = random.Random(seed)
        family: CampaignFamily = "default_unselected" if idx == 0 else "default_jittered"
        candidate_id = "default_unselected_control" if idx == 0 else f"default_jittered_control_{idx:04d}"
        genome = copy.deepcopy(default) if idx == 0 else jitter_numeric_tree(default, local_rng, float(config.default_jitter_scale))
        path = control_dir / f"{candidate_id}.json"
        write_candidate(path, candidate_id=candidate_id, genome=genome, family=family, seed=None if idx == 0 else seed)
        candidates.append(
            CampaignCandidate(
                candidate_id=candidate_id,
                candidate_path=path,
                family=family,
                split=split_name(idx, int(config.calibration_default_jitter_count)),
            )
        )
    return candidates


def characterization_config_for(candidate: CampaignCandidate, config: CampaignConfig) -> CharacterizationConfig:
    return CharacterizationConfig(
        candidate_path=candidate.candidate_path,
        archive_root=candidate.candidate_path.parent.parent,
        output_dir=config.output_dir / "characterizations" / candidate.split / candidate.family / candidate.candidate_id,
        seeds=list(config.seeds),
        perturb_scales=list(config.perturb_scales),
        motifs=list(config.motifs),
        steps=int(config.steps),
        perturb_step=int(config.perturb_step),
        capsule_pause_step=int(config.capsule_pause_step),
        hidden_size=int(config.hidden_size),
        rank=int(config.rank),
        device=config.device,
        write_parameter_signatures=False,
    )


def candidate_spec(candidate: CampaignCandidate) -> CandidateSpec:
    return CandidateSpec(candidate_id=candidate.candidate_id, candidate_path=candidate.candidate_path, family=candidate.family)


def confidence_interval_wilson(success: int, total: int, z: float = 1.96) -> dict[str, float | None]:
    if total <= 0:
        return {"low": None, "high": None}
    p = success / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return {"low": max(0.0, center - half), "high": min(1.0, center + half)}


def summarize_gate_rows(rows: list[CandidateGateRow], *, split: str) -> list[dict[str, Any]]:
    output = []
    for family in sorted({row.family for row in rows}):
        items = [row for row in rows if row.family == family]
        for gate in ("mechanism_gate_pass", "strict_profile_gate_pass"):
            passed = sum(bool(getattr(row, gate)) for row in items)
            total = len(items)
            output.append(
                {
                    "split": split,
                    "family": family,
                    "gate": gate,
                    "pass_count": passed,
                    "total": total,
                    "pass_rate": passed / total if total else 0.0,
                    **{f"ci95_{key}": value for key, value in confidence_interval_wilson(passed, total).items()},
                    "message_carrier_dominance_mean": mean([row.message_carrier_dominance for row in items]) if items else 0.0,
                    "message_carrier_dominance_std": pstdev([row.message_carrier_dominance for row in items]) if len(items) > 1 else 0.0,
                }
            )
    return output


def pseudo_channel_slices(hidden_size: int, labels: tuple[str, ...] = tuple(CHANNELS)) -> dict[str, tuple[int, int]]:
    if hidden_size < len(labels):
        raise ValueError("hidden_size must be at least the number of pseudo-channels")
    base = hidden_size // len(labels)
    remainder = hidden_size % len(labels)
    start = 0
    slices = {}
    for idx, label in enumerate(labels):
        width = base + (1 if idx < remainder else 0)
        slices[label] = (start, start + width)
        start += width
    return slices


def run_baseline_comparisons(config: CampaignConfig) -> dict[str, Any]:
    rows = []
    for substrate in config.baseline_substrates:
        for seed in config.baseline_seeds:
            result = perturbation_pair(
                substrate,
                hidden_size=int(config.hidden_size),
                steps=int(config.steps),
                seed=int(seed),
                perturb_step=int(config.perturb_step),
                perturb_scale=float(config.baseline_perturb_scale),
                device=config.device,
            )
            rows.append(
                {
                    "substrate": substrate,
                    "seed": int(seed),
                    "final_l2_gap": float(result["final_l2_gap"]),
                    "final_cosine": float(result["final_cosine"]),
                    "recovery_window_mean_gap": float(result["recovery_window_mean_gap"]),
                    "clean_attractor_type": result["clean"]["attractor_type"],
                    "perturbed_attractor_type": result["perturbed"]["attractor_type"],
                }
            )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["substrate"])].append(row)
    summary_rows = []
    for substrate, items in sorted(grouped.items()):
        summary_rows.append(
            {
                "substrate": substrate,
                "run_count": len(items),
                "final_l2_gap_mean": mean([row["final_l2_gap"] for row in items]),
                "final_l2_gap_std": pstdev([row["final_l2_gap"] for row in items]) if len(items) > 1 else 0.0,
                "recovery_window_mean_gap_mean": mean([row["recovery_window_mean_gap"] for row in items]),
                "pseudo_channel_slices": json.dumps(pseudo_channel_slices(int(config.hidden_size)), sort_keys=True),
            }
        )
    return {
        "rows": rows,
        "summary": summary_rows,
        "pseudo_channel_slices": pseudo_channel_slices(int(config.hidden_size)),
        "interpretation": "Plain recurrence baselines are measured with matched hidden-size perturbation runs and fixed pseudo-channel partitions.",
    }


def swap_state_components(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
    channels: list[str],
) -> tuple[torch.Tensor, ...]:
    if len(left) != len(CHANNELS) or len(right) != len(CHANNELS):
        raise ValueError("state surgery expects five-channel v9 release states")
    values = [tensor.detach().clone() for tensor in left]
    for channel in channels:
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel: {channel}")
        idx = list(CHANNELS).index(channel)
        values[idx] = right[idx].detach().clone()
    return tuple(values)


def zero_state_components(state: tuple[torch.Tensor, ...], channels: list[str]) -> tuple[torch.Tensor, ...]:
    values = [tensor.detach().clone() for tensor in state]
    for channel in channels:
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel: {channel}")
        idx = list(CHANNELS).index(channel)
        values[idx] = torch.zeros_like(values[idx])
    return tuple(values)


def surgery_pause_schedule(config: CampaignConfig) -> list[int]:
    if config.surgery_pause_steps:
        return sorted({int(step) for step in config.surgery_pause_steps})
    candidates = {
        int(config.perturb_step),
        int(config.capsule_pause_step),
        max(1, int(config.steps) // 4),
        max(1, int(config.steps) // 2),
        max(1, (3 * int(config.steps)) // 4),
    }
    return sorted(step for step in candidates if 0 < step < int(config.steps))


def windowed_gap_rows(gaps: list[float]) -> list[dict[str, Any]]:
    if not gaps:
        return [
            {"window": "full", "window_start_offset": 1, "window_end_offset": 0, "mean_step_gap": 0.0},
        ]
    windows = [{"window": "full", "start": 0, "end": len(gaps)}]
    if len(gaps) >= 3:
        first = max(1, len(gaps) // 3)
        second = max(first + 1, (2 * len(gaps)) // 3)
        windows.extend(
            [
                {"window": "early", "start": 0, "end": first},
                {"window": "mid", "start": first, "end": second},
                {"window": "late", "start": second, "end": len(gaps)},
            ]
        )
    rows = []
    for window in windows:
        values = gaps[int(window["start"]): int(window["end"])]
        rows.append(
            {
                "window": window["window"],
                "window_start_offset": int(window["start"]) + 1,
                "window_end_offset": int(window["end"]),
                "mean_step_gap": mean(values) if values else 0.0,
            }
        )
    return rows


def summarize_state_surgery(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["candidate_id"]), int(row["pause_step"]), str(row["window"]))].append(row)
    summary = []
    for (candidate_id, pause_step, window), items in sorted(grouped.items()):
        by_intervention = {str(row["intervention"]): row for row in items}
        message_carrier = float(by_intervention.get("message_carrier_swap", {}).get("mean_step_gap", 0.0))
        slow_control = float(by_intervention.get("slow_control_swap", {}).get("mean_step_gap", 0.0))
        summary.append(
            {
                "candidate_id": candidate_id,
                "pause_step": pause_step,
                "window": window,
                "message_carrier_swap_gap": message_carrier,
                "slow_control_swap_gap": slow_control,
                "message_carrier_minus_slow_control": message_carrier - slow_control,
                "dominant_swap": "message_carrier" if message_carrier > slow_control else "slow_control",
            }
        )
    return summary


def run_state_surgery(config: CampaignConfig, candidates: list[CampaignCandidate]) -> dict[str, Any]:
    requested_splits = set(config.surgery_splits)
    track_b = [
        candidate
        for candidate in candidates
        if candidate.family == "track_b_positive" and candidate.split in requested_splits
    ]
    if not track_b or len(config.seeds) < 2:
        return {"rows": [], "summary": [], "note": "state surgery requires a held-out Track B candidate and at least two seeds"}
    rows = []
    pause_steps = surgery_pause_schedule(config)
    for candidate in track_b:
        genome = load_json(candidate.candidate_path)["genome"]
        for pause in pause_steps:
            char_config = characterization_config_for(candidate, config).model_copy(update={"capsule_pause_step": pause})
            motif = motif_from_spec(config.motifs[0], int(config.hidden_size), int(config.seeds[0]))
            left = run_to_pause_and_baseline(
                genome,
                candidate_id=candidate.candidate_id,
                config=char_config,
                seed=int(config.seeds[0]),
                perturb_scale=float(config.perturb_scales[0]),
                motif=motif,
            )
            right_motif = motif_from_spec(config.motifs[0], int(config.hidden_size), int(config.seeds[1]))
            right = run_to_pause_and_baseline(
                genome,
                candidate_id=candidate.candidate_id,
                config=char_config,
                seed=int(config.seeds[1]),
                perturb_scale=float(config.perturb_scales[0]),
                motif=right_motif,
            )
            target = left["surfaces"][pause:]
            interventions = {
                "message_carrier_swap": swap_state_components(left["paused_state"], right["paused_state"], ["message", "carrier"]),
                "slow_control_swap": swap_state_components(left["paused_state"], right["paused_state"], ["slow", "control"]),
                "message_carrier_zero": zero_state_components(left["paused_state"], ["message", "carrier"]),
                "slow_control_zero": zero_state_components(left["paused_state"], ["slow", "control"]),
            }
            for name, state in interventions.items():
                surfaces, _ = continue_resume(copy.deepcopy(left["paused_model"]), state, start_step=pause, steps=int(config.steps))
                gaps = [normalized_l2(observed, expected) for observed, expected in zip(surfaces, target)]
                final_gap = normalized_l2(surfaces[-1], target[-1]) if surfaces and target else 0.0
                for window_row in windowed_gap_rows(gaps):
                    rows.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "pause_step": pause,
                            "intervention": name,
                            "window": window_row["window"],
                            "window_start_offset": window_row["window_start_offset"],
                            "window_end_offset": window_row["window_end_offset"],
                            "seed_left": int(config.seeds[0]),
                            "seed_right": int(config.seeds[1]),
                            "mean_step_gap": window_row["mean_step_gap"],
                            "final_l2_gap": final_gap,
                        }
                    )
    return {"rows": rows, "summary": summarize_state_surgery(rows), "pause_steps": pause_steps}


def run_gate_rows(config: CampaignConfig, candidates: list[CampaignCandidate]) -> tuple[list[CandidateGateRow], dict[str, Any]]:
    raw_rows = []
    split_by_id = {}
    for candidate in candidates:
        summary = run_characterization(characterization_config_for(candidate, config))
        row = row_for_summary(candidate_spec(candidate), summary)
        raw_rows.append(row)
        split_by_id[row.candidate_id] = candidate.split
    calibration_rows = [row for row in raw_rows if split_by_id[row.candidate_id] == "calibration"]
    thresholds = locked_strict_thresholds_from_calibration(calibration_rows)
    rows = apply_strict_profile_gate(raw_rows, thresholds)
    return rows, {"thresholds": thresholds, "split_by_candidate_id": split_by_id}


def locked_strict_thresholds_from_calibration(rows: list[CandidateGateRow]) -> dict[str, float]:
    eligible = [row for row in rows if row.family == "track_b_positive" and row.mechanism_gate_pass]
    if not eligible:
        calibration_track_b = [row for row in rows if row.family == "track_b_positive"]
        details = [
            {
                "candidate_id": row.candidate_id,
                "mechanism_gate_pass": bool(row.mechanism_gate_pass),
                "gain_zero_clean": bool(row.gain_zero_clean),
                "routes_disabled_divergence_positive": bool(row.routes_disabled_divergence_positive),
                "gain_zero_divergence_positive": bool(row.gain_zero_divergence_positive),
                "full_resume_exact": bool(row.full_resume_exact),
                "surface_resume_gap_positive": bool(row.surface_resume_gap_positive),
                "message_carrier_top2": bool(row.message_carrier_top2),
            }
            for row in calibration_track_b
        ]
        raise RuntimeError(
            "Cannot lock strict-profile thresholds: no calibration Track B candidate passed the broad "
            f"mechanism gate. Calibration Track B diagnostics: {json.dumps(details, sort_keys=True)}"
        )
    return calibrated_strict_thresholds(rows)


def campaign_decision(test_rows: list[CandidateGateRow], baseline_summary: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in test_rows if row.family == "track_b_positive"]
    controls = [row for row in test_rows if row.family != "track_b_positive"]
    positive_strict = mean([1.0 if row.strict_profile_gate_pass else 0.0 for row in positives]) if positives else 0.0
    control_strict = mean([1.0 if row.strict_profile_gate_pass else 0.0 for row in controls]) if controls else 0.0
    max_baseline_gap = max((float(row["recovery_window_mean_gap_mean"]) for row in baseline_summary), default=0.0)
    if not positives:
        label = "undecided_no_held_out_track_b"
    elif positive_strict <= control_strict:
        label = "demote_to_ordinary_or_unselected_recurrence"
    elif max_baseline_gap > 0.0:
        label = "structured_recurrent_history_pending_probe_and_surgery_superiority"
    else:
        label = "gate_state_claim_not_established"
    return {
        "decision": label,
        "held_out_track_b_strict_pass_rate": positive_strict,
        "held_out_control_strict_pass_rate": control_strict,
        "max_plain_baseline_recovery_gap": max_baseline_gap,
        "rule": "Keep the strong name only if held-out Track B beats controls and baselines on strict gate, probes, and surgery.",
    }


def run_truth_campaign(config: CampaignConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    locked_config = {
        **config.model_dump(mode="json"),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "primary_hypothesis": "Track B candidates show stronger message/carrier-mediated history propagation than matched controls and recurrence baselines.",
        "null_hypothesis": "The same pattern appears in ordinary recurrence or random five-channel controls at similar rates or effect sizes.",
        "split_discipline": "Thresholds are calibrated only on calibration split rows, then reused unchanged for held-out test rows.",
    }
    write_json(config.output_dir / "locked_config.json", locked_config)

    candidates = build_campaign_candidates(config)
    candidate_rows = [candidate.model_dump(mode="json") for candidate in candidates]
    write_csv(config.output_dir / "candidate_splits.csv", candidate_rows)
    write_json(config.output_dir / "candidate_splits.json", candidate_rows)

    gate_rows, threshold_payload = run_gate_rows(config, candidates)
    split_by_id = threshold_payload["split_by_candidate_id"]
    calibration_rows = [row for row in gate_rows if split_by_id[row.candidate_id] == "calibration"]
    test_rows = [row for row in gate_rows if split_by_id[row.candidate_id] == "test"]

    baseline = run_baseline_comparisons(config)
    surgery = run_state_surgery(config, candidates)

    row_dicts = []
    for row in gate_rows:
        payload = row.model_dump(mode="json")
        payload["split"] = split_by_id[row.candidate_id]
        row_dicts.append(payload)
    calibration_summary = summarize_gate_rows(calibration_rows, split="calibration")
    test_summary = summarize_gate_rows(test_rows, split="test")
    decision = campaign_decision(test_rows, baseline["summary"])
    payload = {
        "experiment": "gate_state_truth_campaign",
        "config_path": str(config.output_dir / "locked_config.json"),
        "candidate_count": len(candidates),
        "locked_thresholds": threshold_payload["thresholds"],
        "calibration_summary": calibration_summary,
        "held_out_test_summary": test_summary,
        "pass_rate_comparisons": pass_rate_comparisons(test_rows),
        "baseline_comparison_summary": baseline["summary"],
        "state_surgery_summary": surgery["summary"],
        "probe_summary": {
            "source": "per-candidate characterization channel_probe_summary.json artifacts",
            "required_evidence": "Internal message/carrier channels must predict prior gate/release history better than surface state and baselines.",
        },
        "decision": decision,
    }
    write_json(config.output_dir / "locked_thresholds.json", threshold_payload["thresholds"])
    write_csv(config.output_dir / "gate_rows.csv", row_dicts)
    write_csv(config.output_dir / "calibration_summary.csv", calibration_summary)
    write_csv(config.output_dir / "held_out_test_summary.csv", test_summary)
    write_csv(config.output_dir / "baseline_comparison_rows.csv", baseline["rows"])
    write_csv(config.output_dir / "baseline_comparison_summary.csv", baseline["summary"])
    write_json(config.output_dir / "baseline_comparison_summary.json", baseline)
    write_csv(config.output_dir / "state_surgery_rows.csv", surgery["rows"])
    write_json(config.output_dir / "state_surgery_summary.json", surgery)
    write_json(config.output_dir / "truth_campaign_summary.json", payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-summary-path", type=Path, default=DEFAULT_REPLICATION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="96,97,98")
    parser.add_argument("--perturb-scales", default="0.35")
    parser.add_argument("--motifs", default="basis:0,gaussian:0")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--perturb-step", type=int, default=64)
    parser.add_argument("--capsule-pause-step", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--track-b-count", type=int, default=30)
    parser.add_argument("--random-control-count", type=int, default=1000)
    parser.add_argument("--default-jitter-control-count", type=int, default=100)
    parser.add_argument("--calibration-track-b-count", type=int, default=10)
    parser.add_argument("--calibration-random-count", type=int, default=300)
    parser.add_argument("--calibration-default-jitter-count", type=int, default=30)
    parser.add_argument("--baseline-substrates", default="rnn,gru,lstm,diag_ssm,demian_native_v8,demian_native_v9")
    parser.add_argument("--baseline-seeds", default="96,97,98")
    parser.add_argument("--surgery-pause-steps", default="", help="Optional comma-separated pause steps for multi-timepoint state surgery.")
    parser.add_argument("--surgery-splits", default="test", help="Comma-separated split names to include in state surgery.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Use tiny counts for a fast artifact-contract run.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = parse_csv_ints(args.seeds)
    scales = parse_csv_floats(args.perturb_scales)
    motifs = parse_motifs(args.motifs)
    baseline_substrates = [item.strip() for item in args.baseline_substrates.split(",") if item.strip()]
    baseline_seeds = parse_csv_ints(args.baseline_seeds)
    surgery_pause_steps = parse_csv_ints(args.surgery_pause_steps) if args.surgery_pause_steps.strip() else []
    surgery_splits = [item.strip() for item in args.surgery_splits.split(",") if item.strip()]
    if args.smoke:
        seeds = seeds[:2] if len(seeds) >= 2 else [96, 97]
        scales = scales[:1]
        motifs = motifs[:1]
        args.steps = min(args.steps, 12)
        args.perturb_step = min(args.perturb_step, 4)
        args.capsule_pause_step = min(args.capsule_pause_step, 4)
        args.track_b_count = 1
        args.random_control_count = 2
        args.default_jitter_control_count = 1
        args.calibration_track_b_count = 1
        args.calibration_random_count = 1
        args.calibration_default_jitter_count = 1
        baseline_substrates = baseline_substrates[:1]
        baseline_seeds = baseline_seeds[:1]
    payload = run_truth_campaign(
        CampaignConfig(
            replication_summary_path=args.replication_summary_path,
            output_dir=args.output_dir,
            seeds=seeds,
            perturb_scales=scales,
            motifs=motifs,
            steps=args.steps,
            perturb_step=args.perturb_step,
            capsule_pause_step=args.capsule_pause_step,
            hidden_size=args.hidden_size,
            rank=args.rank,
            track_b_count=args.track_b_count,
            random_control_count=args.random_control_count,
            default_jitter_control_count=args.default_jitter_control_count,
            calibration_track_b_count=args.calibration_track_b_count,
            calibration_random_count=args.calibration_random_count,
            calibration_default_jitter_count=args.calibration_default_jitter_count,
            baseline_substrates=baseline_substrates,
            baseline_seeds=baseline_seeds,
            surgery_pause_steps=surgery_pause_steps,
            surgery_splits=surgery_splits,
            device=args.device,
        )
    )
    print(f"wrote {args.output_dir / 'truth_campaign_summary.json'}")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
