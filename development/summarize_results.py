#!/usr/bin/env python3
"""Compact local summary of core Demian experiment artifacts.

The goal is observability for the operator:
    - dual-GRU architecture progression
    - transformer reservoir baseline
    - Mamba reservoir baseline
    - competition runs

This script reads existing local artifacts only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def summarize_dual_gru_family() -> None:
    candidates = [
        DATA / "substrate_lab" / "dual_gru_family_report.json",
        DATA / "substrate_lab" / "dual_gru_family_summary.json",
    ]
    data = None
    path = None
    for candidate in candidates:
        payload = _load_json(candidate)
        if payload:
            data = payload
            path = candidate
            break
    if not data or not path:
        return

    _print_header("Dual-GRU Family")
    print(f"source: {path.relative_to(ROOT)}")
    print(f"focus_substrate: {data.get('focus_substrate')}")
    for name, row in data.get("substrates", {}).items():
        print(f"{name}:")
        print(f"  interiors:                  {row.get('interior_counts', {})}")
        print(f"  attractors:                 {row.get('attractor_counts', {})}")
        print(f"  interior_class_count:       {row.get('interior_class_count', 0)}")
        print(f"  accumulating_fp_share:      {row.get('accumulating_fixed_point_share', 0.0):.4f}")
        print(f"  mean_message_norm:          {row.get('mean_message_norm', 0.0):.4f}")
        print(f"  mean_message_contraction:   {row.get('mean_message_contraction', 0.0):.4f}")
        print(f"  bottleneck_code_entropy:    {row.get('mean_bottleneck_code_entropy', 0.0):.4f}")
        print(f"  mean_flow_dimension:        {row.get('mean_flow_dimension', 0.0):.4f}")


def summarize_dual_gru_v3b_regimes() -> None:
    path = DATA / "substrate_lab" / "dual_gru_v3b_regimes.json"
    data = _load_json(path)
    if not data:
        return

    _print_header("Dual-GRU v3b Regimes")
    print(f"source: {path.relative_to(ROOT)}")
    print(f"focus_regime: {data.get('focus_regime')}")
    for name, row in data.get("substrates", {}).items():
        print(f"{name}:")
        print(f"  interiors:                 {row.get('interior_counts', {})}")
        print(f"  accumulating_fp_share:     {row.get('accumulating_fixed_point_share', 0.0):.4f}")
        print(f"  mean_message_norm:         {row.get('mean_message_norm', 0.0):.4f}")
        print(f"  mean_message_contraction:  {row.get('mean_message_contraction', 0.0):.4f}")
        print(f"  bottleneck_code_entropy:   {row.get('mean_bottleneck_code_entropy', 0.0):.4f}")


def summarize_transformer() -> None:
    path = DATA / "reservoir_batch" / "batch_summary.json"
    data = _load_json(path)
    if not data:
        return

    runs = list(data.values())
    if not runs:
        return

    _print_header("Transformer Reservoir")
    print(f"source: {path.relative_to(ROOT)}")
    print(f"runs:   {len(runs)}")
    print(f"energy_mean:        {mean(r['energy_mean'] for r in runs):.4f}")
    print(f"coherence_mean:     {mean(r['coherence_mean'] for r in runs):.4f}")
    print(f"velocity_align:     {mean(r['velocity_align_mean'] for r in runs):.4f}")
    print(f"autocorr_lag2:      {mean(r['autocorr_lag2'] for r in runs):.4f}")
    print(f"period2_switches:   {mean(r['period2_switches'] for r in runs):.1f}")


def summarize_mamba() -> None:
    path = DATA / "mamba_batch" / "mamba_batch_summary.json"
    data = _load_json(path)
    if not data:
        return

    _print_header("Mamba Reservoir")
    print(f"source: {path.relative_to(ROOT)}")

    for mode in ("cache", "no_cache"):
        runs = list(data.get(mode, {}).values())
        if not runs:
            continue
        print(f"{mode}:")
        print(f"  runs:             {len(runs)}")
        print(f"  energy_mean:      {mean(r['energy_mean'] for r in runs):.4f}")
        print(f"  coherence_mean:   {mean(r['coherence_mean'] for r in runs):.4f}")
        print(f"  velocity_align:   {mean(r['velocity_align_mean'] for r in runs):.4f}")
        print(f"  autocorr_lag2:    {mean(r['autocorr_lag2'] for r in runs):.4f}")
        print(f"  period2_switches: {mean(r['period2_switches'] for r in runs):.1f}")


def _summarize_leaderboard(path: Path) -> dict | None:
    if not path.exists():
        return None

    rounds = set()
    alive_counts: dict[int, int] = {}
    max_step = 0
    max_fitness = 0.0
    final_round = 0
    final_alive = 0

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_alive = "alive" in fieldnames
        fitness_key = "fitness" if "fitness" in fieldnames else "rdelta_mean" if "rdelta_mean" in fieldnames else None
        for row in reader:
            round_n = int(row["round"])
            step_count = int(row["step_count"])
            fitness = float(row[fitness_key]) if fitness_key else 0.0

            rounds.add(round_n)
            if has_alive:
                alive_counts[round_n] = alive_counts.get(round_n, 0) + int(row["alive"])
            max_step = max(max_step, step_count)
            max_fitness = max(max_fitness, fitness)
            if round_n >= final_round:
                final_round = round_n

        final_alive = alive_counts.get(final_round, 0) if has_alive else -1

    return {
        "rounds": len(rounds),
        "final_round": final_round,
        "final_alive": final_alive,
        "max_step_count": max_step,
        "max_fitness": max_fitness,
        "has_alive": has_alive,
        "fitness_key": fitness_key,
    }


def _summarize_competition_log(path: Path) -> dict | None:
    if not path.exists():
        return None

    rounds = 0
    total_deaths = 0
    total_births = 0
    max_round_deaths = 0
    max_round_births = 0

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rounds += 1
            row = json.loads(line)
            deaths = len(row.get("deaths", []))
            births = len(row.get("births", []))
            total_deaths += deaths
            total_births += births
            max_round_deaths = max(max_round_deaths, deaths)
            max_round_births = max(max_round_births, births)

    return {
        "rounds_logged": rounds,
        "total_deaths": total_deaths,
        "total_births": total_births,
        "max_round_deaths": max_round_deaths,
        "max_round_births": max_round_births,
    }


def summarize_competition(name: str) -> None:
    base = DATA / name
    board = _summarize_leaderboard(base / "leaderboard.csv")
    log = _summarize_competition_log(base / "competition_log.jsonl")
    if not board and not log:
        return

    _print_header(f"Competition: {name}")
    if board:
        print(f"leaderboard:       {base.joinpath('leaderboard.csv').relative_to(ROOT)}")
        print(f"rounds:            {board['rounds']}")
        if board["has_alive"]:
            print(f"final_alive:       {board['final_alive']}")
        print(f"max_step_count:    {board['max_step_count']}")
        metric_label = board["fitness_key"] or "fitness"
        print(f"max_{metric_label}:      {board['max_fitness']:.6f}")
    if log:
        print(f"log:               {base.joinpath('competition_log.jsonl').relative_to(ROOT)}")
        print(f"rounds_logged:     {log['rounds_logged']}")
        print(f"total_deaths:      {log['total_deaths']}")
        print(f"total_births:      {log['total_births']}")
        print(f"max_round_deaths:  {log['max_round_deaths']}")
        print(f"max_round_births:  {log['max_round_births']}")


def main() -> None:
    print("Demian experiment summary")
    print(f"root: {ROOT}")

    summarize_dual_gru_v3b_regimes()
    summarize_dual_gru_family()
    summarize_transformer()
    summarize_mamba()
    for name in ("competition50", "competition50_run2", "competition_v3", "competition_hebbian", "competition_smoke"):
        summarize_competition(name)


if __name__ == "__main__":
    main()
