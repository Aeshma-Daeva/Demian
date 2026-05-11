#!/usr/bin/env python3
"""Diagnostic evolution for demian_native_v7.2 metabolic parameters.

This is not an optimizer for a single winner. It searches for parameter
families that trade off identity recovery, message containment, coupling, and
clean basin health, then records the Pareto frontier for inspection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrate_lab import (  # noqa: E402
    basin_map,
    coupling_stress_test,
    perturbation_stress_test,
    save_json,
)


@dataclass(frozen=True)
class GeneSpec:
    low: float
    high: float
    default: float
    step: float
    log_mutation: bool = False


GENES: Dict[str, GeneSpec] = {
    "resource_retention": GeneSpec(0.78, 0.99, 0.94, 0.035),
    "resource_gain": GeneSpec(0.0, 0.16, 0.08, 0.025),
    "resource_cost_scale": GeneSpec(0.0, 0.18, 0.07, 0.025),
    "environment_constraint_scale": GeneSpec(0.0, 0.20, 0.10, 0.035),
    "anti_self_scale": GeneSpec(0.0, 0.35, 0.18, 0.05),
    "rss_floor": GeneSpec(0.0, 0.18, 0.08, 0.025),
    "rss_target": GeneSpec(0.12, 0.60, 0.32, 0.055),
    "resource_tightness_scale": GeneSpec(0.0, 0.08, 0.03, 0.012),
    "v72_exposure_scale": GeneSpec(0.0, 0.08, 0.04, 0.012),
}


def _clamp(value: float, spec: GeneSpec) -> float:
    return min(spec.high, max(spec.low, value))


def _repair(genome: Dict[str, float]) -> Dict[str, float]:
    fixed = {name: _clamp(float(genome[name]), spec) for name, spec in GENES.items()}
    if fixed["rss_target"] <= fixed["rss_floor"] + 0.03:
        fixed["rss_target"] = min(
            GENES["rss_target"].high,
            fixed["rss_floor"] + 0.03,
        )
    return fixed


def default_genome() -> Dict[str, float]:
    return {name: spec.default for name, spec in GENES.items()}


def random_genome(rng: random.Random) -> Dict[str, float]:
    genome = {}
    for name, spec in GENES.items():
        genome[name] = rng.uniform(spec.low, spec.high)
    return _repair(genome)


def mutate_genome(
    parent: Dict[str, float],
    rng: random.Random,
    sigma: float,
) -> Dict[str, float]:
    child = dict(parent)
    for name, spec in GENES.items():
        if rng.random() < 0.72:
            if spec.log_mutation and child[name] > 0.0:
                child[name] *= math.exp(rng.gauss(0.0, sigma))
            else:
                child[name] += rng.gauss(0.0, spec.step * sigma)
    return _repair(child)


def crossover(
    left: Dict[str, float],
    right: Dict[str, float],
    rng: random.Random,
) -> Dict[str, float]:
    child = {}
    for name in GENES:
        child[name] = left[name] if rng.random() < 0.5 else right[name]
    return _repair(child)


def evaluate_genome(
    genome: Dict[str, float],
    *,
    hidden_size: int,
    steps: int,
    seeds: List[int],
    perturb_step: int,
    perturb_scales: List[float],
    coupling_strengths: List[float],
    coupling_dim: int,
    coupling_interval: int,
    device: str,
) -> Dict[str, float | Dict[str, int]]:
    basin_rows = basin_map(
        "demian_native_v7.2",
        hidden_size=hidden_size,
        steps=steps,
        seeds=seeds,
        device=device,
        substrate_kwargs=genome,
    )
    perturb_recoveries: List[float] = []
    perturb_peak_message_ratios: List[float] = []
    coupling_finals: List[float] = []
    coupling_abs_finals: List[float] = []

    for seed in seeds:
        perturb = perturbation_stress_test(
            "demian_native_v7.2",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            perturb_step=perturb_step,
            perturb_scales=perturb_scales,
            perturb_mode="rss_negation",
            device=device,
            substrate_kwargs=genome,
        )
        perturb_recoveries.extend(
            case["final_cosine_vs_baseline"] for case in perturb["cases"]
        )
        perturb_peak_message_ratios.extend(
            case["peak_norm_ratio_vs_baseline"]["message"]
            for case in perturb["cases"]
        )
        coupling = coupling_stress_test(
            "demian_native_v7.2",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            coupling_dim=coupling_dim,
            coupling_interval=coupling_interval,
            coupling_strengths=coupling_strengths,
            device=device,
            substrate_kwargs=genome,
        )
        coupling_finals.extend(case["final_cosine"] for case in coupling["cases"])
        coupling_abs_finals.extend(abs(case["final_cosine"]) for case in coupling["cases"])

    attractors: Dict[str, int] = {}
    interiors: Dict[str, int] = {}
    for row in basin_rows:
        attractors[row.attractor_type] = attractors.get(row.attractor_type, 0) + 1
        interiors[row.interior_class] = interiors.get(row.interior_class, 0) + 1

    rss_recovery = mean(perturb_recoveries)
    message_peak_ratio = mean(perturb_peak_message_ratios)
    mean_message_norm = mean(row.mean_message_norm for row in basin_rows)
    clean_coherence = mean(row.mean_coherence for row in basin_rows)
    flow_dimension = mean(row.flow_dimension for row in basin_rows)
    clean_delta = mean(row.mean_delta for row in basin_rows)
    coupling_alignment = mean(coupling_finals)
    coupling_magnitude = mean(coupling_abs_finals)
    seed_stability = -pstdev(perturb_recoveries) if len(perturb_recoveries) > 1 else 0.0
    fixed_point_share = attractors.get("FIXED_POINT", 0) / max(len(basin_rows), 1)

    return {
        "rss_recovery": rss_recovery,
        "message_containment": -message_peak_ratio,
        "message_peak_ratio": message_peak_ratio,
        "mean_message_norm": mean_message_norm,
        "clean_coherence": clean_coherence,
        "flow_dimension": flow_dimension,
        "clean_delta": clean_delta,
        "coupling_alignment": coupling_alignment,
        "coupling_magnitude": coupling_magnitude,
        "seed_stability": seed_stability,
        "fixed_point_share": fixed_point_share,
        "attractors": attractors,
        "interiors": interiors,
    }


PARETO_OBJECTIVES = (
    "rss_recovery",
    "message_containment",
    "clean_coherence",
    "flow_dimension",
    "seed_stability",
    "fixed_point_share",
)


def dominates(a: Dict[str, object], b: Dict[str, object]) -> bool:
    am = a["metrics"]
    bm = b["metrics"]
    assert isinstance(am, dict) and isinstance(bm, dict)
    no_worse = all(float(am[k]) >= float(bm[k]) for k in PARETO_OBJECTIVES)
    better = any(float(am[k]) > float(bm[k]) for k in PARETO_OBJECTIVES)
    return no_worse and better


def pareto_front(candidates: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = list(candidates)
    front = []
    for row in rows:
        if not any(dominates(other, row) for other in rows if other is not row):
            front.append(row)
    front.sort(
        key=lambda r: (
            float(r["metrics"]["rss_recovery"]),
            float(r["metrics"]["message_containment"]),
            float(r["metrics"]["seed_stability"]),
        ),
        reverse=True,
    )
    return front


def scalar_rank(row: Dict[str, object]) -> float:
    metrics = row["metrics"]
    assert isinstance(metrics, dict)
    return (
        2.0 * float(metrics["rss_recovery"])
        + 1.2 * float(metrics["message_containment"])
        + 0.8 * float(metrics["clean_coherence"])
        + 0.4 * float(metrics["flow_dimension"])
        + 0.6 * float(metrics["seed_stability"])
        + 0.5 * float(metrics["fixed_point_share"])
    )


def reproduce(
    frontier: List[Dict[str, object]],
    evaluated: List[Dict[str, object]],
    *,
    population_size: int,
    rng: random.Random,
    generation: int,
    mutation_sigma: float,
) -> List[Dict[str, float]]:
    ranked = sorted(evaluated, key=scalar_rank, reverse=True)
    parents = frontier[: max(2, min(len(frontier), population_size // 2))]
    if len(parents) < 2:
        parents = ranked[:2]

    next_population: List[Dict[str, float]] = []
    next_population.append(default_genome())
    for parent in parents[: max(1, population_size // 5)]:
        genome = parent["genome"]
        assert isinstance(genome, dict)
        next_population.append(dict(genome))

    while len(next_population) < population_size:
        if rng.random() < 0.30 and len(parents) >= 2:
            left, right = rng.sample(parents, 2)
            left_genome = left["genome"]
            right_genome = right["genome"]
            assert isinstance(left_genome, dict) and isinstance(right_genome, dict)
            child = crossover(left_genome, right_genome, rng)
        else:
            parent = rng.choice(parents)
            parent_genome = parent["genome"]
            assert isinstance(parent_genome, dict)
            child = mutate_genome(parent_genome, rng, mutation_sigma)
        if generation > 0 and rng.random() < 0.10:
            child = random_genome(rng)
        next_population.append(child)
    return next_population[:population_size]


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run diagnostic evolution over demian_native_v7.2 metabolism kwargs"
    )
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--eval-seeds", default="94,95,96")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--perturb-step", type=int, default=96)
    parser.add_argument("--perturb-scales", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--coupling-strengths", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--coupling-dim", type=int, default=8)
    parser.add_argument("--coupling-interval", type=int, default=16)
    parser.add_argument("--mutation-sigma", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-dir",
        default="data/evolution/v72_metabolism_probe",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    eval_seeds = parse_int_list(args.eval_seeds)
    perturb_scales = parse_float_list(args.perturb_scales)
    coupling_strengths = parse_float_list(args.coupling_strengths)

    config = {
        "population": args.population,
        "generations": args.generations,
        "seed": args.seed,
        "eval_seeds": eval_seeds,
        "hidden_size": args.hidden_size,
        "steps": args.steps,
        "perturb_step": args.perturb_step,
        "perturb_scales": perturb_scales,
        "coupling_strengths": coupling_strengths,
        "coupling_dim": args.coupling_dim,
        "coupling_interval": args.coupling_interval,
        "mutation_sigma": args.mutation_sigma,
        "device": args.device,
        "genes": {name: asdict(spec) for name, spec in GENES.items()},
        "pareto_objectives": list(PARETO_OBJECTIVES),
    }
    save_json(config, out_dir / "config.json")

    population = [default_genome()]
    while len(population) < args.population:
        population.append(random_genome(rng))

    all_rows: List[Dict[str, object]] = []
    generations_path = out_dir / "generations.jsonl"
    if generations_path.exists():
        generations_path.unlink()

    print("v7.2 metabolism diagnostic evolution")
    print(f"out_dir: {out_dir}")
    print(f"population={args.population} generations={args.generations} seeds={eval_seeds}")

    for generation in range(args.generations):
        evaluated: List[Dict[str, object]] = []
        generation_start = time.time()
        print()
        print(f"[generation {generation}]")
        for idx, genome in enumerate(population):
            candidate_id = f"gen{generation:03d}_candidate{idx:03d}"
            metrics = evaluate_genome(
                genome,
                hidden_size=args.hidden_size,
                steps=args.steps,
                seeds=eval_seeds,
                perturb_step=args.perturb_step,
                perturb_scales=perturb_scales,
                coupling_strengths=coupling_strengths,
                coupling_dim=args.coupling_dim,
                coupling_interval=args.coupling_interval,
                device=args.device,
            )
            row: Dict[str, object] = {
                "id": candidate_id,
                "generation": generation,
                "index": idx,
                "genome": genome,
                "metrics": metrics,
                "rank_score": scalar_rank({"metrics": metrics}),
            }
            evaluated.append(row)
            all_rows.append(row)
            save_json(row, candidates_dir / f"{candidate_id}.json")
            print(
                "  {id} rss={rss:.4f} msg_ratio={msg:.4f} coh={coh:.4f} flow={flow:.4f} stable={stable:.4f}".format(
                    id=candidate_id,
                    rss=float(metrics["rss_recovery"]),
                    msg=float(metrics["message_peak_ratio"]),
                    coh=float(metrics["clean_coherence"]),
                    flow=float(metrics["flow_dimension"]),
                    stable=float(metrics["seed_stability"]),
                )
            )

        generation_front = pareto_front(evaluated)
        global_front = pareto_front(all_rows)
        summary = {
            "generation": generation,
            "elapsed_seconds": time.time() - generation_start,
            "frontier_ids": [row["id"] for row in generation_front],
            "global_frontier_ids": [row["id"] for row in global_front],
            "best_rank_id": max(evaluated, key=scalar_rank)["id"],
            "best_rss_id": max(evaluated, key=lambda r: float(r["metrics"]["rss_recovery"]))["id"],
            "best_message_id": max(evaluated, key=lambda r: float(r["metrics"]["message_containment"]))["id"],
        }
        with generations_path.open("a") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
        save_json(generation_front, out_dir / f"frontier_gen{generation:03d}.json")
        save_json(global_front, out_dir / "frontier.json")
        print(
            "  frontier={} global_frontier={} best_rank={}".format(
                len(generation_front),
                len(global_front),
                summary["best_rank_id"],
            )
        )

        if generation < args.generations - 1:
            population = reproduce(
                generation_front,
                evaluated,
                population_size=args.population,
                rng=rng,
                generation=generation,
                mutation_sigma=args.mutation_sigma,
            )

    global_front = pareto_front(all_rows)
    best_by_objective = {
        objective: max(all_rows, key=lambda r: float(r["metrics"][objective]))
        for objective in PARETO_OBJECTIVES
    }
    save_json(global_front, out_dir / "frontier.json")
    save_json(best_by_objective, out_dir / "best_by_objective.json")
    print()
    print(f"Saved: {out_dir / 'frontier.json'}")
    print(f"Saved: {out_dir / 'best_by_objective.json'}")


if __name__ == "__main__":
    main()
