#!/usr/bin/env python3
"""Evolutionary archive for the v9 five-channel release substrate.

This search avoids human-oriented recovery targets. It preserves bounded,
non-collapsed machine morphologies, endogenous phase structure, channel
differentiation, and release as a geometric event in the substrate trajectory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import json
import math
import os
import random
import sys
import time
import multiprocessing
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legacy.demian_runtime.machine_observables import classify_attractor, compute_observables
from development.evolution.config import (
    BASE_EXCLUDED_SCALARS,
    CHANNELS,
    LOW_RANK_TARGETS,
    OBSERVABLE_KEYS,
    PARETO_OBJECTIVES,
    PHASE_CURVATURE_SCALE,
    PHASE_DELTA_SCALE,
    PHASE_TRANSITION_RANK_CAP,
    RELEASE_GEOMETRIC_RANK_CAP,
    RELEASE_TARGET_SCALARS,
)
from development.evolution.diagnostics import generation_diagnostics
from development.evolution.lineage import (
    lineage_child_entry,
    lineage_deltas,
    lineage_stability_score,
    merge_ancestor_ids,
    population_entry,
)
from development.evolution.scoring import (
    CAUSAL_MODE_COMBINED,
    CAUSAL_MODE_GATE_STATE,
    CAUSAL_MODE_ROUTE_RELEASE,
    CAUSAL_MODES,
    DEFAULT_CAUSAL_MODE,
    DEFAULT_NATIVE_OBJECTIVE,
    DYNAMIC_SELECTION_PROBE_RANK_MODE,
    ENGINEERED_TARGET_RANK_MODE,
    NATIVE_EMERGENCE_RANK_MODE,
    NATIVE_OBJECTIVES,
    NATIVE_OBJECTIVE_GATE_STATE,
    archive_bins,
    causal_divergence_components,
    pareto_front,
    rank_components,
    rare_release_score,
    scalar_rank,
)
from development.probe_v9_message_carrier_strange import ExperimentalV9MessageCarrier

CUDA_LOCK_DIR = Path(os.environ.get("DEMIAN_CUDA_LOCK_DIR", "/tmp/demian_cuda_locks"))


@dataclass(frozen=True)
class GeneSpec:
    low: float
    high: float
    default: float
    step: float


SCALAR_GENES: dict[str, GeneSpec] = {
    "init_scale": GeneSpec(0.04, 0.24, 0.15, 0.025),
    "state_gain": GeneSpec(1.05, 1.75, 1.40, 0.060),
    "slow_decay": GeneSpec(0.72, 0.985, 0.88, 0.030),
    "control_decay": GeneSpec(0.30, 0.92, 0.58, 0.060),
    "message_decay": GeneSpec(0.70, 0.995, 0.94, 0.035),
    "carrier_decay": GeneSpec(0.78, 0.998, 0.985, 0.018),
    "slow_readout_scale": GeneSpec(0.04, 0.42, 0.20, 0.040),
    "control_to_fast_scale": GeneSpec(0.05, 0.82, 0.50, 0.070),
    "fast_to_message_scale": GeneSpec(0.04, 0.72, 0.35, 0.060),
    "message_to_carrier_scale": GeneSpec(0.08, 0.95, 0.65, 0.070),
    "carrier_to_slow_scale": GeneSpec(0.06, 0.78, 0.45, 0.070),
    "message_to_fast_scale": GeneSpec(0.00, 0.44, 0.22, 0.050),
    "carrier_to_fast_scale": GeneSpec(0.00, 0.36, 0.16, 0.040),
    "message_readout_scale": GeneSpec(0.00, 0.26, 0.14, 0.025),
    "carrier_readout_scale": GeneSpec(0.00, 0.34, 0.22, 0.030),
    "release_threshold": GeneSpec(-0.40, 0.85, 0.12, 0.070),
    "release_temperature": GeneSpec(0.35, 1.80, 0.85, 0.110),
    "release_gain": GeneSpec(0.04, 0.46, 0.24, 0.045),
    "release_to_fast_scale": GeneSpec(0.00, 1.00, 1.00, 0.080),
    "release_to_slow_scale": GeneSpec(0.00, 1.00, 0.00, 0.080),
    "release_to_control_scale": GeneSpec(0.00, 1.00, 0.00, 0.080),
    "release_to_message_scale": GeneSpec(0.00, 1.00, 0.00, 0.080),
    "release_to_carrier_scale": GeneSpec(0.00, 1.00, 0.00, 0.080),
    "plastic_decay": GeneSpec(0.70, 0.995, 0.92, 0.030),
    "plastic_update_scale": GeneSpec(0.00, 0.12, 0.035, 0.014),
    "plastic_clip": GeneSpec(0.02, 0.32, 0.12, 0.025),
    "matrix_delta_scale": GeneSpec(0.00, 0.18, 0.045, 0.018),
}

REPRODUCTION_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("mutation", 0.50),
    ("causal_release_template_mutation", 0.28),
    ("delayed_eligibility_template_mutation", 0.15),
    ("release_phase_mutation", 0.04),
    ("crossover", 0.03),
)
NATIVE_REPRODUCTION_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("mutation", 0.70),
    ("crossover", 0.30),
)
RANDOM_INJECTION_RATE = 0.08


def clamp(value: float, spec: GeneSpec) -> float:
    return min(spec.high, max(spec.low, float(value)))


def target_dims(hidden_size: int) -> dict[str, tuple[int, int]]:
    return {
        "message_gate": (hidden_size, hidden_size),
        "carrier_gate": (hidden_size, hidden_size),
        "release_gate": (1, hidden_size * 5),
        "message_to_carrier": (hidden_size, hidden_size),
        "carrier_to_slow": (hidden_size, hidden_size),
    }


def default_scalars() -> dict[str, float]:
    return {name: spec.default for name, spec in SCALAR_GENES.items()}


def random_matrix_gene(
    rng: random.Random,
    out_dim: int,
    in_dim: int,
    rank: int,
    scale: float = 0.05,
) -> dict[str, list[list[float]]]:
    return {
        "u": [[rng.gauss(0.0, scale) for _ in range(rank)] for _ in range(out_dim)],
        "v": [[rng.gauss(0.0, scale) for _ in range(in_dim)] for _ in range(rank)],
        "obs": [[rng.gauss(0.0, scale) for _ in OBSERVABLE_KEYS] for _ in range(rank)],
    }


def default_genome(hidden_size: int, rank: int = 2) -> dict[str, Any]:
    rng = random.Random(1709 + hidden_size + rank)
    return {
        "scalars": default_scalars(),
        "matrices": {
            name: random_matrix_gene(rng, *dims, rank=rank, scale=0.015)
            for name, dims in target_dims(hidden_size).items()
        },
    }


def random_genome(rng: random.Random, hidden_size: int, rank: int = 2) -> dict[str, Any]:
    genome = default_genome(hidden_size, rank)
    genome["scalars"] = {
        name: rng.uniform(spec.low, spec.high)
        for name, spec in SCALAR_GENES.items()
    }
    genome["matrices"] = {
        name: random_matrix_gene(rng, *dims, rank=rank, scale=0.045)
        for name, dims in target_dims(hidden_size).items()
    }
    return repair_genome(genome, hidden_size, rank)


def repair_genome(genome: dict[str, Any], hidden_size: int, rank: int = 2) -> dict[str, Any]:
    scalars = dict(genome.get("scalars", {}))
    repaired = {
        "scalars": {
            name: clamp(scalars.get(name, spec.default), spec)
            for name, spec in SCALAR_GENES.items()
        },
        "matrices": {},
    }
    dims_by_name = target_dims(hidden_size)
    source_matrices = genome.get("matrices", {})
    rng = random.Random(2903)
    for name, (out_dim, in_dim) in dims_by_name.items():
        gene = source_matrices.get(name) or random_matrix_gene(rng, out_dim, in_dim, rank)
        repaired["matrices"][name] = {
            "u": fix_matrix(gene.get("u", []), out_dim, rank),
            "v": fix_matrix(gene.get("v", []), rank, in_dim),
            "obs": fix_matrix(gene.get("obs", []), rank, len(OBSERVABLE_KEYS)),
        }
    return repaired


def fix_matrix(values: list[list[float]], rows: int, cols: int) -> list[list[float]]:
    fixed = []
    for row_idx in range(rows):
        source = values[row_idx] if row_idx < len(values) else []
        fixed.append([
            float(source[col_idx]) if col_idx < len(source) else 0.0
            for col_idx in range(cols)
        ])
    return fixed


def mutate_genome(
    parent: dict[str, Any],
    rng: random.Random,
    hidden_size: int,
    rank: int,
    sigma: float,
) -> dict[str, Any]:
    child = json.loads(json.dumps(parent))
    for name, spec in SCALAR_GENES.items():
        if rng.random() < 0.82:
            child["scalars"][name] = clamp(
                child["scalars"][name] + rng.gauss(0.0, spec.step * sigma),
                spec,
            )
    for target in LOW_RANK_TARGETS:
        for key in ("u", "v", "obs"):
            for row in child["matrices"][target][key]:
                for idx in range(len(row)):
                    if rng.random() < 0.22:
                        row[idx] += rng.gauss(0.0, 0.018 * sigma)
    return repair_genome(child, hidden_size, rank)


def mutate_release_phase_genome(
    parent: dict[str, Any],
    rng: random.Random,
    hidden_size: int,
    rank: int,
    sigma: float,
) -> dict[str, Any]:
    """Bias mutation toward sparse, state-triggered release gates."""
    child = mutate_genome(parent, rng, hidden_size, rank, sigma)
    scalars = child["scalars"]
    scalars["release_threshold"] = clamp(
        scalars["release_threshold"] + abs(rng.gauss(0.0, 0.10 * sigma)),
        SCALAR_GENES["release_threshold"],
    )
    scalars["release_temperature"] = clamp(
        scalars["release_temperature"] - abs(rng.gauss(0.0, 0.08 * sigma)),
        SCALAR_GENES["release_temperature"],
    )
    scalars["release_gain"] = clamp(
        scalars["release_gain"] + rng.gauss(0.0, 0.035 * sigma),
        SCALAR_GENES["release_gain"],
    )
    scalars["plastic_update_scale"] = clamp(
        scalars["plastic_update_scale"] + abs(rng.gauss(0.0, 0.012 * sigma)),
        SCALAR_GENES["plastic_update_scale"],
    )
    release_gene = child["matrices"]["release_gate"]
    pressure_idx = OBSERVABLE_KEYS.index("release_pressure")
    surface_delta_idx = OBSERVABLE_KEYS.index("surface_delta")
    for row in release_gene["obs"]:
        row[pressure_idx] += rng.gauss(0.030 * sigma, 0.015 * sigma)
        row[surface_delta_idx] += rng.gauss(0.025 * sigma, 0.015 * sigma)
    return repair_genome(child, hidden_size, rank)


CAUSAL_RELEASE_ROUTE_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "name": "distributed_inband_release",
        "threshold": 0.24,
        "temperature": 0.58,
        "gain": 0.34,
        "routes": (0.16, 0.24, 0.18, 0.22, 0.20),
        "obs": {
            "message_norm": 0.020,
            "carrier_norm": 0.020,
            "message_carrier_gap": -0.012,
            "release_pressure": 0.080,
            "surface_delta": 0.060,
        },
    },
    {
        "name": "message_carrier_release",
        "threshold": 0.30,
        "temperature": 0.52,
        "gain": 0.38,
        "routes": (0.08, 0.16, 0.14, 0.32, 0.30),
        "obs": {
            "message_norm": 0.030,
            "carrier_norm": 0.030,
            "message_carrier_gap": -0.018,
            "release_pressure": 0.090,
            "surface_delta": 0.050,
        },
    },
    {
        "name": "control_slow_release",
        "threshold": 0.18,
        "temperature": 0.68,
        "gain": 0.30,
        "routes": (0.10, 0.34, 0.24, 0.16, 0.16),
        "obs": {
            "slow_norm": 0.018,
            "control_norm": 0.018,
            "release_pressure": 0.070,
            "surface_delta": 0.075,
        },
    },
)

DELAYED_ELIGIBILITY_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "name": "slow_carrier_delayed_gate",
        "threshold": 0.68,
        "temperature": 0.82,
        "message_decay": 0.91,
        "carrier_decay": 0.992,
        "obs": {
            "time_since_perturbation": 0.090,
            "perturbation_magnitude": 0.060,
            "surface_delta": 0.050,
            "release_pressure": 0.060,
        },
    },
    {
        "name": "high_threshold_delayed_pressure",
        "threshold": 0.76,
        "temperature": 0.74,
        "message_decay": 0.88,
        "carrier_decay": 0.996,
        "obs": {
            "time_since_perturbation": 0.110,
            "perturbation_magnitude": 0.050,
            "surface_delta": 0.040,
            "release_pressure": 0.075,
        },
    },
    {
        "name": "moderate_delayed_surface_gate",
        "threshold": 0.62,
        "temperature": 0.96,
        "message_decay": 0.93,
        "carrier_decay": 0.988,
        "obs": {
            "time_since_perturbation": 0.080,
            "perturbation_magnitude": 0.070,
            "surface_delta": 0.065,
            "release_pressure": 0.045,
        },
    },
)


def mutate_causal_release_template_genome(
    parent: dict[str, Any],
    rng: random.Random,
    hidden_size: int,
    rank: int,
    sigma: float,
) -> dict[str, Any]:
    """Jointly mutate the causal release gate and routing bundle."""
    child = mutate_genome(parent, rng, hidden_size, rank, sigma)
    scalars = child["scalars"]
    template = rng.choice(CAUSAL_RELEASE_ROUTE_TEMPLATES)
    blend = rng.uniform(0.45, 0.80)
    scalar_targets = {
        "release_threshold": template["threshold"],
        "release_temperature": template["temperature"],
        "release_gain": template["gain"],
    }
    for name, target in scalar_targets.items():
        spec = SCALAR_GENES[name]
        jitter = rng.gauss(0.0, spec.step * 0.35 * sigma)
        scalars[name] = clamp((1.0 - blend) * scalars[name] + blend * float(target) + jitter, spec)

    route_values = []
    for channel, target in zip(CHANNELS, template["routes"]):
        name = f"release_to_{channel}_scale"
        spec = SCALAR_GENES[name]
        jitter = rng.gauss(0.0, spec.step * 0.25 * sigma)
        value = clamp((1.0 - blend) * scalars[name] + blend * float(target) + jitter, spec)
        route_values.append(max(0.0, value))
    route_total = sum(route_values)
    if route_total <= 1e-8:
        route_values[0] = 1.0
        route_total = 1.0
    for channel, value in zip(CHANNELS, route_values):
        scalars[f"release_to_{channel}_scale"] = clamp(
            value / route_total,
            SCALAR_GENES[f"release_to_{channel}_scale"],
        )

    release_obs = child["matrices"]["release_gate"]["obs"]
    for row in release_obs:
        for observable, target in template["obs"].items():
            idx = OBSERVABLE_KEYS.index(observable)
            row[idx] += rng.gauss(float(target) * sigma, abs(float(target)) * 0.25 * sigma)
        pressure_idx = OBSERVABLE_KEYS.index("release_pressure")
        surface_delta_idx = OBSERVABLE_KEYS.index("surface_delta")
        row[pressure_idx] = max(row[pressure_idx], 0.035 * sigma)
        row[surface_delta_idx] = max(row[surface_delta_idx], 0.030 * sigma)
    return repair_genome(child, hidden_size, rank)


def mutate_delayed_eligibility_template_genome(
    parent: dict[str, Any],
    rng: random.Random,
    hidden_size: int,
    rank: int,
    sigma: float,
) -> dict[str, Any]:
    """Bias release timing toward post-perturbation delayed eligibility."""
    child = mutate_genome(parent, rng, hidden_size, rank, sigma)
    scalars = child["scalars"]
    template = rng.choice(DELAYED_ELIGIBILITY_TEMPLATES)
    blend = rng.uniform(0.50, 0.82)
    scalar_template_keys = {
        "release_threshold": "threshold",
        "release_temperature": "temperature",
        "message_decay": "message_decay",
        "carrier_decay": "carrier_decay",
    }
    for name, template_key in scalar_template_keys.items():
        spec = SCALAR_GENES[name]
        jitter = rng.gauss(0.0, spec.step * 0.30 * sigma)
        scalars[name] = clamp((1.0 - blend) * scalars[name] + blend * float(template[template_key]) + jitter, spec)

    release_obs = child["matrices"]["release_gate"]["obs"]
    floor_targets = {
        "time_since_perturbation": 0.045,
        "perturbation_magnitude": 0.035,
        "surface_delta": 0.030,
        "release_pressure": 0.035,
    }
    for row in release_obs:
        for observable, target in template["obs"].items():
            idx = OBSERVABLE_KEYS.index(observable)
            row[idx] += rng.gauss(float(target) * sigma, abs(float(target)) * 0.25 * sigma)
        for observable, floor in floor_targets.items():
            idx = OBSERVABLE_KEYS.index(observable)
            row[idx] = max(row[idx], floor * sigma)

    route_total = sum(max(0.0, scalars[name]) for name in RELEASE_TARGET_SCALARS)
    if route_total <= 1e-8:
        scalars["release_to_fast_scale"] = 1.0
    return repair_genome(child, hidden_size, rank)


def crossover(
    left: dict[str, Any],
    right: dict[str, Any],
    rng: random.Random,
    hidden_size: int,
    rank: int,
) -> dict[str, Any]:
    child = {"scalars": {}, "matrices": {}}
    for name in SCALAR_GENES:
        child["scalars"][name] = (
            left["scalars"][name] if rng.random() < 0.5 else right["scalars"][name]
        )
    for target in LOW_RANK_TARGETS:
        child["matrices"][target] = {}
        for key in ("u", "v", "obs"):
            child["matrices"][target][key] = (
                left["matrices"][target][key]
                if rng.random() < 0.5
                else right["matrices"][target][key]
            )
    return repair_genome(child, hidden_size, rank)


class AdaptiveV9FiveChannel(ExperimentalV9MessageCarrier):
    """v9-5ch with bounded low-rank runtime deltas on selected gates/routes."""

    def __init__(
        self,
        hidden_size: int,
        genome: dict[str, Any],
        rank: int = 2,
        *,
        release_routes_disabled: bool = False,
    ):
        scalars = dict(genome["scalars"])
        super().__init__(
            hidden_size,
            release_policy="learned",
            binding_start_step=1,
            initial_message_scale=0.0,
            initial_carrier_scale=0.0,
            **{
                key: scalars[key]
                for key in scalars
                if key not in BASE_EXCLUDED_SCALARS
            },
        )
        self.rank = rank
        self.plastic_decay = scalars["plastic_decay"]
        self.plastic_update_scale = scalars["plastic_update_scale"]
        self.plastic_clip = scalars["plastic_clip"]
        self.matrix_delta_scale = scalars["matrix_delta_scale"]
        self.release_routes_disabled = bool(release_routes_disabled)
        if self.release_routes_disabled:
            self.release_target_mix = torch.zeros(len(RELEASE_TARGET_SCALARS), dtype=torch.float32)
        else:
            raw_release_targets = torch.tensor(
                [scalars[name] for name in RELEASE_TARGET_SCALARS],
                dtype=torch.float32,
            )
            total_release_target = float(raw_release_targets.sum().item())
            if total_release_target <= 1e-8:
                raw_release_targets[0] = 1.0
                total_release_target = 1.0
            self.release_target_mix = raw_release_targets / total_release_target
        self.low_rank = {}
        for name, gene in genome["matrices"].items():
            self.low_rank[name] = {
                "u": torch.tensor(gene["u"], dtype=torch.float32),
                "v": torch.tensor(gene["v"], dtype=torch.float32),
                "obs": torch.tensor(gene["obs"], dtype=torch.float32),
            }
        self.plastic_state: dict[str, torch.Tensor] = {}
        self._prev_surface: torch.Tensor | None = None

    def to(self, *args: Any, **kwargs: Any):
        module = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for parts in self.low_rank.values():
            for key, value in parts.items():
                parts[key] = value.to(device=device, dtype=dtype)
        self.release_target_mix = self.release_target_mix.to(device=device, dtype=dtype)
        return module

    def record_perturbation(self, step: int, magnitude: float) -> None:
        self._perturb_step = int(step)
        self._perturb_magnitude = float(abs(magnitude))

    def initial_state(self, batch_size: int, device: torch.device):
        self.plastic_state = {
            name: torch.zeros(self.rank, device=device) for name in LOW_RANK_TARGETS
        }
        self._prev_surface = None
        self._perturb_step: int | None = None
        self._perturb_magnitude = 0.0
        return super().initial_state(batch_size, device)

    def _observe(self, fast, slow, control, message, carrier) -> torch.Tensor:
        pressure = (
            torch.norm(message, dim=-1).mean() + torch.norm(carrier, dim=-1).mean()
        ) / math.sqrt(max(self.hidden_size, 1))
        surface = self.state_vector((fast, slow, control, message, carrier)).detach()
        if self._prev_surface is None:
            surface_delta = torch.zeros((), device=fast.device, dtype=fast.dtype)
        else:
            surface_delta = torch.norm(surface - self._prev_surface) / math.sqrt(max(self.hidden_size, 1))
        self._prev_surface = surface.clone()
        if self._perturb_step is None or self._step_index <= self._perturb_step:
            time_since_perturbation = torch.zeros((), device=fast.device, dtype=fast.dtype)
            perturbation_magnitude = torch.zeros((), device=fast.device, dtype=fast.dtype)
        else:
            elapsed = float(self._step_index - self._perturb_step)
            time_since_perturbation = torch.tensor(
                min(1.0, elapsed / 32.0),
                device=fast.device,
                dtype=fast.dtype,
            )
            perturbation_magnitude = torch.tensor(
                self._perturb_magnitude * math.exp(-elapsed / 32.0),
                device=fast.device,
                dtype=fast.dtype,
            )
        values = [
            torch.norm(fast) / math.sqrt(max(fast.numel(), 1)),
            torch.norm(slow) / math.sqrt(max(slow.numel(), 1)),
            torch.norm(message) / math.sqrt(max(message.numel(), 1)),
            torch.norm(carrier) / math.sqrt(max(carrier.numel(), 1)),
            torch.norm(control) / math.sqrt(max(control.numel(), 1)),
            torch.abs(torch.norm(message) - torch.norm(carrier)) / math.sqrt(max(message.numel(), 1)),
            pressure,
            surface_delta,
            time_since_perturbation,
            perturbation_magnitude,
        ]
        return torch.stack([value.to(device=fast.device, dtype=fast.dtype) for value in values])

    def _update_plastic(self, obs: torch.Tensor) -> None:
        for name in LOW_RANK_TARGETS:
            drive = self.low_rank[name]["obs"] @ obs
            state = self.plastic_decay * self.plastic_state[name] + self.plastic_update_scale * torch.tanh(drive)
            self.plastic_state[name] = torch.clamp(state, -self.plastic_clip, self.plastic_clip)

    def _delta(self, name: str) -> torch.Tensor:
        parts = self.low_rank[name]
        static = self.matrix_delta_scale * (parts["u"] @ parts["v"])
        dynamic = parts["u"] @ torch.diag(self.plastic_state[name]) @ parts["v"]
        return static + dynamic

    def _linear_with_delta(self, layer: torch.nn.Linear, x: torch.Tensor, name: str) -> torch.Tensor:
        return layer(x) + torch.nn.functional.linear(x, self._delta(name))

    def _fit_release_to(self, release_vector: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if release_vector.shape[-1] == target.shape[-1]:
            return release_vector
        if release_vector.shape[-1] > target.shape[-1]:
            return release_vector[..., : target.shape[-1]]
        pad = target.shape[-1] - release_vector.shape[-1]
        return torch.nn.functional.pad(release_vector, (0, pad))

    def step(self, state):
        self._step_index += 1
        fast, slow, control, message, carrier = state
        self._update_plastic(self._observe(fast, slow, control, message, carrier))

        message_gate = torch.sigmoid(self._linear_with_delta(self.message_gate, message, "message_gate"))
        message_candidate = torch.tanh(self.message_mix(message))
        fast_to_message = self.fast_to_message_scale * torch.tanh(self.fast_to_message(fast))
        new_message = self.message_decay * message + message_gate * (message_candidate + fast_to_message)

        carrier_gate = torch.sigmoid(self._linear_with_delta(self.carrier_gate, carrier, "carrier_gate"))
        carrier_candidate = torch.tanh(self.carrier_mix(carrier))
        message_to_carrier = self.message_to_carrier_scale * torch.tanh(
            self.message_to_carrier(new_message)
            + torch.nn.functional.linear(new_message, self._delta("message_to_carrier"))
        )
        fast_to_carrier = self.fast_to_carrier_scale * torch.tanh(self.fast_to_carrier(fast))
        new_carrier = self.carrier_decay * carrier + carrier_gate * (
            carrier_candidate + message_to_carrier + fast_to_carrier
        )

        slow_gate = torch.sigmoid(self.slow_gate(slow))
        slow_candidate = torch.tanh(self.slow_mix(slow))
        fast_to_slow_bias = self.slow_readout_scale * torch.tanh(self.fast_to_slow(fast))
        carrier_to_slow = self.carrier_to_slow_scale * torch.tanh(
            self.carrier_to_slow(new_carrier)
            + torch.nn.functional.linear(new_carrier, self._delta("carrier_to_slow"))
        )
        new_slow = self.slow_decay * slow + slow_gate * (
            slow_candidate + fast_to_slow_bias + carrier_to_slow
        )

        control_gate = torch.sigmoid(self.control_gate(control))
        control_candidate = torch.tanh(self.control_mix(control))
        fast_slow_bias = torch.tanh(self.fast_slow_to_control(torch.cat([fast, slow], dim=-1)))
        new_control = self.control_decay * control + control_gate * (control_candidate + fast_slow_bias)

        fast_gate = torch.sigmoid(self.fast_gate(fast))
        fast_candidate = torch.tanh(self.fast_mix(fast))
        control_bias = self.control_to_fast_scale * torch.tanh(self.control_readout(new_control))
        message_fast_bias = self.message_to_fast_scale * torch.tanh(self.message_to_fast(new_message))
        carrier_fast_bias = self.carrier_to_fast_scale * torch.tanh(self.carrier_to_fast(new_carrier))
        release_open, release_strength, release_pressure, release_drive = self._release_signal(
            fast,
            new_slow,
            new_control,
            new_message,
            new_carrier,
        )
        release_vector = release_strength * torch.tanh(
            self.message_to_fast(new_message) + self.carrier_to_fast(new_carrier)
        )
        if self.release_routes_disabled:
            release_bias = torch.zeros_like(fast)
            slow_release_bias = torch.zeros_like(new_slow)
            control_release_bias = torch.zeros_like(new_control)
            message_release_bias = torch.zeros_like(new_message)
            carrier_release_bias = torch.zeros_like(new_carrier)
        else:
            release_bias = self.release_target_mix[0] * self._fit_release_to(release_vector, fast)
            slow_release_bias = self.release_target_mix[1] * self._fit_release_to(release_vector, new_slow)
            control_release_bias = self.release_target_mix[2] * self._fit_release_to(release_vector, new_control)
            message_release_bias = self.release_target_mix[3] * self._fit_release_to(release_vector, new_message)
            carrier_release_bias = self.release_target_mix[4] * self._fit_release_to(release_vector, new_carrier)
        new_slow = new_slow + slow_release_bias
        new_control = new_control + control_release_bias
        new_message = new_message + message_release_bias
        new_carrier = new_carrier + carrier_release_bias
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(
            fast_candidate + control_bias + message_fast_bias + carrier_fast_bias + release_bias
        )

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "control_write_mean": float(control_gate.mean().item()),
            "message_write_mean": float(message_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "fast_to_slow_bias_norm": float(torch.norm(fast_to_slow_bias).item()),
            "fast_slow_bias_norm": float(torch.norm(fast_slow_bias).item()),
            "control_bias_norm": float(torch.norm(control_bias).item()),
            "fast_to_message_norm": float(torch.norm(fast_to_message).item()),
            "message_to_carrier_norm": float(torch.norm(message_to_carrier).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "message_to_fast_norm": float(torch.norm(message_fast_bias).item()),
            "carrier_to_fast_norm": float(torch.norm(carrier_fast_bias).item()),
            "release_open": float(release_open.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "release_pressure_mean": float(release_pressure.mean().item()),
            "release_drive_mean": float(release_drive.mean().item()),
            "release_bias_norm": float(torch.norm(release_vector).item()),
            "release_routes_disabled": float(self.release_routes_disabled),
            "release_to_fast_norm": float(torch.norm(release_bias).item()),
            "release_to_slow_norm": float(torch.norm(slow_release_bias).item()),
            "release_to_control_norm": float(torch.norm(control_release_bias).item()),
            "release_to_message_norm": float(torch.norm(message_release_bias).item()),
            "release_to_carrier_norm": float(torch.norm(carrier_release_bias).item()),
            "endogenous_release_norm": float(torch.norm(release_vector).item()),
            "carrier_residual_norm": float(torch.norm(self.carrier_decay * carrier).item()),
            "plastic_state_norm": float(
                sum(torch.norm(state).item() for state in self.plastic_state.values())
            ),
            "binding_active": 1.0,
        }
        return new_fast, new_slow, new_control, new_message, new_carrier

    def _release_signal(self, fast, slow, control, message, carrier):
        if self.release_gain <= 0.0:
            zeros = torch.zeros((message.shape[0], 1), device=message.device)
            pressure = (
                torch.norm(message, dim=-1, keepdim=True)
                + torch.norm(carrier, dim=-1, keepdim=True)
            ) / math.sqrt(max(self.hidden_size, 1))
            return zeros, zeros, pressure, zeros
        control_view = torch.tanh(self.control_readout(control))
        gate_input = torch.cat([fast, slow, control_view, message, carrier], dim=-1)
        drive = self._linear_with_delta(self.release_gate, gate_input, "release_gate")
        pressure = (
            torch.norm(message, dim=-1, keepdim=True)
            + torch.norm(carrier, dim=-1, keepdim=True)
        ) / math.sqrt(max(self.hidden_size, 1))
        open_gate = torch.sigmoid((drive + 0.04 * pressure - self.release_threshold) / self.release_temperature)
        strength = self.release_gain * torch.relu(open_gate - 0.5)
        return open_gate, strength, pressure, drive


def motif_vector(kind: str, hidden_size: int, seed: int, index: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 997 * (index + 1))
    vec = torch.zeros(hidden_size)
    if kind == "basis":
        axis = index % hidden_size
        vec[axis] = -1.0 if (index // hidden_size) % 2 else 1.0
    elif kind == "block":
        width = max(2, hidden_size // 6)
        start = (index * width) % hidden_size
        vec[start : min(hidden_size, start + width)] = 1.0
    elif kind == "sine":
        x = torch.linspace(0.0, 2.0 * math.pi, hidden_size)
        vec = torch.sin(x * float(index + 1))
    elif kind == "sparse":
        count = max(2, hidden_size // 8)
        idx = torch.randperm(hidden_size, generator=gen)[:count]
        vec[idx] = torch.randn(count, generator=gen)
    elif kind == "gaussian":
        vec = torch.randn(hidden_size, generator=gen)
    else:
        raise ValueError(f"unknown motif kind: {kind}")
    norm = vec.norm()
    return vec / norm if norm > 1e-10 else vec


def motif_suite(hidden_size: int, seed: int) -> list[dict[str, Any]]:
    motifs = []
    for idx in range(min(4, hidden_size)):
        motifs.append({"family": "basis", "index": idx, "vector": motif_vector("basis", hidden_size, seed, idx)})
    for family in ("block", "sine", "sparse", "gaussian"):
        for idx in range(2):
            motifs.append({"family": family, "index": idx, "vector": motif_vector(family, hidden_size, seed, idx)})
    return motifs


def perturb_signature(
    state: tuple[torch.Tensor, ...],
    signature: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    fast, slow, control, message, carrier = state
    base = signature.to(device=fast.device, dtype=fast.dtype).view_as(fast)
    if mode == "external":
        vec = base
    elif mode == "state_velocity":
        vec = fast - slow
    elif mode == "carrier_pressure":
        vec = carrier + message
    elif mode == "control_shadow":
        vec = torch.tanh(control)
    elif mode == "trajectory_orthogonal":
        ref = torch.tanh(fast + slow + message + carrier)
        denom = torch.sum(base * ref, dim=-1, keepdim=True) / (
            torch.sum(ref * ref, dim=-1, keepdim=True) + 1e-8
        )
        vec = base - denom * ref
    else:
        raise ValueError(f"unknown perturbation mode: {mode}")
    norm = torch.norm(vec, dim=-1, keepdim=True)
    return vec / (norm + 1e-8)


def fit_channel_vector(vector: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] == target.shape[-1]:
        return vector
    if vector.shape[-1] > target.shape[-1]:
        return vector[..., : target.shape[-1]]
    return torch.nn.functional.pad(vector, (0, target.shape[-1] - vector.shape[-1]))


def apply_signature(
    state: tuple[torch.Tensor, ...],
    signature: torch.Tensor,
    scale: float,
    *,
    channel: str = "fast",
    mode: str = "external",
) -> tuple[torch.Tensor, ...]:
    fast, slow, control, message, carrier = state
    sig = scale * perturb_signature(state, signature, mode)
    if channel == "fast":
        return fast + fit_channel_vector(sig, fast), slow, control, message, carrier
    if channel == "slow":
        return fast, slow + fit_channel_vector(sig, slow), control, message, carrier
    if channel == "control":
        return fast, slow, control + fit_channel_vector(sig, control), message, carrier
    if channel == "message":
        return fast, slow, control, message + fit_channel_vector(sig, message), carrier
    if channel == "carrier":
        return fast, slow, control, message, carrier + fit_channel_vector(sig, carrier)
    if channel == "all":
        distributed = sig / math.sqrt(5.0)
        return (
            fast + fit_channel_vector(distributed, fast),
            slow + fit_channel_vector(distributed, slow),
            control + fit_channel_vector(distributed, control),
            message + fit_channel_vector(distributed, message),
            carrier + fit_channel_vector(distributed, carrier),
        )
    raise ValueError(f"unknown perturbation channel: {channel}")


def cosine_to_signature(vec: torch.Tensor, signature: torch.Tensor) -> float:
    sig = signature.to(device=vec.device, dtype=vec.dtype).view(-1)
    flat = vec.view(-1)
    denom = flat.norm() * sig.norm()
    if float(denom) <= 1e-10:
        return 0.0
    return float(torch.dot(flat, sig) / denom)


def run_with_motif(
    genome: dict[str, Any],
    *,
    hidden_size: int,
    steps: int,
    seed: int,
    perturb_step: int,
    perturb_scale: float,
    motif: dict[str, Any],
    rank: int,
    device: str,
    perturb_channel: str = "fast",
    perturb_mode: str = "external",
    release_routes_disabled: bool = False,
    capture_channel_states: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AdaptiveV9FiveChannel(
        hidden_size,
        genome,
        rank=rank,
        release_routes_disabled=release_routes_disabled,
    ).to(device=device)
    state = model.initial_state(1, torch.device(device))
    signature = motif["vector"]
    trajectory = []
    surfaces: list[torch.Tensor] = []
    fasts: list[torch.Tensor] = []
    prev_surface: torch.Tensor | None = None
    prev_vel: torch.Tensor | None = None
    window: list[dict[str, Any]] = []
    residual_buffer: list[torch.Tensor] = []
    prev_components: dict[str, torch.Tensor] | None = None
    with torch.no_grad():
        for step in range(1, steps + 1):
            if step == perturb_step and perturb_mode != "clean":
                state = apply_signature(
                    state,
                    signature,
                    perturb_scale,
                    channel=perturb_channel,
                    mode=perturb_mode,
                )
                model.record_perturbation(step, perturb_scale)
            state = model.step(state)
            surface = model.state_vector(state).view(-1).detach().float().cpu()
            components = {
                name: tensor.view(-1).detach().float().cpu()
                for name, tensor in model.state_components(state).items()
            }
            fasts.append(components["fast"].clone())
            surfaces.append(surface.clone())
            if prev_surface is None:
                delta_vec = torch.zeros_like(surface)
                residual_delta = 0.0
                coherence = 1.0
            else:
                delta_vec = surface - prev_surface
                residual_delta = float(delta_vec.norm()) / math.sqrt(max(surface.shape[0], 1))
                coherence = float(torch.nn.functional.cosine_similarity(surface, prev_surface, dim=0))
            if prev_vel is None:
                velocity_align = 1.0
            else:
                denom = float(prev_vel.norm() * delta_vec.norm())
                velocity_align = float(torch.dot(prev_vel, delta_vec) / denom) if denom > 1e-10 else 0.0

            aux = model.step_aux()
            message = components.get("message", torch.zeros_like(surface))
            carrier = components.get("carrier", torch.zeros_like(surface))
            slow = components.get("slow", torch.zeros_like(surface))
            control = components.get("control", torch.zeros_like(surface))
            fast = components.get("fast", surface)
            row = {
                "step": step,
                "residual_norm": float(surface.norm()) / math.sqrt(max(surface.shape[0], 1)),
                "residual_delta": residual_delta,
                "temporal_coherence": coherence,
                "velocity_align": velocity_align,
                "spectral_centroid": 0.0,
                "spectral_concentration": 0.0,
                "layer_work_ratio": 0.5,
                "fast_state_norm": float(fast.norm()) / math.sqrt(max(fast.shape[0], 1)),
                "slow_state_norm": float(slow.norm()) / math.sqrt(max(slow.shape[0], 1)),
                "message_state_norm": float(message.norm()) / math.sqrt(max(message.shape[0], 1)),
                "carrier_state_norm": float(carrier.norm()) / math.sqrt(max(carrier.shape[0], 1)),
                "message_signature_cosine": cosine_to_signature(message, signature),
                "carrier_signature_cosine": cosine_to_signature(carrier, signature),
                "fast_signature_cosine": cosine_to_signature(fast, signature),
                "route_metrics": dict(aux),
            }
            if capture_channel_states:
                row["_channel_states"] = {
                    name: tensor.clone() for name, tensor in components.items()
                }
            if prev_components is not None:
                row["fast_state_delta"] = component_delta(fast, prev_components.get("fast"))
                row["slow_state_delta"] = component_delta(slow, prev_components.get("slow"))
                row["control_state_delta"] = component_delta(control, prev_components.get("control"))
                row["message_state_delta"] = component_delta(message, prev_components.get("message"))
                row["carrier_state_delta"] = component_delta(carrier, prev_components.get("carrier"))
            else:
                row["fast_state_delta"] = 0.0
                row["slow_state_delta"] = 0.0
                row["control_state_delta"] = 0.0
                row["message_state_delta"] = 0.0
                row["carrier_state_delta"] = 0.0
            trajectory.append(row)
            window.append(row)
            if len(window) > 20:
                window.pop(0)
            residual_buffer.append(surface.clone())
            if len(residual_buffer) > 16:
                residual_buffer.pop(0)
            prev_surface = surface.clone()
            prev_vel = delta_vec.clone()
            prev_components = {name: tensor.clone() for name, tensor in components.items()}

    obs = compute_observables(window, surfaces[-1], residual_buffer)
    attractor = classify_attractor(obs)
    metrics = path_metrics(
        trajectory,
        surfaces,
        fasts,
        signature,
        perturb_step=perturb_step,
    )
    regime = classify_regime(attractor.type, trajectory, metrics)
    summary = {
        "attractor_type": attractor.type,
        "attractor_confidence": float(attractor.stability),
        "regime_class": regime,
        "mean_norm": float(mean(row["residual_norm"] for row in trajectory)),
        "mean_delta": float(mean(row["residual_delta"] for row in trajectory)),
        "mean_coherence": float(mean(row["temporal_coherence"] for row in trajectory)),
        "covariance_rank": float(obs["covariance_rank"]),
        "flow_dimension": float(obs["flow_dimension"]),
        "compression_ratio": float(obs["compression_ratio"]),
    }
    return {
        "seed": seed,
        "motif_family": motif["family"],
        "motif_index": motif["index"],
        "perturb_scale": perturb_scale,
        "perturb_channel": perturb_channel,
        "perturb_mode": perturb_mode,
        "summary": summary,
        "metrics": metrics,
        "trajectory": trajectory,
    }


def strip_runtime_channel_states(run: dict[str, Any]) -> dict[str, Any]:
    """Remove in-memory tensors used only for paired causal measurement."""
    for row in run.get("trajectory", []):
        row.pop("_channel_states", None)
    return run


def release_anchor_steps(run: dict[str, Any]) -> list[int]:
    return [
        int(row["step"])
        for row in run.get("trajectory", [])
        if float(row.get("route_metrics", {}).get("release_strength_mean", 0.0)) > 0.0
    ]


def gain_zero_diagnostics(run: dict[str, Any], *, tolerance: float = 1e-8) -> dict[str, Any]:
    """Check that the gain-zero intervention actually removes release output."""
    max_strength = 0.0
    max_route_norm = 0.0
    max_endogenous_norm = 0.0
    route_keys = [f"release_to_{channel}_norm" for channel in CHANNELS]
    for row in run.get("trajectory", []):
        route_metrics = row.get("route_metrics", {})
        max_strength = max(max_strength, abs(float(route_metrics.get("release_strength_mean", 0.0))))
        max_endogenous_norm = max(
            max_endogenous_norm,
            abs(float(route_metrics.get("endogenous_release_norm", 0.0))),
        )
        for key in route_keys:
            max_route_norm = max(max_route_norm, abs(float(route_metrics.get(key, 0.0))))
    own_anchor_count = len(release_anchor_steps(run))
    clean = (
        max_strength <= tolerance
        and max_route_norm <= tolerance
        and max_endogenous_norm <= tolerance
        and own_anchor_count == 0
    )
    return {
        "gain_zero_clean": clean,
        "gain_zero_max_release_strength": max_strength,
        "gain_zero_max_route_norm": max_route_norm,
        "gain_zero_max_endogenous_release_norm": max_endogenous_norm,
        "gain_zero_own_release_anchor_count": own_anchor_count,
        "gain_zero_tolerance": tolerance,
    }


def causal_release_metrics(
    original: dict[str, Any],
    ablated: dict[str, Any],
    *,
    anchor_offset: int = 4,
) -> dict[str, float]:
    """Measure route-causal channel divergence at original release anchors."""
    original_rows = {
        int(row["step"]): row
        for row in original.get("trajectory", [])
        if "_channel_states" in row
    }
    ablated_rows = {
        int(row["step"]): row
        for row in ablated.get("trajectory", [])
        if "_channel_states" in row
    }
    values_by_channel: dict[str, list[float]] = {channel: [] for channel in CHANNELS}
    for step in release_anchor_steps(original):
        target_step = step + anchor_offset
        left = original_rows.get(target_step)
        right = ablated_rows.get(target_step)
        if left is None or right is None:
            continue
        for channel in CHANNELS:
            left_tensor = left["_channel_states"].get(channel)
            right_tensor = right["_channel_states"].get(channel)
            if left_tensor is None or right_tensor is None:
                continue
            delta = torch.norm(left_tensor - right_tensor) / math.sqrt(max(left_tensor.numel(), 1))
            left_scale = torch.norm(left_tensor) / math.sqrt(max(left_tensor.numel(), 1))
            right_scale = torch.norm(right_tensor) / math.sqrt(max(right_tensor.numel(), 1))
            scale = 0.5 * (left_scale + right_scale) + 1e-8
            values_by_channel[channel].append(float(delta / scale))
    channel_means = {
        f"release_causal_divergence_{channel}": mean_or_zero(values)
        for channel, values in values_by_channel.items()
    }
    return {
        "release_causal_divergence": mean_or_zero(channel_means.values()),
        "release_causal_anchor_count": float(max((len(values) for values in values_by_channel.values()), default=0)),
        **channel_means,
    }


def release_gain_zero_genome(genome: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(genome))
    clone.setdefault("scalars", {})["release_gain"] = 0.0
    return clone


def paired_causal_run(
    genome: dict[str, Any],
    *,
    hidden_size: int,
    steps: int,
    seed: int,
    perturb_step: int,
    perturb_scale: float,
    motif: dict[str, Any],
    rank: int,
    device: str,
    perturb_channel: str = "fast",
    perturb_mode: str = "external",
) -> dict[str, Any]:
    """Run original, route-disabled, and gain-zero variants under one condition."""
    common = {
        "hidden_size": hidden_size,
        "steps": steps,
        "seed": seed,
        "perturb_step": perturb_step,
        "perturb_scale": perturb_scale,
        "motif": motif,
        "rank": rank,
        "device": device,
        "perturb_channel": perturb_channel,
        "perturb_mode": perturb_mode,
        "capture_channel_states": True,
    }
    original = run_with_motif(genome, **common)
    routes_disabled = run_with_motif(genome, release_routes_disabled=True, **common)
    gain_zero = run_with_motif(release_gain_zero_genome(genome), **common)

    routes_metrics = causal_release_metrics(original, routes_disabled, anchor_offset=4)
    gain_zero_metrics = causal_release_metrics(original, gain_zero, anchor_offset=4)
    gain_zero_checks = gain_zero_diagnostics(gain_zero)
    original["metrics"].update(routes_metrics)
    original["metrics"].update(
        {
            f"release_gain_zero_{key}": value
            for key, value in gain_zero_metrics.items()
            if key.startswith("release_causal")
        }
    )
    original["metrics"].update(gain_zero_checks)
    for ablation, run in (
        ("original", original),
        ("release_routes_disabled", routes_disabled),
        ("release_gain_zero", gain_zero),
    ):
        run["ablation"] = ablation
        strip_runtime_channel_states(run)
    return {"runs": [original, routes_disabled, gain_zero]}


def component_delta(current: torch.Tensor, previous: torch.Tensor | None) -> float:
    if previous is None:
        return 0.0
    return float(torch.norm(current - previous)) / math.sqrt(max(current.numel(), 1))


def path_metrics(
    trajectory: list[dict[str, Any]],
    surfaces: list[torch.Tensor],
    fasts: list[torch.Tensor],
    signature: torch.Tensor,
    perturb_step: int,
) -> dict[str, float]:
    post_start = max(perturb_step - 1, 0)
    post = surfaces[post_start:]
    velocities = [post[idx] - post[idx - 1] for idx in range(1, len(post))]
    lengths = [float(vel.norm()) for vel in velocities]
    path_length = float(sum(lengths))
    net_displacement = float((post[-1] - post[0]).norm()) if len(post) > 1 else 0.0
    angles = []
    areas = []
    for idx in range(1, len(velocities)):
        a = velocities[idx - 1]
        b = velocities[idx]
        denom = float(a.norm() * b.norm())
        if denom > 1e-10:
            cos = max(-1.0, min(1.0, float(torch.dot(a, b) / denom)))
            angles.append(math.acos(cos))
            gram = max(float(a.norm() ** 2 * b.norm() ** 2 - torch.dot(a, b) ** 2), 0.0)
            areas.append(0.5 * math.sqrt(gram))
    release_rows = [
        row for row in trajectory
        if float(row.get("route_metrics", {}).get("release_strength_mean", 0.0)) > 0.0
    ]
    release_transfer = []
    for row in release_rows:
        idx = int(row["step"]) - 1
        if 0 < idx < len(fasts):
            release_transfer.append(cosine_to_signature(fasts[idx] - fasts[idx - 1], signature))
    release_local = release_local_metrics(
        trajectory,
        surfaces,
        signature,
        release_rows,
        window=6,
    )
    release_timing = release_timing_metrics(
        trajectory,
        release_rows,
        perturb_step=perturb_step,
    )
    tail = trajectory[-max(4, len(trajectory) // 4):]
    post_rows = trajectory[post_start:]
    channel_sep = channel_separation(post_rows)
    richness = internal_richness(post_rows)
    max_norm = max(row["residual_norm"] for row in trajectory)
    boundedness = 1.0 / (1.0 + max(0.0, max_norm - 8.0))
    return {
        "path_length": path_length,
        "net_displacement": net_displacement,
        "path_directness": net_displacement / (path_length + 1e-10),
        "path_curvature_mean": float(mean(angles)) if angles else 0.0,
        "path_loop_area_proxy": float(sum(areas)),
        "message_signature_tail": float(mean(abs(row["message_signature_cosine"]) for row in tail)),
        "carrier_signature_tail": float(mean(abs(row["carrier_signature_cosine"]) for row in tail)),
        "message_signature_peak": max(abs(row["message_signature_cosine"]) for row in post_rows),
        "carrier_signature_peak": max(abs(row["carrier_signature_cosine"]) for row in post_rows),
        "release_duty_cycle": len(release_rows) / max(len(trajectory), 1),
        "release_strength_mean": float(mean(
            row.get("route_metrics", {}).get("release_strength_mean", 0.0)
            for row in trajectory
        )),
        "release_transfer": float(mean(release_transfer)) if release_transfer else 0.0,
        **release_local,
        **release_timing,
        "internal_richness": richness,
        "channel_separation": channel_sep,
        "boundedness": boundedness,
        "geometric_coherence": boundedness * (0.5 + 0.5 * channel_sep) * (1.0 - min(1.0, float(mean(angles)) if angles else 0.0)),
    }


def release_timing_metrics(
    trajectory: list[dict[str, Any]],
    release_rows: list[dict[str, Any]],
    *,
    perturb_step: int,
) -> dict[str, float]:
    steps = [int(row["step"]) for row in release_rows]
    eligible_start = perturb_step + 8
    eligible_end = min(max(row["step"] for row in trajectory), perturb_step + 32)
    pre_rows = [row for row in trajectory if row["step"] < eligible_start]
    eligible_rows = [row for row in release_rows if eligible_start <= row["step"] <= eligible_end]
    early_rows = [row for row in release_rows if row["step"] < eligible_start]
    late_rows = [row for row in release_rows if row["step"] > eligible_end]
    first_release = float(steps[0]) if steps else 0.0
    step_std = float(pstdev(steps)) if len(steps) > 1 else 0.0
    burstiness = release_burstiness(steps)
    pressure_at_release = mean_or_zero(
        row.get("route_metrics", {}).get("release_pressure_mean", 0.0)
        for row in release_rows
    )
    pressure_pre_release = mean_or_zero(
        row.get("route_metrics", {}).get("release_pressure_mean", 0.0)
        for row in pre_rows
    )
    accumulation_series = [
        0.5 * (abs(row["message_signature_cosine"]) + abs(row["carrier_signature_cosine"]))
        for row in pre_rows
    ]
    accumulation_slope = normalized_slope(accumulation_series)
    pre_tail = accumulation_series[-max(3, min(12, len(accumulation_series))):]
    pre_tail_accumulation = mean_or_zero(pre_tail)
    release_count = len(release_rows)
    eligible_fraction = len(eligible_rows) / max(release_count, 1)
    early_fraction = len(early_rows) / max(release_count, 1)
    late_fraction = len(late_rows) / max(release_count, 1)
    delayed_window_steps = max(0, eligible_end - eligible_start + 1)
    delayed_window_fraction = delayed_window_steps / max(len(trajectory), 1)
    concentration_lift = max(0.0, eligible_fraction - delayed_window_fraction) / max(
        1.0 - delayed_window_fraction,
        1e-8,
    )
    pressure_lift = max(0.0, pressure_at_release - pressure_pre_release)
    duty = release_count / max(len(trajectory), 1)
    after_accumulation = pre_tail_accumulation * eligible_fraction * (1.0 - early_fraction)
    timing_score = min(1.0, (
        concentration_lift
        * (1.0 + pressure_lift)
        * (1.0 - 0.5 * late_fraction)
        * rare_release_score(duty)
    ))
    delayed_release_pressure = (
        pre_tail_accumulation
        * eligible_fraction
        * (1.0 - early_fraction) ** 2
        * (1.0 - 0.5 * late_fraction)
        * delayed_release_duty_score(duty)
        * (0.5 + 0.5 * burstiness)
        * (1.0 + pressure_lift)
    )
    return {
        "first_release_step": first_release,
        "release_step_std": step_std,
        "release_burstiness": burstiness,
        "release_eligible_fraction": eligible_fraction,
        "release_early_fraction": early_fraction,
        "release_late_fraction": late_fraction,
        "pre_release_accumulation_slope": accumulation_slope,
        "pre_release_tail_accumulation": pre_tail_accumulation,
        "pressure_at_release_mean": pressure_at_release,
        "pressure_pre_release_mean": pressure_pre_release,
        "release_after_accumulation_score": after_accumulation,
        "release_timing_score": timing_score,
        "delayed_release_pressure": delayed_release_pressure,
    }


def delayed_release_duty_score(duty: float) -> float:
    return math.exp(-((float(duty) - 0.10) / 0.12) ** 2)


def release_burstiness(steps: list[int]) -> float:
    if not steps:
        return 0.0
    sorted_steps = sorted(steps)
    longest = 1
    current = 1
    for idx in range(1, len(sorted_steps)):
        if sorted_steps[idx] == sorted_steps[idx - 1] + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return float(longest / len(sorted_steps))


def normalized_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = np.arange(len(values), dtype=float)
    ys = np.asarray(values, dtype=float)
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = float(np.dot(xs, xs))
    if denom <= 1e-12:
        return 0.0
    slope = float(np.dot(xs, ys) / denom)
    return slope / (float(np.std(values)) + 1e-9)


def release_local_metrics(
    trajectory: list[dict[str, Any]],
    surfaces: list[torch.Tensor],
    signature: torch.Tensor,
    release_rows: list[dict[str, Any]],
    *,
    window: int,
) -> dict[str, float]:
    if not release_rows:
        return {
            "release_pre_accumulation": 0.0,
            "release_pre_event_pressure": 0.0,
            "release_local_displacement": 0.0,
            "release_local_delta_lift": 0.0,
            "release_local_curvature_lift": 0.0,
            "release_local_signature_push": 0.0,
            "release_local_causality": 0.0,
            "release_geometric_event": 0.0,
            "phase_transition_score": 0.0,
        }
    pre_accumulation = []
    pre_event_pressure = []
    local_displacements = []
    delta_lifts = []
    curvature_lifts = []
    signature_pushes = []
    for row in release_rows:
        idx = int(row["step"]) - 1
        before_idx = max(0, idx - window)
        after_idx = min(len(surfaces) - 1, idx + window)
        if before_idx == after_idx:
            continue
        before_surface = surfaces[before_idx]
        release_surface = surfaces[idx]
        after_surface = surfaces[after_idx]
        displacement = float(torch.norm(after_surface - before_surface)) / math.sqrt(max(after_surface.numel(), 1))
        local_displacements.append(displacement)
        pre_rows = trajectory[before_idx:idx] or [row]
        post_rows = trajectory[idx : after_idx + 1] or [row]
        pre_delta = mean(float(item["residual_delta"]) for item in pre_rows)
        post_delta = mean(float(item["residual_delta"]) for item in post_rows)
        delta_lifts.append(max(0.0, post_delta - pre_delta))
        pre_curvature = local_curvature(surfaces[before_idx : idx + 1])
        post_curvature = local_curvature(surfaces[idx : after_idx + 1])
        curvature_lifts.append(max(0.0, post_curvature - pre_curvature))
        pre_accumulation.append(
            0.5
            * (
                abs(float(row.get("message_signature_cosine", 0.0)))
                + abs(float(row.get("carrier_signature_cosine", 0.0)))
            )
        )
        pre_event_pressure.append(
            mean_or_zero(
                item.get("route_metrics", {}).get("release_pressure_mean", 0.0)
                for item in pre_rows
            )
        )
        push = after_surface - release_surface
        signature_pushes.append(abs(cosine_to_signature(push, signature)))
    strength = mean(
        float(row.get("route_metrics", {}).get("release_strength_mean", 0.0))
        for row in release_rows
    )
    duty = len(release_rows) / max(len(trajectory), 1)
    sparse_gate = 1.0 - min(1.0, max(0.0, duty - 0.18) / 0.42)
    geometric_event = (
        (0.25 + mean_or_zero(pre_event_pressure))
        * mean_or_zero(local_displacements)
        * (1.0 + mean_or_zero(delta_lifts))
        * (1.0 + mean_or_zero(curvature_lifts))
        * strength
        * sparse_gate
    )
    delta_phase = min(1.0, mean_or_zero(delta_lifts) / PHASE_DELTA_SCALE)
    curvature_phase = min(1.0, mean_or_zero(curvature_lifts) / PHASE_CURVATURE_SCALE)
    phase_transition = (
        delta_phase
        * (0.5 + 0.5 * curvature_phase)
        * sparse_gate
        * (0.5 + min(1.0, duty * 8.0))
    )
    causality = (
        mean_or_zero(pre_accumulation)
        * mean_or_zero(local_displacements)
        * (1.0 + mean_or_zero(delta_lifts))
        * (1.0 + mean_or_zero(curvature_lifts))
        * mean_or_zero(signature_pushes)
        * strength
        * sparse_gate
    )
    return {
        "release_pre_accumulation": mean_or_zero(pre_accumulation),
        "release_pre_event_pressure": mean_or_zero(pre_event_pressure),
        "release_local_displacement": mean_or_zero(local_displacements),
        "release_local_delta_lift": mean_or_zero(delta_lifts),
        "release_local_curvature_lift": mean_or_zero(curvature_lifts),
        "release_local_signature_push": mean_or_zero(signature_pushes),
        "release_local_causality": causality,
        "release_geometric_event": geometric_event,
        "phase_transition_score": phase_transition,
    }


def local_curvature(points: list[torch.Tensor]) -> float:
    if len(points) < 3:
        return 0.0
    angles = []
    velocities = [points[idx] - points[idx - 1] for idx in range(1, len(points))]
    for idx in range(1, len(velocities)):
        a = velocities[idx - 1]
        b = velocities[idx]
        denom = float(a.norm() * b.norm())
        if denom > 1e-10:
            cos = max(-1.0, min(1.0, float(torch.dot(a, b) / denom)))
            angles.append(math.acos(cos))
    return float(mean(angles)) if angles else 0.0


def mean_or_zero(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(mean(rows)) if rows else 0.0


def channel_separation(rows: list[dict[str, Any]]) -> float:
    ratios = []
    for row in rows:
        vals = [
            row["fast_state_norm"],
            row["slow_state_norm"],
            row["message_state_norm"],
            row["carrier_state_norm"],
        ]
        spread = float(np.std(vals))
        scale = float(np.mean(vals)) + 1e-10
        energy = float(np.sum(np.square(vals)))
        participation = (float(np.sum(vals)) ** 2) / (4.0 * energy + 1e-10)
        ratios.append(min(1.0, spread / scale) * math.sqrt(max(0.0, min(1.0, participation))))
    return float(mean(ratios)) if ratios else 0.0


def internal_richness(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 3:
        return 0.0
    mat = np.asarray([
        [
            row["fast_state_norm"],
            row["slow_state_norm"],
            row["message_state_norm"],
            row["carrier_state_norm"],
            row["fast_state_delta"],
            row["slow_state_delta"],
            row["control_state_delta"],
            row["message_state_delta"],
            row["carrier_state_delta"],
            row["route_metrics"].get("release_open_mean", 0.0),
            row["route_metrics"].get("plastic_state_norm", 0.0),
        ]
        for row in rows
    ], dtype=float)
    mat = mat - mat.mean(axis=0, keepdims=True)
    _, singular, _ = np.linalg.svd(mat, full_matrices=False)
    if singular.size == 0:
        return 0.0
    threshold = max(float(singular[0]) * 1e-3, 1e-9)
    return float(np.sum(singular > threshold)) / float(mat.shape[1])


def classify_regime(
    attractor_type: str,
    trajectory: list[dict[str, Any]],
    metrics: dict[str, float],
) -> str:
    max_norm = max(row["residual_norm"] for row in trajectory)
    if not math.isfinite(max_norm) or max_norm > 40.0:
        return "unbounded_expanding"
    if metrics["channel_separation"] < 0.04 and metrics["internal_richness"] < 0.12:
        return "collapsed_incoherent"
    if attractor_type == "FIXED_POINT":
        if (
            metrics["internal_richness"] >= 0.22
            or metrics["phase_transition_score"] >= 0.012
            or metrics["release_geometric_event"] >= 0.002
        ):
            return "surface_fixed_accumulating"
        return "surface_fixed_internal_quiet"
    if attractor_type == "CHAOTIC":
        return "bounded_chaotic_structured"
    if attractor_type == "STRANGE":
        return "bounded_strange"
    return str(attractor_type).lower()


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [run["metrics"] for run in runs]
    regimes: dict[str, int] = {}
    motifs: dict[str, list[float]] = {}
    for run in runs:
        regime = run["summary"]["regime_class"]
        regimes[regime] = regimes.get(regime, 0) + 1
        motifs.setdefault(run["motif_family"], []).append(
            max(run["metrics"]["message_signature_tail"], run["metrics"]["carrier_signature_tail"])
        )
    motif_means = [mean(values) for values in motifs.values()]
    motif_spread = pstdev(motif_means) if len(motif_means) > 1 else 0.0
    release_effect = mean(
        m["release_strength_mean"] * abs(m["release_transfer"]) * (1.0 - min(1.0, 3.0 * m["release_duty_cycle"]))
        for m in metrics
    )
    internal_richness = mean(m["internal_richness"] for m in metrics)
    memory_specificity = mean(max(m["message_signature_tail"], m["carrier_signature_tail"]) for m in metrics) + motif_spread
    channel_sep = mean(m["channel_separation"] for m in metrics)
    release_geometric_event = mean(m["release_geometric_event"] for m in metrics)
    release_causal_divergence = mean(m.get("release_causal_divergence", 0.0) for m in metrics)
    release_gain_zero_causal_divergence = mean(
        m.get("release_gain_zero_release_causal_divergence", 0.0) for m in metrics
    )
    gain_zero_clean_fraction = mean(1.0 if m.get("gain_zero_clean", False) else 0.0 for m in metrics)
    phase_transition_score = mean(m["phase_transition_score"] for m in metrics)
    boundedness = mean(m["boundedness"] for m in metrics)
    geometric_coherence = mean(m["geometric_coherence"] for m in metrics)
    path_curvature = mean(m["path_curvature_mean"] for m in metrics)
    regime_bonus = min(1.0, len(regimes) / 4.0)
    mathematical_curiosity = boundedness * (
        0.30 * internal_richness
        + 0.20 * channel_sep
        + 0.20 * min(1.0, 3.0 * path_curvature)
        + 0.15 * min(1.0, 120.0 * release_geometric_event)
        + 0.15 * geometric_coherence
        + 0.10 * regime_bonus
    )
    causal_components = causal_divergence_components(
        {
            "release_causal_divergence": release_causal_divergence,
            "release_gain_zero_release_causal_divergence": release_gain_zero_causal_divergence,
            "gain_zero_clean_fraction": gain_zero_clean_fraction,
        }
    )
    return {
        "internal_richness": internal_richness,
        "memory_specificity": memory_specificity,
        "channel_separation": channel_sep,
        "release_effectiveness": release_effect,
        "release_local_causality": mean(m["release_local_causality"] for m in metrics),
        "release_geometric_event": release_geometric_event,
        "release_causal_divergence": release_causal_divergence,
        "release_gain_zero_release_causal_divergence": release_gain_zero_causal_divergence,
        "release_route_specific_causal_divergence": causal_components["route_release_causal_divergence"],
        "gate_state_causal_divergence": causal_components["gate_state_causal_divergence"],
        "gain_zero_clean_fraction": gain_zero_clean_fraction,
        "gain_zero_max_release_strength": max(m.get("gain_zero_max_release_strength", 0.0) for m in metrics),
        "gain_zero_max_route_norm": max(m.get("gain_zero_max_route_norm", 0.0) for m in metrics),
        "gain_zero_max_endogenous_release_norm": max(
            m.get("gain_zero_max_endogenous_release_norm", 0.0) for m in metrics
        ),
        "gain_zero_own_release_anchor_count": mean(
            m.get("gain_zero_own_release_anchor_count", 0.0) for m in metrics
        ),
        **{
            f"release_causal_divergence_{channel}": mean(
                m.get(f"release_causal_divergence_{channel}", 0.0)
                for m in metrics
            )
            for channel in CHANNELS
        },
        **{
            f"release_gain_zero_release_causal_divergence_{channel}": mean(
                m.get(f"release_gain_zero_release_causal_divergence_{channel}", 0.0)
                for m in metrics
            )
            for channel in CHANNELS
        },
        "release_causal_anchor_count": mean(m.get("release_causal_anchor_count", 0.0) for m in metrics),
        "release_gain_zero_release_causal_anchor_count": mean(
            m.get("release_gain_zero_release_causal_anchor_count", 0.0) for m in metrics
        ),
        "phase_transition_score": phase_transition_score,
        "release_pre_accumulation": mean(m["release_pre_accumulation"] for m in metrics),
        "release_pre_event_pressure": mean(m["release_pre_event_pressure"] for m in metrics),
        "release_local_displacement": mean(m["release_local_displacement"] for m in metrics),
        "release_local_delta_lift": mean(m["release_local_delta_lift"] for m in metrics),
        "release_local_curvature_lift": mean(m["release_local_curvature_lift"] for m in metrics),
        "release_local_signature_push": mean(m["release_local_signature_push"] for m in metrics),
        "first_release_step_mean": mean(m["first_release_step"] for m in metrics),
        "release_step_std": mean(m["release_step_std"] for m in metrics),
        "release_burstiness": mean(m["release_burstiness"] for m in metrics),
        "release_eligible_fraction": mean(m["release_eligible_fraction"] for m in metrics),
        "release_early_fraction": mean(m["release_early_fraction"] for m in metrics),
        "release_late_fraction": mean(m["release_late_fraction"] for m in metrics),
        "pre_release_accumulation_slope": mean(m["pre_release_accumulation_slope"] for m in metrics),
        "pre_release_tail_accumulation": mean(m["pre_release_tail_accumulation"] for m in metrics),
        "pressure_at_release_mean": mean(m["pressure_at_release_mean"] for m in metrics),
        "pressure_pre_release_mean": mean(m["pressure_pre_release_mean"] for m in metrics),
        "release_after_accumulation_score": mean(m["release_after_accumulation_score"] for m in metrics),
        "release_timing_score": mean(m["release_timing_score"] for m in metrics),
        "delayed_release_pressure": mean(m["delayed_release_pressure"] for m in metrics),
        "mathematical_curiosity": mathematical_curiosity,
        "boundedness": boundedness,
        "geometric_coherence": geometric_coherence,
        "release_duty_cycle": mean(m["release_duty_cycle"] for m in metrics),
        "path_curvature_mean": path_curvature,
        "regime_bonus": regime_bonus,
        "regimes": regimes,
        "motif_memory": {key: mean(values) for key, values in motifs.items()},
    }


def evaluate_genome(
    genome: dict[str, Any],
    *,
    hidden_size: int,
    steps: int,
    seeds: list[int],
    perturb_step: int,
    perturb_scales: list[float],
    rank: int,
    device: str,
    perturb_channels: list[str] | None = None,
    perturb_modes: list[str] | None = None,
    keep_trajectories: bool = False,
    paired_causal: bool = False,
) -> dict[str, Any]:
    runs = []
    paired_runs = []
    channels = perturb_channels or ["fast"]
    modes = perturb_modes or ["external"]
    for seed in seeds:
        motifs = motif_suite(hidden_size, seed)
        for scale in perturb_scales:
            for motif in motifs:
                for channel in channels:
                    for mode in modes:
                        if paired_causal:
                            paired = paired_causal_run(
                                genome,
                                hidden_size=hidden_size,
                                steps=steps,
                                seed=seed,
                                perturb_step=perturb_step,
                                perturb_scale=scale,
                                motif=motif,
                                rank=rank,
                                device=device,
                                perturb_channel=channel,
                                perturb_mode=mode,
                            )
                            run = next(item for item in paired["runs"] if item["ablation"] == "original")
                            paired_runs.extend(paired["runs"])
                        else:
                            run = run_with_motif(
                                genome,
                                hidden_size=hidden_size,
                                steps=steps,
                                seed=seed,
                                perturb_step=perturb_step,
                                perturb_scale=scale,
                                motif=motif,
                                rank=rank,
                                device=device,
                                perturb_channel=channel,
                                perturb_mode=mode,
                            )
                        if not keep_trajectories:
                            run.pop("trajectory", None)
                        runs.append(run)
    aggregate = aggregate_runs(runs)
    result = {"aggregate": aggregate, "runs": runs}
    if paired_causal:
        if not keep_trajectories:
            for run in paired_runs:
                run.pop("trajectory", None)
        result["paired_runs"] = paired_runs
    return result


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but torch.cuda.is_available() is false")
    return device


def parse_worker_devices(text: str, fallback: str, workers: int = 1) -> list[str]:
    devices = [item.strip() for item in text.split(",") if item.strip()]
    if not devices and fallback == "auto" and torch.cuda.is_available():
        count = min(max(1, int(workers)), max(1, torch.cuda.device_count()))
        return [f"cuda:{idx}" for idx in range(count)]
    return devices if devices else [fallback]


def normalize_workers_for_devices(workers: int, worker_devices: list[str]) -> int:
    """Avoid spawning duplicate CUDA evaluators for the same physical device."""
    requested = max(1, int(workers))
    if os.environ.get("DEMIAN_ALLOW_DUPLICATE_CUDA_WORKERS", "0").lower() in {"1", "true", "yes", "on"}:
        return requested
    cuda_devices = [device for device in worker_devices if str(device).startswith("cuda")]
    if not cuda_devices:
        return requested
    unique_cuda_devices = len(set(cuda_devices))
    non_cuda_devices = len(worker_devices) - len(cuda_devices)
    return max(1, min(requested, unique_cuda_devices + non_cuda_devices))


def cuda_lock_name(device: str) -> str:
    return device.replace(":", "_").replace("/", "_")


def cuda_lock_enabled() -> bool:
    return os.environ.get("DEMIAN_CUDA_SERIALIZE", "0").lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def cuda_device_lock(device: str):
    """Serialize CUDA jobs across independent island processes on the same GPU."""
    if not str(device).startswith("cuda") or not cuda_lock_enabled():
        yield
        return
    CUDA_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = CUDA_LOCK_DIR / f"{cuda_lock_name(device)}.lock"
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def evaluate_candidate_job(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("torch_threads"):
        torch.set_num_threads(int(job["torch_threads"]))
    device = resolve_device(str(job["device"]))
    with cuda_device_lock(device):
        result = evaluate_genome(
            job["genome"],
            hidden_size=int(job["hidden_size"]),
            steps=int(job["steps"]),
            seeds=list(job["seeds"]),
            perturb_step=int(job["perturb_step"]),
            perturb_scales=list(job["perturb_scales"]),
            rank=int(job["rank"]),
            device=device,
            perturb_channels=list(job.get("perturb_channels", ["fast"])),
            perturb_modes=list(job.get("perturb_modes", ["external"])),
            keep_trajectories=False,
            paired_causal=bool(job.get("paired_causal", False)),
        )
    return {
        "id": job["candidate_id"],
        "generation": int(job["generation"]),
        "index": int(job["index"]),
        "genome": job["genome"],
        "reproduction_kind": job["reproduction_kind"],
        "parent_ids": job["parent_ids"],
        "ancestor_ids": job["ancestor_ids"],
        "generation_of_origin": int(job["generation_of_origin"]),
        "mutation_count": int(job["mutation_count"]),
        "worker_device": device,
        "rank_mode": str(job.get("rank_mode", ENGINEERED_TARGET_RANK_MODE)),
        "causal_mode": str(job.get("causal_mode", DEFAULT_CAUSAL_MODE)),
        "native_objective": str(job.get("native_objective", DEFAULT_NATIVE_OBJECTIVE)),
        "metrics": result["aggregate"],
        "runs": result["runs"],
        "paired_runs": result.get("paired_runs", []),
    }


def choose_reproduction_kind(
    rng: random.Random,
    weights: tuple[tuple[str, float], ...] = REPRODUCTION_WEIGHTS,
) -> str:
    roll = rng.random()
    cumulative = 0.0
    for kind, weight in weights:
        cumulative += float(weight)
        if roll < cumulative:
            return kind
    return weights[-1][0]


def reproduce(
    evaluated: list[dict[str, Any]],
    global_archive: list[dict[str, Any]],
    *,
    population_size: int,
    generation_of_origin: int,
    rng: random.Random,
    hidden_size: int,
    rank: int,
    mutation_sigma: float,
    reproduction_weights: tuple[tuple[str, float], ...] = REPRODUCTION_WEIGHTS,
    random_injection_rate: float = RANDOM_INJECTION_RATE,
    include_default_seed: bool = True,
    elitism: bool = True,
    allow_structured_operators: bool = True,
    use_archive_bins_for_parents: bool = True,
) -> list[dict[str, Any]]:
    if use_archive_bins_for_parents:
        parents = archive_bins(global_archive)[: max(2, population_size // 2)]
    else:
        parents = sorted(global_archive, key=scalar_rank, reverse=True)[: max(2, population_size // 2)]
    if len(parents) < 2:
        parents = sorted(evaluated, key=scalar_rank, reverse=True)[:2]
    population = []
    if include_default_seed:
        population.append(population_entry(default_genome(hidden_size, rank), "default_seed", []))
    if elitism:
        for parent in parents[: max(1, population_size // 5)]:
            population.append(
                population_entry(
                    parent["genome"],
                    "elite_copy",
                    [parent["id"]],
                    ancestor_ids=parent.get("ancestor_ids", [parent["id"]]),
                    generation_of_origin=int(parent.get("generation_of_origin", parent["generation"])),
                    mutation_count=int(parent.get("mutation_count", 0)),
                )
            )
    while len(population) < population_size:
        reproduction_kind = choose_reproduction_kind(rng, reproduction_weights)
        if reproduction_kind == "crossover" and len(parents) >= 2:
            left, right = rng.sample(parents, 2)
            child = crossover(left["genome"], right["genome"], rng, hidden_size, rank)
            entry = population_entry(
                child,
                "crossover",
                [left["id"], right["id"]],
                ancestor_ids=merge_ancestor_ids(left, right),
                generation_of_origin=generation_of_origin,
                mutation_count=max(
                    int(left.get("mutation_count", 0)),
                    int(right.get("mutation_count", 0)),
                ) + 1,
            )
        else:
            parent = rng.choice(parents)
            if allow_structured_operators and reproduction_kind == "causal_release_template_mutation":
                child = mutate_causal_release_template_genome(
                    parent["genome"], rng, hidden_size, rank, mutation_sigma
                )
                entry = lineage_child_entry(
                    child,
                    "causal_release_template_mutation",
                    parent,
                    generation_of_origin,
                )
            elif allow_structured_operators and reproduction_kind == "delayed_eligibility_template_mutation":
                child = mutate_delayed_eligibility_template_genome(
                    parent["genome"], rng, hidden_size, rank, mutation_sigma
                )
                entry = lineage_child_entry(
                    child,
                    "delayed_eligibility_template_mutation",
                    parent,
                    generation_of_origin,
                )
            elif allow_structured_operators and reproduction_kind == "release_phase_mutation":
                child = mutate_release_phase_genome(
                    parent["genome"], rng, hidden_size, rank, mutation_sigma
                )
                entry = lineage_child_entry(
                    child,
                    "release_phase_mutation",
                    parent,
                    generation_of_origin,
                )
            else:
                child = mutate_genome(parent["genome"], rng, hidden_size, rank, mutation_sigma)
                entry = lineage_child_entry(
                    child,
                    "mutation",
                    parent,
                    generation_of_origin,
                )
        if random_injection_rate > 0.0 and rng.random() < random_injection_rate:
            child = random_genome(rng, hidden_size, rank)
            entry = population_entry(child, "random", [])
        population.append(entry)
    return population[:population_size]


def candidate_jobs(
    population: list[dict[str, Any]],
    *,
    generation: int,
    args: argparse.Namespace,
    seeds: list[int],
    scales: list[float],
    worker_devices: list[str],
) -> list[dict[str, Any]]:
    jobs = []
    for idx, entry in enumerate(population):
        candidate_id = f"gen{generation:03d}_candidate{idx:03d}"
        ancestor_ids = list(entry.get("ancestor_ids", [candidate_id]))
        jobs.append(
            {
                "candidate_id": candidate_id,
                "generation": generation,
                "index": idx,
                "genome": entry["genome"],
                "reproduction_kind": entry["reproduction_kind"],
                "parent_ids": entry["parent_ids"],
                "ancestor_ids": ancestor_ids,
                "generation_of_origin": int(entry.get("generation_of_origin", generation)),
                "mutation_count": int(entry.get("mutation_count", 0)),
                "hidden_size": args.hidden_size,
                "steps": args.steps,
                "seeds": seeds,
                "perturb_step": args.perturb_step,
                "perturb_scales": scales,
                "perturb_channels": getattr(args, "perturb_channels", ["fast"]),
                "perturb_modes": getattr(args, "perturb_modes", ["external"]),
                "rank": args.rank,
                "device": worker_devices[idx % len(worker_devices)],
                "torch_threads": args.torch_threads,
                "paired_causal": getattr(args, "paired_causal", False),
                "rank_mode": getattr(args, "rank_mode", ENGINEERED_TARGET_RANK_MODE),
                "causal_mode": getattr(args, "causal_mode", DEFAULT_CAUSAL_MODE),
                "native_objective": getattr(args, "native_objective", DEFAULT_NATIVE_OBJECTIVE),
            }
        )
    return jobs


def evaluate_generation_jobs(
    jobs: list[dict[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [evaluate_candidate_job(job) for job in jobs]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(evaluate_candidate_job, job) for job in jobs]
        rows = []
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row["index"]))
    return rows


def top_ablation_report(
    rows: list[dict[str, Any]],
    *,
    generation: int,
    top_n: int = 3,
) -> dict[str, Any]:
    """Summarize paired ablations for the top ranked rows in a generation."""
    reports = []
    for row in sorted(rows, key=scalar_rank, reverse=True)[:top_n]:
        rank_mode = str(row.get("rank_mode", ENGINEERED_TARGET_RANK_MODE))
        causal_mode = str(row.get("causal_mode", DEFAULT_CAUSAL_MODE))
        native_objective = str(row.get("native_objective", DEFAULT_NATIVE_OBJECTIVE))
        ablation_runs: dict[str, list[dict[str, Any]]] = {}
        for run in row.get("paired_runs", []):
            ablation_runs.setdefault(str(run.get("ablation", "unknown")), []).append(run)
        ablations = {}
        for ablation, runs in ablation_runs.items():
            aggregate = aggregate_runs(runs)
            rank_row = {
                "metrics": aggregate,
                "rank_mode": rank_mode,
                "causal_mode": causal_mode,
                "native_objective": native_objective,
            }
            ablations[ablation] = {
                "run_count": len(runs),
                "rank_score": scalar_rank(rank_row),
                "rank_components": rank_components(rank_row),
                "metrics": aggregate,
            }
        reports.append(
            {
                "id": row["id"],
                "rank_score": scalar_rank(row),
                "rank_components": rank_components(row),
                "rank_mode": rank_mode,
                "causal_mode": causal_mode,
                "native_objective": native_objective,
                "metrics": row["metrics"],
                "ablations": ablations,
            }
        )
    return {
        "generation": generation,
        "top_n": top_n,
        "report_kind": "paired_causal_top_ablation",
        "candidates": reports,
    }


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_choices(text: str, allowed: tuple[str, ...]) -> list[str]:
    choices = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in choices if item not in allowed]
    if unknown:
        raise ValueError(f"unknown choices {unknown}; allowed={allowed}")
    return choices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolve v9-5ch release gate archives")
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--eval-seeds", default="94,95,96,97")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--perturb-step", type=int, default=64)
    parser.add_argument("--perturb-scales", default="0.35,0.7")
    parser.add_argument("--perturb-channels", default="fast")
    parser.add_argument("--perturb-modes", default="external")
    parser.add_argument("--experiment-version", default="v9.1")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--mutation-sigma", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-devices", default="")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--paired-causal", action="store_true")
    parser.add_argument(
        "--rank-mode",
        choices=(
            ENGINEERED_TARGET_RANK_MODE,
            NATIVE_EMERGENCE_RANK_MODE,
            DYNAMIC_SELECTION_PROBE_RANK_MODE,
        ),
        default=ENGINEERED_TARGET_RANK_MODE,
    )
    parser.add_argument(
        "--causal-mode",
        choices=CAUSAL_MODES,
        default=DEFAULT_CAUSAL_MODE,
        help=(
            "Select the causal signal used by ranking: route-specific release, "
            "persistent gate-state propagation, or both as separated terms."
        ),
    )
    parser.add_argument(
        "--native-objective",
        choices=NATIVE_OBJECTIVES,
        default=DEFAULT_NATIVE_OBJECTIVE,
        help="Track B native objective variant; independent from causal_mode.",
    )
    parser.add_argument(
        "--reproduction-mode",
        choices=("structured", "native"),
        default="structured",
    )
    parser.add_argument("--no-default-seed", action="store_true")
    parser.add_argument("--no-elitism", action="store_true")
    parser.add_argument("--no-random-injection", action="store_true")
    parser.add_argument("--demian-v1-minimal-causal", action="store_true")
    parser.add_argument("--demian-v1-delayed-island", action="store_true")
    parser.add_argument("--demian-v2-track-a-island", action="store_true")
    parser.add_argument("--demian-v2-track-b-island", action="store_true")
    parser.add_argument("--dynamic-selection-probe", action="store_true")
    parser.add_argument("--island-index", type=int, default=1)
    parser.add_argument("--out-dir", default="data/evolution/v9_5ch_release_20260509")
    args = parser.parse_args()
    if args.demian_v1_minimal_causal:
        args.population = 16
        args.generations = 10
        args.eval_seeds = "94,95"
        args.paired_causal = True
        args.experiment_version = "demian-v1-causal-minimal"
        args.device = "cpu"
        if args.out_dir == "data/evolution/v9_5ch_release_20260509":
            args.out_dir = "data/evolution/demian_v1_causal_minimal_20260511"
    if args.demian_v1_delayed_island:
        island_index = max(1, int(args.island_index))
        args.population = 8
        args.generations = 20
        args.eval_seeds = "94,95"
        args.steps = 128
        args.perturb_step = 64
        args.perturb_scales = "0.35,0.7"
        args.paired_causal = True
        args.experiment_version = "demian-v1-delayed-eligibility"
        args.seed = 2026051100 + island_index
        if args.out_dir == "data/evolution/v9_5ch_release_20260509":
            args.out_dir = f"data/evolution/demian_v1_delayed_island_{island_index}_20260511"
    if args.demian_v2_track_a_island:
        island_index = max(1, int(args.island_index))
        args.population = 8
        args.generations = 20
        args.eval_seeds = "94,95"
        args.steps = 128
        args.perturb_step = 64
        args.perturb_scales = "0.35,0.7"
        args.paired_causal = True
        args.experiment_version = "demian-v2-track-a-engineered-target"
        args.rank_mode = ENGINEERED_TARGET_RANK_MODE
        args.causal_mode = CAUSAL_MODE_ROUTE_RELEASE
        args.reproduction_mode = "structured"
        args.device = "cpu"
        args.torch_threads = args.torch_threads or 1
        args.seed = 2026051200 + island_index
        if args.out_dir == "data/evolution/v9_5ch_release_20260509":
            args.out_dir = f"data/evolution/demian_v2_track_a_island_{island_index}_20260511"
    if args.demian_v2_track_b_island:
        args.population = 8
        args.generations = 20
        args.eval_seeds = "94,95"
        args.steps = 128
        args.perturb_step = 64
        args.perturb_scales = "0.35,0.7"
        args.paired_causal = True
        args.experiment_version = "demian-v2-track-b-native-emergence"
        args.rank_mode = NATIVE_EMERGENCE_RANK_MODE
        args.causal_mode = CAUSAL_MODE_GATE_STATE
        args.native_objective = NATIVE_OBJECTIVE_GATE_STATE
        args.reproduction_mode = "native"
        args.no_default_seed = True
        args.no_elitism = True
        args.no_random_injection = True
        args.device = "cpu"
        args.torch_threads = args.torch_threads or 1
        args.seed = 2026051291
        if args.out_dir == "data/evolution/v9_5ch_release_20260509":
            args.out_dir = "data/evolution/demian_v2_track_b_island_1_20260511"
    if args.dynamic_selection_probe:
        args.population = 16
        args.generations = 60
        args.eval_seeds = "94,95"
        args.steps = 128
        args.perturb_step = 64
        args.perturb_scales = "0.35,0.7"
        args.paired_causal = True
        args.experiment_version = "dynamic-selection-probe"
        args.rank_mode = DYNAMIC_SELECTION_PROBE_RANK_MODE
        args.causal_mode = CAUSAL_MODE_ROUTE_RELEASE
        args.reproduction_mode = "native"
        args.no_default_seed = True
        args.no_elitism = True
        args.no_random_injection = True
        args.device = "cpu"
        args.torch_threads = args.torch_threads or 1
        args.seed = 2026051301
        if args.out_dir == "data/evolution/v9_5ch_release_20260509":
            args.out_dir = "data/evolution/dynamic_selection_probe_20260513"
    return args


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    candidates_dir = out_dir / "candidates"
    diagnostics_dir = out_dir / "diagnostics"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.eval_seeds)
    scales = parse_floats(args.perturb_scales)
    args.perturb_channels = parse_choices(args.perturb_channels, (*CHANNELS, "all"))
    args.perturb_modes = parse_choices(
        args.perturb_modes,
        (
            "clean",
            "external",
            "state_velocity",
            "carrier_pressure",
            "control_shadow",
            "trajectory_orthogonal",
        ),
    )
    worker_devices = parse_worker_devices(args.worker_devices, args.device, workers=args.workers)
    resolved_main_device = resolve_device(args.device)
    reproduction_weights = (
        NATIVE_REPRODUCTION_WEIGHTS
        if args.reproduction_mode == "native"
        else REPRODUCTION_WEIGHTS
    )
    random_injection_rate = 0.0 if args.no_random_injection else RANDOM_INJECTION_RATE
    include_default_seed = not args.no_default_seed
    elitism = not args.no_elitism
    allow_structured_operators = args.reproduction_mode == "structured"
    requested_workers = args.workers
    args.workers = normalize_workers_for_devices(args.workers, worker_devices)
    if args.workers != requested_workers:
        print(
            f"reduced workers from {requested_workers} to {args.workers} "
            f"for worker_devices={worker_devices}"
        )
    config = {
        "population": args.population,
        "generations": args.generations,
        "seed": args.seed,
        "eval_seeds": seeds,
        "cross_validation_note": (
            "CPU paired-causal reproduction is the reference for Demian v1 causal "
            "release checks. CUDA comparisons are diagnostic only and are not "
            "assumed equivalent."
        ),
        "hidden_size": args.hidden_size,
        "steps": args.steps,
        "perturb_step": args.perturb_step,
        "perturb_scales": scales,
        "perturb_channels": args.perturb_channels,
        "perturb_modes": args.perturb_modes,
        "experiment_version": args.experiment_version,
        "rank": args.rank,
        "mutation_sigma": args.mutation_sigma,
        "device": args.device,
        "resolved_main_device": resolved_main_device,
        "requested_workers": requested_workers,
        "workers": args.workers,
        "worker_devices": worker_devices,
        "cuda_lock_dir": str(CUDA_LOCK_DIR),
        "cuda_lock_enabled": cuda_lock_enabled(),
        "torch_threads": args.torch_threads,
        "paired_causal": args.paired_causal,
        "rank_mode": args.rank_mode,
        "causal_mode": args.causal_mode,
        "native_objective": args.native_objective,
        "reproduction_mode": args.reproduction_mode,
        "include_default_seed": include_default_seed,
        "elitism": elitism,
        "allow_structured_operators": allow_structured_operators,
        "use_archive_bins_for_parent_selection": args.reproduction_mode != "native",
        "scalar_genes": {name: asdict(spec) for name, spec in SCALAR_GENES.items()},
        "low_rank_targets": list(LOW_RANK_TARGETS),
        "release_target_scalars": list(RELEASE_TARGET_SCALARS),
        "pareto_objectives": list(PARETO_OBJECTIVES),
        "rank_pressure": {
            "engineered_target": {
                "causal_mode": args.causal_mode,
                "score_a": "internal_richness + 1.2*channel_separation + 0.7*mathematical_curiosity + geometric_coherence",
                "score_b": (
                    "1.6*causal_multiplier*min(1, selected_causal_divergence/0.25) "
                    "+ 1.6*causal_multiplier*duty_band_multiplier*release_timing_score "
                    "+ 1.2*causal_multiplier*min(1, phase_transition_score/1.0) "
                    "+ 0.45*regime_bonus - high-duty penalty"
                ),
                "causal_modes": {
                    CAUSAL_MODE_ROUTE_RELEASE: (
                        "release_causal_divergence > 0 and "
                        "release_gain_zero_release_causal_divergence <= 1e-8"
                    ),
                    CAUSAL_MODE_GATE_STATE: (
                        "gain_zero_clean_fraction == 1 and "
                        "release_gain_zero_release_causal_divergence persists"
                    ),
                    CAUSAL_MODE_COMBINED: "route_release_causal_divergence + gate_state_causal_divergence",
                },
                "causal_multiplier": "clamp(selected_causal_divergence / 0.01, 0, 1)",
                "duty_band_multiplier": "1.0 if 0.06 <= release_duty_cycle <= 0.18 else 0.0",
                "rank": "sqrt(max(0, score_a) * max(0, score_b))",
            },
            "native_emergence": {
                "causal_mode": args.causal_mode,
                "native_objective": args.native_objective,
                "objectives": list(NATIVE_OBJECTIVES),
                "gate_state": (
                    "morphology + selected_causal_divergence_raw + "
                    "0.3*release_geometric_event + 0.3*phase_transition_score"
                ),
                "morphology_only": (
                    "internal_richness + 1.2*channel_separation + "
                    "0.7*mathematical_curiosity + geometric_coherence"
                ),
                "morphology_low_duty": "morphology_only + capped low_duty_preference <= 0.15",
                "combined_discovery": (
                    "morphology_only + 0.25*selected_causal_divergence_raw + "
                    "0.15*release_geometric_event + 0.15*phase_transition_score"
                ),
                "excluded": "no causal gate, duty band, timing score, regime bonus, or structured operators",
            },
            "dynamic_selection_probe": {
                "phase_1_generations": [0, 19],
                "phase_1": (
                    "morphology + 1.5*raw release_causal_divergence + "
                    "0.3*release_geometric_event + 0.3*phase_transition_score"
                ),
                "phase_2_generations": [20, 39],
                "phase_2": (
                    "phase_1 terms with event/phase gated by clamp(raw causal / 0.01, 0, 1), "
                    "plus high-duty penalty"
                ),
                "phase_3_generations": [40, 59],
                "phase_3": (
                    "phase_2 terms plus "
                    "1.6*causal_multiplier*duty_band_multiplier*release_timing_score"
                ),
                "raw_causal_signal": "metrics.release_causal_divergence",
                "target_duty_band": [0.06, 0.18],
                "timing_window": [args.perturb_step + 8, min(args.steps, args.perturb_step + 32)],
            },
        },
        "reproduction_weights": {
            kind: weight for kind, weight in reproduction_weights
        },
        "random_injection_rate": random_injection_rate,
        "effective_reproduction_rates_after_random_injection": {
            kind: weight * (1.0 - random_injection_rate)
            for kind, weight in reproduction_weights
        } | {"random": random_injection_rate},
        "structured_causal_release_template_operator": {
            "reproduction_kind": "causal_release_template_mutation",
            "selection_probability": 0.28,
            "joint_scalar_targets": [
                "release_threshold",
                "release_temperature",
                "release_gain",
            ],
            "joint_route_targets": list(RELEASE_TARGET_SCALARS),
            "release_gate_observable_targets": list(OBSERVABLE_KEYS),
            "templates": CAUSAL_RELEASE_ROUTE_TEMPLATES,
        },
        "structured_delayed_eligibility_template_operator": {
            "reproduction_kind": "delayed_eligibility_template_mutation",
            "selection_probability": 0.15,
            "joint_scalar_targets": [
                "release_threshold",
                "release_temperature",
                "message_decay",
                "carrier_decay",
            ],
            "release_gate_observable_targets": [
                "time_since_perturbation",
                "perturbation_magnitude",
                "surface_delta",
                "release_pressure",
            ],
            "templates": DELAYED_ELIGIBILITY_TEMPLATES,
        },
        "phase_windows": {
            "accumulation": [1, args.perturb_step - 1],
            "perturb_step": args.perturb_step,
            "delayed_eligibility": [args.perturb_step + 8, min(args.steps, args.perturb_step + 32)],
            "post_perturb_observation": [args.perturb_step + 1, args.steps],
        },
    }
    save_json(config, out_dir / "config.json")
    population = []
    if include_default_seed:
        population.append(population_entry(default_genome(args.hidden_size, args.rank), "default_seed", []))
    while len(population) < args.population:
        population.append(population_entry(random_genome(rng, args.hidden_size, args.rank), "random", []))
    all_rows: list[dict[str, Any]] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    generation_log = out_dir / "generations.jsonl"
    if generation_log.exists():
        generation_log.unlink()
    print("v9-5ch release evolutionary archive")
    print(f"out_dir={out_dir}")
    print(f"device={args.device} resolved_main_device={resolved_main_device} workers={args.workers} worker_devices={worker_devices}")
    for generation in range(args.generations):
        start = time.time()
        evaluated = []
        print(f"[generation {generation}]")
        jobs = candidate_jobs(
            population,
            generation=generation,
            args=args,
            seeds=seeds,
            scales=scales,
            worker_devices=worker_devices,
        )
        for row in evaluate_generation_jobs(jobs, workers=args.workers):
            row["lineage_deltas"] = lineage_deltas(row["metrics"], row["parent_ids"], rows_by_id)
            row["lineage_stability_score"] = lineage_stability_score(row["lineage_deltas"])
            row["rank_score"] = scalar_rank(row)
            row["rank_components"] = rank_components(row)
            evaluated.append(row)
            all_rows.append(row)
            rows_by_id[row["id"]] = row
            save_json(row, candidates_dir / f"{row['id']}.json")
            print(
                "  {id} kind={kind} dev={dev} rich={rich:.3f} sep={sep:.3f} event={event:.4f} phase={phase:.4f} duty={duty:.3f} reg={reg}".format(
                    id=row["id"],
                    kind=row["reproduction_kind"],
                    dev=row.get("worker_device", "?"),
                    rich=float(row["metrics"]["internal_richness"]),
                    sep=float(row["metrics"]["channel_separation"]),
                    event=float(row["metrics"]["release_geometric_event"]),
                    phase=float(row["metrics"]["phase_transition_score"]),
                    duty=float(row["metrics"]["release_duty_cycle"]),
                    reg=row["metrics"]["regimes"],
                )
            )
        generation_diag = generation_diagnostics(evaluated, generation=generation)
        diagnostics.append(generation_diag)
        save_json(generation_diag, diagnostics_dir / f"generation_{generation:03d}.json")
        if args.paired_causal and (generation + 1) in {5, 10, 15, 20}:
            save_json(
                top_ablation_report(evaluated, generation=generation + 1, top_n=3),
                diagnostics_dir / f"generation_{generation + 1:03d}_top3_causal_ablation.json",
            )
        save_json(diagnostics, out_dir / "diagnostics.json")
        front = pareto_front(evaluated)
        archive = archive_bins(all_rows)
        global_front = pareto_front(all_rows)
        save_json(front, out_dir / f"frontier_gen{generation:03d}.json")
        save_json(global_front, out_dir / "frontier.json")
        save_json(archive, out_dir / "archive.json")
        summary = {
            "generation": generation,
            "elapsed_seconds": time.time() - start,
            "frontier_ids": [row["id"] for row in front],
            "archive_ids": [row["id"] for row in archive],
            "best_rank_id": max(evaluated, key=scalar_rank)["id"],
        }
        with generation_log.open("a") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
        print(f"  frontier={len(front)} archive={len(archive)} best={summary['best_rank_id']}")
        if generation < args.generations - 1:
            population = reproduce(
                evaluated,
                archive,
                population_size=args.population,
                generation_of_origin=generation + 1,
                rng=rng,
                hidden_size=args.hidden_size,
                rank=args.rank,
                mutation_sigma=args.mutation_sigma,
                reproduction_weights=reproduction_weights,
                random_injection_rate=random_injection_rate,
                include_default_seed=include_default_seed,
                elitism=elitism,
                allow_structured_operators=allow_structured_operators,
                use_archive_bins_for_parents=args.reproduction_mode != "native",
            )
    save_json({
        objective: max(all_rows, key=lambda row: float(row["metrics"][objective]))
        for objective in PARETO_OBJECTIVES
    }, out_dir / "best_by_objective.json")
    print(f"saved {out_dir / 'archive.json'}")


if __name__ == "__main__":
    main()
