#!/usr/bin/env python3
"""Probe v9 with and without explicit message/carrier channels.

This is intentionally an experimental script, not a new canonical substrate.
It asks two narrow questions:

1. Does adding a minimal message/carrier pathway to v9 produce the accumulating
   fixed-point signature seen in v8?
2. What distinguishes the tuned v9 STRANGE boundary from default fixed-point v9?
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from development.substrates.legacy import (
    DemianNativeV9Substrate,
    SelfLoopRunner,
)

HIDDEN_SIZE = 32
STEPS = 128
PERTURB_STEP = 64
PERTURB_SCALES = (0.25, 0.5, 1.0)
SEEDS = (94, 95, 96, 97)
OUT_DIR = Path("data/substrate_lab/v9_message_carrier_strange_20260508")


class ExperimentalV9MessageCarrier(DemianNativeV9Substrate):
    """Minimal v9 extension with explicit message and carrier state.

    The base v9 fast/slow/control loop is preserved. The added pathway is:
    fast -> message -> carrier -> slow, with optional message/carrier readout
    back into the exposed surface and fast update.
    """

    def __init__(
        self,
        hidden_size: int,
        message_decay: float = 0.88,
        carrier_decay: float = 0.96,
        fast_to_message_scale: float = 0.25,
        fast_to_carrier_scale: float = 0.0,
        message_to_carrier_scale: float = 0.35,
        carrier_to_slow_scale: float = 0.25,
        message_to_fast_scale: float = 0.12,
        carrier_to_fast_scale: float = 0.08,
        message_readout_scale: float = 0.08,
        carrier_readout_scale: float = 0.12,
        binding_start_step: int = 1,
        initial_message_scale: float | None = None,
        initial_carrier_scale: float | None = None,
        release_step: int | None = None,
        release_duration: int = 1,
        release_gain: float = 0.0,
        release_policy: str = "manual",
        release_threshold: float = 0.0,
        release_temperature: float = 1.0,
        **kwargs: float,
    ):
        super().__init__(hidden_size, **kwargs)
        self.message_decay = message_decay
        self.carrier_decay = carrier_decay
        self.fast_to_message_scale = fast_to_message_scale
        self.fast_to_carrier_scale = fast_to_carrier_scale
        self.message_to_carrier_scale = message_to_carrier_scale
        self.carrier_to_slow_scale = carrier_to_slow_scale
        self.message_to_fast_scale = message_to_fast_scale
        self.carrier_to_fast_scale = carrier_to_fast_scale
        self.message_readout_scale = message_readout_scale
        self.carrier_readout_scale = carrier_readout_scale
        self.binding_start_step = binding_start_step
        self.initial_message_scale = self.init_scale if initial_message_scale is None else initial_message_scale
        self.initial_carrier_scale = self.init_scale if initial_carrier_scale is None else initial_carrier_scale
        self.release_step = release_step
        self.release_duration = release_duration
        self.release_gain = release_gain
        self.release_policy = release_policy
        self.release_threshold = release_threshold
        self.release_temperature = max(release_temperature, 1e-6)
        self._step_index = 0

        self.message_gate = nn.Linear(hidden_size, hidden_size)
        self.message_mix = nn.Linear(hidden_size, hidden_size)
        self.carrier_gate = nn.Linear(hidden_size, hidden_size)
        self.carrier_mix = nn.Linear(hidden_size, hidden_size)

        self.fast_to_message = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fast_to_carrier = nn.Linear(hidden_size, hidden_size, bias=False)
        self.message_to_carrier = nn.Linear(hidden_size, hidden_size, bias=False)
        self.carrier_to_slow = nn.Linear(hidden_size, hidden_size, bias=False)
        self.message_to_fast = nn.Linear(hidden_size, hidden_size, bias=False)
        self.carrier_to_fast = nn.Linear(hidden_size, hidden_size, bias=False)
        self.message_readout = nn.Linear(hidden_size, hidden_size, bias=False)
        self.carrier_readout = nn.Linear(hidden_size, hidden_size, bias=False)
        self.release_gate = nn.Linear(hidden_size * 5, 1)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, control = super().initial_state(batch_size, device)
        self._step_index = 0
        message = torch.randn(batch_size, self.hidden_size, device=device) * self.initial_message_scale
        carrier = torch.randn(batch_size, self.hidden_size, device=device) * self.initial_carrier_scale
        return fast, slow, control, message, carrier

    def state_components(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        fast, slow, control, message, carrier = state
        return {
            "fast": fast,
            "slow": slow,
            "control": control,
            "message": message,
            "carrier": carrier,
        }

    def state_vector(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fast, _, _, message, carrier = state
        return (
            fast
            + self.message_readout_scale * torch.tanh(self.message_readout(message))
            + self.carrier_readout_scale * torch.tanh(self.carrier_readout(carrier))
        )

    def inject_coupling_message(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        message: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, control, msg_state, carrier = state
        delta = strength * message
        if delta.shape != fast.shape:
            delta = delta.view(fast.shape)
        return fast + delta, slow, control, msg_state, carrier

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._step_index += 1
        fast, slow, control, message, carrier = state

        if self._step_index < self.binding_start_step:
            new_fast, new_slow, new_control = super().step((fast, slow, control))
            self._step_aux.update(
                {
                    "message_write_mean": 0.0,
                    "carrier_write_mean": 0.0,
                    "fast_to_message_norm": 0.0,
                    "fast_to_carrier_norm": 0.0,
                    "message_to_carrier_norm": 0.0,
                    "carrier_to_slow_norm": 0.0,
                    "message_to_fast_norm": 0.0,
                    "carrier_to_fast_norm": 0.0,
                    "carrier_residual_norm": float(torch.norm(self.carrier_decay * carrier).item()),
                    "release_open": 0.0,
                    "release_open_mean": 0.0,
                    "release_strength_mean": 0.0,
                    "release_pressure_mean": float(
                        (
                            torch.norm(message, dim=-1, keepdim=True)
                            + torch.norm(carrier, dim=-1, keepdim=True)
                        ).mean().item()
                        / math.sqrt(max(self.hidden_size, 1))
                    ),
                    "release_drive_mean": 0.0,
                    "release_bias_norm": 0.0,
                    "endogenous_release_norm": 0.0,
                    "binding_active": 0.0,
                }
            )
            return new_fast, new_slow, new_control, message, carrier

        message_gate = torch.sigmoid(self.message_gate(message))
        message_candidate = torch.tanh(self.message_mix(message))
        fast_to_message = self.fast_to_message_scale * torch.tanh(self.fast_to_message(fast))
        new_message = (
            self.message_decay * message
            + message_gate * (message_candidate + fast_to_message)
        )

        carrier_gate = torch.sigmoid(self.carrier_gate(carrier))
        carrier_candidate = torch.tanh(self.carrier_mix(carrier))
        message_to_carrier = self.message_to_carrier_scale * torch.tanh(
            self.message_to_carrier(new_message)
        )
        fast_to_carrier = self.fast_to_carrier_scale * torch.tanh(
            self.fast_to_carrier(fast)
        )
        new_carrier = (
            self.carrier_decay * carrier
            + carrier_gate * (carrier_candidate + message_to_carrier + fast_to_carrier)
        )

        slow_gate = torch.sigmoid(self.slow_gate(slow))
        slow_candidate = torch.tanh(self.slow_mix(slow))
        fast_to_slow_bias = self.slow_readout_scale * torch.tanh(self.fast_to_slow(fast))
        carrier_to_slow = self.carrier_to_slow_scale * torch.tanh(
            self.carrier_to_slow(new_carrier)
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * (slow_candidate + fast_to_slow_bias + carrier_to_slow)
        )

        control_gate = torch.sigmoid(self.control_gate(control))
        control_candidate = torch.tanh(self.control_mix(control))
        fast_slow_bias = torch.tanh(self.fast_slow_to_control(torch.cat([fast, slow], dim=-1)))
        new_control = (
            self.control_decay * control
            + control_gate * (control_candidate + fast_slow_bias)
        )

        fast_gate = torch.sigmoid(self.fast_gate(fast))
        fast_candidate = torch.tanh(self.fast_mix(fast))
        control_bias = self.control_to_fast_scale * torch.tanh(self.control_readout(new_control))
        message_fast_bias = self.message_to_fast_scale * torch.tanh(
            self.message_to_fast(new_message)
        )
        carrier_fast_bias = self.carrier_to_fast_scale * torch.tanh(
            self.carrier_to_fast(new_carrier)
        )
        release_open, release_strength, release_pressure, release_drive = self._release_signal(
            fast,
            new_slow,
            new_control,
            new_message,
            new_carrier,
        )
        release_bias = release_strength * torch.tanh(
            self.message_to_fast(new_message) + self.carrier_to_fast(new_carrier)
        )
        new_fast = (
            (1.0 - fast_gate) * fast
            + fast_gate
            * self.state_gain
            * torch.tanh(
                fast_candidate
                + control_bias
                + message_fast_bias
                + carrier_fast_bias
                + release_bias
            )
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
            "fast_to_carrier_norm": float(torch.norm(fast_to_carrier).item()),
            "message_to_carrier_norm": float(torch.norm(message_to_carrier).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "message_to_fast_norm": float(torch.norm(message_fast_bias).item()),
            "carrier_to_fast_norm": float(torch.norm(carrier_fast_bias).item()),
            "release_open": float(release_open.item() if release_open.numel() == 1 else release_open.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "release_pressure_mean": float(release_pressure.mean().item()),
            "release_drive_mean": float(release_drive.mean().item()),
            "release_bias_norm": float(torch.norm(release_bias).item()),
            "endogenous_release_norm": float(torch.norm(release_bias).item()),
            "carrier_residual_norm": float(torch.norm(self.carrier_decay * carrier).item()),
            "binding_active": 1.0,
        }

        return new_fast, new_slow, new_control, new_message, new_carrier

    def _release_signal(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        control: torch.Tensor,
        message: torch.Tensor,
        carrier: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros((message.shape[0], 1), device=message.device)
        pressure = (
            torch.norm(message, dim=-1, keepdim=True)
            + torch.norm(carrier, dim=-1, keepdim=True)
        ) / math.sqrt(max(self.hidden_size, 1))
        if self.release_gain <= 0.0:
            return zeros, zeros, pressure, zeros
        if self.release_policy == "manual":
            if self.release_step is None:
                return zeros, zeros, pressure, zeros
            is_open = self.release_step <= self._step_index < self.release_step + self.release_duration
            open_gate = torch.ones((message.shape[0], 1), device=message.device) if is_open else zeros
            return open_gate, self.release_gain * open_gate, pressure, open_gate
        if self.release_policy == "pressure":
            open_gate = (pressure >= self.release_threshold).float()
            return open_gate, self.release_gain * open_gate, pressure, open_gate
        if self.release_policy == "learned":
            control_view = torch.tanh(self.control_readout(control))
            gate_input = torch.cat([fast, slow, control_view, message, carrier], dim=-1)
            drive = self.release_gate(gate_input)
            open_gate = torch.sigmoid((drive - self.release_threshold) / self.release_temperature)
            strength = self.release_gain * torch.relu(open_gate - 0.5)
            return open_gate, strength, pressure, drive
        raise ValueError(f"unknown release_policy: {self.release_policy}")


def run_variant(
    label: str,
    cls: type[DemianNativeV9Substrate],
    kwargs: dict[str, float],
) -> dict[str, Any]:
    clean_runs = []
    perturb_runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = cls(HIDDEN_SIZE, **kwargs)
        runner = SelfLoopRunner(model, device="cpu")
        clean_traj, clean_summary, clean_final = runner.run(
            steps=STEPS,
            seed=seed,
        )
        clean_runs.append(
            {
                "seed": seed,
                "summary": asdict(clean_summary),
                "strange_diagnostics": strange_diagnostics(clean_traj, clean_summary),
            }
        )

        for scale in PERTURB_SCALES:
            pert_traj, pert_summary, pert_final = runner.run(
                steps=STEPS,
                seed=seed,
                perturb_step=PERTURB_STEP,
                perturb_scale=scale,
            )
            final_cosine = float(
                torch.nn.functional.cosine_similarity(clean_final, pert_final, dim=0).item()
            )
            final_l2_gap = float(torch.norm(clean_final - pert_final).item())
            perturb_runs.append(
                {
                    "seed": seed,
                    "scale": scale,
                    "final_cosine": final_cosine,
                    "final_l2_gap": final_l2_gap,
                    "summary": asdict(pert_summary),
                    "strange_diagnostics": strange_diagnostics(pert_traj, pert_summary),
                }
            )

    return {
        "label": label,
        "class": cls.__name__,
        "kwargs": kwargs,
        "clean_runs": clean_runs,
        "perturb_runs": perturb_runs,
        "aggregate": aggregate_variant(clean_runs, perturb_runs),
    }


def strange_diagnostics(trajectory, summary) -> dict[str, Any]:
    deltas = [step.residual_delta for step in trajectory]
    coherences = [step.temporal_coherence for step in trajectory]
    fast_norms = [step.fast_state_norm for step in trajectory]
    slow_norms = [step.slow_state_norm for step in trajectory]
    message_norms = [step.message_state_norm for step in trajectory]
    route_metrics = [step.route_metrics or {} for step in trajectory]
    carrier_norms = [float(row.get("carrier_residual_norm", 0.0)) for row in route_metrics]
    tail = max(16, len(trajectory) // 4)

    return {
        "attractor_type": summary.attractor_type,
        "tail_mean_delta": mean(deltas[-tail:]),
        "tail_std_delta": std(deltas[-tail:]),
        "tail_min_coherence": min(coherences[-tail:]),
        "tail_mean_coherence": mean(coherences[-tail:]),
        "tail_mean_fast_norm": mean(fast_norms[-tail:]),
        "tail_std_fast_norm": std(fast_norms[-tail:]),
        "tail_mean_slow_norm": mean(slow_norms[-tail:]),
        "tail_std_slow_norm": std(slow_norms[-tail:]),
        "tail_mean_message_norm": mean(message_norms[-tail:]),
        "tail_std_message_norm": std(message_norms[-tail:]),
        "tail_mean_carrier_residual": mean(carrier_norms[-tail:]),
        "tail_std_carrier_residual": std(carrier_norms[-tail:]),
        "max_delta": max(deltas),
        "max_fast_norm": max(fast_norms),
        "max_slow_norm": max(slow_norms),
        "max_message_norm": max(message_norms),
        "max_carrier_residual": max(carrier_norms),
        "cycle_period": summary.cycle_period,
        "covariance_rank": summary.covariance_rank,
        "flow_dimension": summary.flow_dimension,
        "compression_ratio": summary.compression_ratio,
    }


def aggregate_variant(clean_runs, perturb_runs) -> dict[str, Any]:
    summaries = [run["summary"] for run in clean_runs]
    diagnostics = [run["strange_diagnostics"] for run in clean_runs]
    by_scale: dict[str, list[dict[str, Any]]] = {}
    for run in perturb_runs:
        by_scale.setdefault(str(run["scale"]), []).append(run)
    return {
        "attractor_counts": counts(row["attractor_type"] for row in summaries),
        "mean_norm": mean(row["mean_norm"] for row in summaries),
        "mean_delta": mean(row["mean_delta"] for row in summaries),
        "mean_coherence": mean(row["mean_coherence"] for row in summaries),
        "mean_covariance_rank": mean(row["covariance_rank"] for row in summaries),
        "mean_flow_dimension": mean(row["flow_dimension"] for row in summaries),
        "mean_compression_ratio": mean(row["compression_ratio"] for row in summaries),
        "mean_fast_norm": mean(row["mean_fast_norm"] for row in summaries),
        "mean_slow_norm": mean(row["mean_slow_norm"] for row in summaries),
        "mean_message_norm": mean(row["mean_message_norm"] for row in summaries),
        "max_message_norm": mean(row["max_message_norm"] for row in summaries),
        "tail_mean_carrier_residual": mean(row["tail_mean_carrier_residual"] for row in diagnostics),
        "max_carrier_residual": mean(row["max_carrier_residual"] for row in diagnostics),
        "perturbation": {
            scale: {
                "mean_final_cosine": mean(row["final_cosine"] for row in rows),
                "mean_final_l2_gap": mean(row["final_l2_gap"] for row in rows),
                "attractor_counts": counts(row["summary"]["attractor_type"] for row in rows),
            }
            for scale, rows in sorted(by_scale.items(), key=lambda item: float(item[0]))
        },
    }


def run_channel_factorial(tuned_strange: dict[str, float]) -> list[dict[str, Any]]:
    """Separate no-channel, message-only, carrier-only, and full pathways."""
    channel_variants = [
        ("no_message_no_carrier", DemianNativeV9Substrate, tuned_strange),
        (
            "message_only",
            ExperimentalV9MessageCarrier,
            {
                **tuned_strange,
                "message_decay": 0.94,
                "carrier_decay": 0.0,
                "fast_to_message_scale": 0.35,
                "fast_to_carrier_scale": 0.0,
                "message_to_carrier_scale": 0.0,
                "carrier_to_slow_scale": 0.0,
                "message_to_fast_scale": 0.22,
                "carrier_to_fast_scale": 0.0,
                "message_readout_scale": 0.14,
                "carrier_readout_scale": 0.0,
            },
        ),
        (
            "carrier_only",
            ExperimentalV9MessageCarrier,
            {
                **tuned_strange,
                "message_decay": 0.0,
                "carrier_decay": 0.985,
                "fast_to_message_scale": 0.0,
                "fast_to_carrier_scale": 0.35,
                "message_to_carrier_scale": 0.0,
                "carrier_to_slow_scale": 0.45,
                "message_to_fast_scale": 0.0,
                "carrier_to_fast_scale": 0.16,
                "message_readout_scale": 0.0,
                "carrier_readout_scale": 0.22,
            },
        ),
        (
            "message_plus_carrier",
            ExperimentalV9MessageCarrier,
            accumulator_kwargs(tuned_strange),
        ),
    ]
    return [run_variant(label, cls, kwargs) for label, cls, kwargs in channel_variants]


def run_carrier_decay_sweep(tuned_strange: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for carrier_decay in (0.80, 0.90, 0.96, 0.985, 0.995):
        kwargs = {
            **accumulator_kwargs(tuned_strange),
            "carrier_decay": carrier_decay,
        }
        result = run_variant(f"carrier_decay_{carrier_decay}", ExperimentalV9MessageCarrier, kwargs)
        rows.append(
            {
                "carrier_decay": carrier_decay,
                "kwargs": kwargs,
                "aggregate": result["aggregate"],
                "clean_runs": result["clean_runs"],
            }
        )
    return rows


def run_strange_robustness_search() -> list[dict[str, Any]]:
    rows = []
    for state_gain in (1.32, 1.40, 1.48, 1.56, 1.65):
        for control_to_fast_scale in (0.45, 0.50, 0.62, 0.70):
            for control_decay in (0.45, 0.58, 0.68, 0.78):
                for slow_decay in (0.86, 0.88, 0.92):
                    kwargs = {
                        "init_scale": 0.15,
                        "state_gain": state_gain,
                        "slow_decay": slow_decay,
                        "control_decay": control_decay,
                        "slow_readout_scale": 0.2,
                        "control_to_fast_scale": control_to_fast_scale,
                        "fast_to_slow_gate_bias": 0.0,
                    }
                    clean_runs = run_clean_only(DemianNativeV9Substrate, kwargs)
                    aggregate = aggregate_clean_only(clean_runs)
                    aggregate["score"] = strange_search_score(aggregate)
                    rows.append({"kwargs": kwargs, "aggregate": aggregate, "clean_runs": clean_runs})
    rows.sort(key=lambda row: row["aggregate"]["score"], reverse=True)
    return rows[:30]


def run_carrier_on_strange_sweep(tuned_strange: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for carrier_decay in (0.0, 0.80, 0.90, 0.96, 0.985):
        for carrier_to_fast_scale in (0.0, 0.08, 0.16):
            kwargs = {
                **tuned_strange,
                "message_decay": 0.0,
                "carrier_decay": carrier_decay,
                "fast_to_message_scale": 0.0,
                "fast_to_carrier_scale": 0.35,
                "message_to_carrier_scale": 0.0,
                "carrier_to_slow_scale": 0.35,
                "message_to_fast_scale": 0.0,
                "carrier_to_fast_scale": carrier_to_fast_scale,
                "message_readout_scale": 0.0,
                "carrier_readout_scale": 0.16,
            }
            clean_runs = run_clean_only(ExperimentalV9MessageCarrier, kwargs)
            aggregate = aggregate_clean_only(clean_runs)
            aggregate["score"] = strange_search_score(aggregate)
            rows.append({"kwargs": kwargs, "aggregate": aggregate, "clean_runs": clean_runs})
    rows.sort(key=lambda row: row["aggregate"]["score"], reverse=True)
    return rows


def run_potential_transformation_test(tuned_strange: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    binding_steps = (1, 32, 64, 96, 10_000)
    for binding_start_step in binding_steps:
        label = (
            "never_bind"
            if binding_start_step > STEPS
            else f"bind_at_{binding_start_step}"
        )
        kwargs = {
            **accumulator_kwargs(tuned_strange),
            "binding_start_step": binding_start_step,
            "initial_message_scale": 0.0,
            "initial_carrier_scale": 0.0,
        }
        clean_runs = []
        perturb_runs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = ExperimentalV9MessageCarrier(HIDDEN_SIZE, **kwargs)
            runner = SelfLoopRunner(model, device="cpu")
            clean_traj, clean_summary, clean_final = runner.run(steps=STEPS, seed=seed)
            clean_runs.append(
                {
                    "seed": seed,
                    "summary": asdict(clean_summary),
                    "strange_diagnostics": strange_diagnostics(clean_traj, clean_summary),
                    "transformation": transformation_diagnostics(clean_traj, binding_start_step),
                }
            )

            pert_traj, pert_summary, pert_final = runner.run(
                steps=STEPS,
                seed=seed,
                perturb_step=PERTURB_STEP,
                perturb_scale=1.0,
            )
            perturb_runs.append(
                {
                    "seed": seed,
                    "scale": 1.0,
                    "final_cosine": float(
                        torch.nn.functional.cosine_similarity(clean_final, pert_final, dim=0).item()
                    ),
                    "final_l2_gap": float(torch.norm(clean_final - pert_final).item()),
                    "summary": asdict(pert_summary),
                    "strange_diagnostics": strange_diagnostics(pert_traj, pert_summary),
                    "transformation": transformation_diagnostics(pert_traj, binding_start_step),
                }
            )
        aggregate = aggregate_variant(clean_runs, perturb_runs)
        aggregate["transformation"] = aggregate_transformation(clean_runs)
        rows.append(
            {
                "label": label,
                "binding_start_step": binding_start_step,
                "kwargs": kwargs,
                "clean_runs": clean_runs,
                "perturb_runs": perturb_runs,
                "aggregate": aggregate,
            }
        )
    return rows


def run_release_reuse_test(tuned_strange: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    base = {
        **accumulator_kwargs(tuned_strange),
        "binding_start_step": 32,
        "initial_message_scale": 0.0,
        "initial_carrier_scale": 0.0,
    }
    configs = [
        ("bound_no_release", {}),
        ("manual_release_weak", {"release_step": 96, "release_duration": 1, "release_gain": 0.12}),
        ("manual_release_medium", {"release_step": 96, "release_duration": 2, "release_gain": 0.25}),
        ("manual_release_strong", {"release_step": 96, "release_duration": 4, "release_gain": 0.45}),
        ("pressure_release_low", {"release_policy": "pressure", "release_threshold": 16.0, "release_gain": 0.18}),
        ("pressure_release_high", {"release_policy": "pressure", "release_threshold": 32.0, "release_gain": 0.25}),
        ("learned_release_low_threshold", {"release_policy": "learned", "release_threshold": -0.25, "release_temperature": 0.75, "release_gain": 0.30}),
        ("learned_release_neutral", {"release_policy": "learned", "release_threshold": 0.0, "release_temperature": 1.0, "release_gain": 0.30}),
        ("learned_release_high_threshold", {"release_policy": "learned", "release_threshold": 0.25, "release_temperature": 0.75, "release_gain": 0.30}),
    ]
    for label, overrides in configs:
        kwargs = {**base, **overrides}
        clean_runs = []
        perturb_runs = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = ExperimentalV9MessageCarrier(HIDDEN_SIZE, **kwargs)
            runner = SelfLoopRunner(model, device="cpu")
            clean_traj, clean_summary, clean_final = runner.run(steps=STEPS, seed=seed)
            clean_runs.append(
                {
                    "seed": seed,
                    "summary": asdict(clean_summary),
                    "strange_diagnostics": strange_diagnostics(clean_traj, clean_summary),
                    "transformation": transformation_diagnostics(clean_traj, 32),
                    "release": release_diagnostics(clean_traj),
                }
            )

            pert_traj, pert_summary, pert_final = runner.run(
                steps=STEPS,
                seed=seed,
                perturb_step=PERTURB_STEP,
                perturb_scale=1.0,
            )
            perturb_runs.append(
                {
                    "seed": seed,
                    "scale": 1.0,
                    "final_cosine": float(
                        torch.nn.functional.cosine_similarity(clean_final, pert_final, dim=0).item()
                    ),
                    "final_l2_gap": float(torch.norm(clean_final - pert_final).item()),
                    "summary": asdict(pert_summary),
                    "strange_diagnostics": strange_diagnostics(pert_traj, pert_summary),
                    "transformation": transformation_diagnostics(pert_traj, 32),
                    "release": release_diagnostics(pert_traj),
                }
            )
        aggregate = aggregate_variant(clean_runs, perturb_runs)
        aggregate["release"] = aggregate_release(clean_runs)
        rows.append(
            {
                "label": label,
                "kwargs": kwargs,
                "clean_runs": clean_runs,
                "perturb_runs": perturb_runs,
                "aggregate": aggregate,
            }
        )
    return rows


def transformation_diagnostics(trajectory, binding_start_step: int) -> dict[str, float]:
    pre = [
        step for step in trajectory
        if max(1, binding_start_step - 24) <= step.step < binding_start_step
    ]
    post = [
        step for step in trajectory
        if binding_start_step <= step.step < min(STEPS + 1, binding_start_step + 24)
    ]
    tail = trajectory[-32:]

    def route_mean(rows, key: str) -> float:
        return mean((step.route_metrics or {}).get(key, 0.0) for step in rows)

    return {
        "pre_mean_delta": mean(step.residual_delta for step in pre),
        "post_mean_delta": mean(step.residual_delta for step in post),
        "tail_mean_delta": mean(step.residual_delta for step in tail),
        "pre_mean_message_norm": mean(step.message_state_norm for step in pre),
        "post_mean_message_norm": mean(step.message_state_norm for step in post),
        "tail_mean_message_norm": mean(step.message_state_norm for step in tail),
        "pre_mean_carrier_residual": route_mean(pre, "carrier_residual_norm"),
        "post_mean_carrier_residual": route_mean(post, "carrier_residual_norm"),
        "tail_mean_carrier_residual": route_mean(tail, "carrier_residual_norm"),
        "post_binding_active": route_mean(post, "binding_active"),
        "tail_binding_active": route_mean(tail, "binding_active"),
    }


def aggregate_transformation(clean_runs: list[dict[str, Any]]) -> dict[str, float]:
    rows = [run["transformation"] for run in clean_runs]
    return {
        key: mean(row[key] for row in rows)
        for key in rows[0]
    } if rows else {}


def release_diagnostics(trajectory) -> dict[str, float]:
    release_rows = [
        step for step in trajectory
        if float((step.route_metrics or {}).get("release_open", 0.0)) > 0.0
    ]
    post_release_steps = []
    if release_rows:
        first_release = release_rows[0].step
        post_release_steps = [
            step for step in trajectory
            if first_release <= step.step < min(STEPS + 1, first_release + 16)
        ]
    tail = trajectory[-32:]

    def route_mean(rows, key: str) -> float:
        return mean((step.route_metrics or {}).get(key, 0.0) for step in rows)

    return {
        "release_count": float(len(release_rows)),
        "first_release_step": float(release_rows[0].step if release_rows else 0),
        "mean_release_open": route_mean(trajectory, "release_open_mean"),
        "mean_release_strength": route_mean(trajectory, "release_strength_mean"),
        "max_release_strength": max(
            [float((step.route_metrics or {}).get("release_strength_mean", 0.0)) for step in trajectory]
            or [0.0]
        ),
        "mean_release_pressure": route_mean(trajectory, "release_pressure_mean"),
        "mean_release_drive": route_mean(trajectory, "release_drive_mean"),
        "mean_release_bias_norm": route_mean(release_rows, "release_bias_norm"),
        "post_release_mean_delta": mean(step.residual_delta for step in post_release_steps),
        "post_release_max_delta": max([step.residual_delta for step in post_release_steps] or [0.0]),
        "post_release_mean_coherence": mean(step.temporal_coherence for step in post_release_steps),
        "tail_mean_delta": mean(step.residual_delta for step in tail),
        "tail_mean_message_norm": mean(step.message_state_norm for step in tail),
        "tail_mean_carrier_residual": route_mean(tail, "carrier_residual_norm"),
    }


def aggregate_release(clean_runs: list[dict[str, Any]]) -> dict[str, float]:
    rows = [run["release"] for run in clean_runs]
    return {
        key: mean(row[key] for row in rows)
        for key in rows[0]
    } if rows else {}


def run_clean_only(
    cls: type[DemianNativeV9Substrate],
    kwargs: dict[str, float],
) -> list[dict[str, Any]]:
    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = cls(HIDDEN_SIZE, **kwargs)
        runner = SelfLoopRunner(model, device="cpu")
        trajectory, summary, _ = runner.run(steps=STEPS, seed=seed)
        runs.append(
            {
                "seed": seed,
                "summary": asdict(summary),
                "strange_diagnostics": strange_diagnostics(trajectory, summary),
            }
        )
    return runs


def aggregate_clean_only(clean_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_variant(clean_runs, [])


def strange_search_score(aggregate: dict[str, Any]) -> float:
    attractors = aggregate["attractor_counts"]
    strange = attractors.get("STRANGE", 0)
    expanding = attractors.get("EXPANDING", 0)
    fixed = attractors.get("FIXED_POINT", 0)
    return (
        3.0 * strange
        - 4.0 * expanding
        - 0.7 * fixed
        + 4.0 * aggregate["mean_delta"]
        + 1.5 * aggregate["mean_covariance_rank"]
        - 0.001 * aggregate["tail_mean_carrier_residual"]
    )


def accumulator_kwargs(tuned_strange: dict[str, float]) -> dict[str, float]:
    return {
        **tuned_strange,
        "message_decay": 0.94,
        "carrier_decay": 0.985,
        "fast_to_message_scale": 0.35,
        "fast_to_carrier_scale": 0.0,
        "message_to_carrier_scale": 0.65,
        "carrier_to_slow_scale": 0.45,
        "message_to_fast_scale": 0.22,
        "carrier_to_fast_scale": 0.16,
        "message_readout_scale": 0.14,
        "carrier_readout_scale": 0.22,
    }


def counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def mean(values) -> float:
    vals = [float(value) for value in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def std(values) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    avg = mean(vals)
    return float(math.sqrt(sum((value - avg) ** 2 for value in vals) / len(vals)))


def main() -> None:
    tuned_strange = {
        "init_scale": 0.15,
        "state_gain": 1.4,
        "slow_decay": 0.88,
        "control_decay": 0.58,
        "slow_readout_scale": 0.2,
        "control_to_fast_scale": 0.5,
        "fast_to_slow_gate_bias": 0.0,
    }
    variants = [
        ("v9_base_default", DemianNativeV9Substrate, {}),
        ("v9_base_tuned_strange_boundary", DemianNativeV9Substrate, tuned_strange),
        (
            "v9_message_carrier_default",
            ExperimentalV9MessageCarrier,
            {
                "message_decay": 0.88,
                "carrier_decay": 0.96,
                "fast_to_message_scale": 0.25,
                "fast_to_carrier_scale": 0.0,
                "message_to_carrier_scale": 0.35,
                "carrier_to_slow_scale": 0.25,
                "message_to_fast_scale": 0.12,
                "carrier_to_fast_scale": 0.08,
                "message_readout_scale": 0.08,
                "carrier_readout_scale": 0.12,
            },
        ),
        (
            "v9_message_carrier_accumulator",
            ExperimentalV9MessageCarrier,
            accumulator_kwargs(tuned_strange),
        ),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for label, cls, kwargs in variants:
        print(f"running {label}", flush=True)
        result = run_variant(label, cls, kwargs)
        results.append(result)
        print(
            f"  attractors={result['aggregate']['attractor_counts']} "
            f"mean_norm={result['aggregate']['mean_norm']:.5f} "
            f"mean_delta={result['aggregate']['mean_delta']:.5f} "
            f"msg={result['aggregate']['mean_message_norm']:.5f} "
            f"carrier={result['aggregate']['tail_mean_carrier_residual']:.5f}",
            flush=True,
        )

    print("running channel factorial", flush=True)
    channel_factorial = run_channel_factorial(tuned_strange)
    print("running carrier decay sweep", flush=True)
    carrier_decay_sweep = run_carrier_decay_sweep(tuned_strange)
    print("running strange robustness search", flush=True)
    strange_robustness_search = run_strange_robustness_search()
    print("running carrier-on-strange sweep", flush=True)
    carrier_on_strange_sweep = run_carrier_on_strange_sweep(tuned_strange)
    print("running potential transformation test", flush=True)
    potential_transformation = run_potential_transformation_test(tuned_strange)
    print("running release/reuse test", flush=True)
    release_reuse = run_release_reuse_test(tuned_strange)

    payload = {
        "config": {
            "hidden_size": HIDDEN_SIZE,
            "steps": STEPS,
            "perturb_step": PERTURB_STEP,
            "perturb_scales": list(PERTURB_SCALES),
            "seeds": list(SEEDS),
        },
        "variants": results,
        "channel_factorial": channel_factorial,
        "carrier_decay_sweep": carrier_decay_sweep,
        "strange_robustness_search": strange_robustness_search,
        "carrier_on_strange_sweep": carrier_on_strange_sweep,
        "potential_transformation": potential_transformation,
        "release_reuse": release_reuse,
    }
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
