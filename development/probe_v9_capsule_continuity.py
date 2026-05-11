#!/usr/bin/env python3
"""Probe capsule continuity on canonical v9 and v9 five-channel substrates.

This isolates the old v7.4 resume-capsule idea on smaller substrate surfaces.
It compares uninterrupted continuation with:

- full capsule: restored model body and full internal state
- surface-only: empty state with only the exposed surface rewritten
- body-only: restored model body with empty state
- component-only: one internal component restored into an empty state

The result is a compact JSON artifact suitable for a 2D front-page figure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.probe_v9_message_carrier_strange import ExperimentalV9MessageCarrier
from development.substrates.legacy import DemianNativeV9Substrate


ModelFactory = Callable[[int], torch.nn.Module]


def clone_state(state: object) -> object:
    if isinstance(state, tuple):
        return tuple(clone_state(item) for item in state)
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    return state


def step_trace(
    model: Any,
    state: object,
    steps: int,
) -> tuple[object, list[torch.Tensor], list[dict[str, float]]]:
    vectors: list[torch.Tensor] = []
    route_metrics: list[dict[str, float]] = []
    with torch.no_grad():
        for _ in range(steps):
            state = model.step(state)
            vectors.append(model.state_vector(state).view(-1).detach().float().cpu())
            route_metrics.append(dict(model.step_aux()))
    return state, vectors, route_metrics


def set_step_index(model: Any, step_index: int) -> None:
    if hasattr(model, "_step_index"):
        model._step_index = step_index


def empty_state(model: Any, seed: int, device: torch.device) -> object:
    torch.manual_seed(seed)
    return model.initial_state(1, device)


def mix_component_state(base_state: object, capsule_state: object, index: int) -> object:
    if not isinstance(base_state, tuple) or not isinstance(capsule_state, tuple):
        if index != 0:
            raise IndexError(index)
        return clone_state(capsule_state)
    mixed = list(clone_state(base_state))
    mixed[index] = clone_state(capsule_state[index])
    return tuple(mixed)


def final_cosine(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> float:
    if not reference or not candidate:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(reference[-1], candidate[-1], dim=0).item())


def final_l2_gap(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> float:
    if not reference or not candidate:
        return 0.0
    return float(torch.norm(reference[-1] - candidate[-1]).item())


def mean_step_gap(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> float:
    gaps = [
        float(torch.norm(left - right).item()) / math.sqrt(max(left.shape[0], 1))
        for left, right in zip(reference, candidate)
    ]
    return float(np.mean(gaps)) if gaps else 0.0


def summarize_arm(
    reference: list[torch.Tensor],
    candidate: list[torch.Tensor],
) -> dict[str, float]:
    return {
        "final_cosine_vs_uninterrupted": final_cosine(reference, candidate),
        "final_l2_gap_vs_uninterrupted": final_l2_gap(reference, candidate),
        "mean_step_gap_vs_uninterrupted": mean_step_gap(reference, candidate),
    }


def run_capsule_probe(
    label: str,
    factory: ModelFactory,
    component_names: tuple[str, ...],
    hidden_size: int,
    seed: int,
    pause_steps: int,
    resume_steps: int,
    device_name: str,
) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    source = factory(hidden_size).to(device=device, dtype=torch.float32)
    pause_state = source.initial_state(1, device)
    pause_state, _pre_vectors, pre_metrics = step_trace(source, pause_state, pause_steps)
    pause_surface = source.state_vector(pause_state).detach().clone()
    pause_body = {name: value.detach().clone() for name, value in source.state_dict().items()}

    set_step_index(source, pause_steps)
    _uninterrupted_state, uninterrupted_vectors, uninterrupted_metrics = step_trace(
        source,
        clone_state(pause_state),
        resume_steps,
    )

    def run_arm(
        *,
        restore_body: bool,
        restore_full_state: bool,
        restore_surface: bool,
        component_index: int | None = None,
    ) -> dict[str, float]:
        model = factory(hidden_size).to(device=device, dtype=torch.float32)
        if restore_body:
            model.load_state_dict(pause_body)
        state = empty_state(model, seed + 10_000, device)
        if restore_full_state:
            state = clone_state(pause_state)
        if component_index is not None:
            state = mix_component_state(state, pause_state, component_index)
        if restore_surface:
            state = model.write_surface_state(state, pause_surface)
        set_step_index(model, pause_steps)
        _state, vectors, _metrics = step_trace(model, state, resume_steps)
        return summarize_arm(uninterrupted_vectors, vectors)

    arms: dict[str, dict[str, float]] = {
        "full_capsule": run_arm(
            restore_body=True,
            restore_full_state=True,
            restore_surface=False,
        ),
        "state_only": run_arm(
            restore_body=False,
            restore_full_state=True,
            restore_surface=False,
        ),
        "body_only": run_arm(
            restore_body=True,
            restore_full_state=False,
            restore_surface=False,
        ),
        "surface_only": run_arm(
            restore_body=False,
            restore_full_state=False,
            restore_surface=True,
        ),
        "body_surface": run_arm(
            restore_body=True,
            restore_full_state=False,
            restore_surface=True,
        ),
    }
    for index, name in enumerate(component_names):
        arms[f"{name}_only"] = run_arm(
            restore_body=True,
            restore_full_state=False,
            restore_surface=False,
            component_index=index,
        )

    capsule_gap = arms["full_capsule"]["mean_step_gap_vs_uninterrupted"]
    surface_gap = arms["surface_only"]["mean_step_gap_vs_uninterrupted"]
    surface_gap_ratio = None
    if capsule_gap > 1e-9:
        surface_gap_ratio = float(surface_gap / capsule_gap)
    return {
        "label": label,
        "config": {
            "hidden_size": hidden_size,
            "seed": seed,
            "pause_steps": pause_steps,
            "resume_steps": resume_steps,
            "device": device_name,
            "component_names": list(component_names),
        },
        "pause": {
            "surface_norm": float(torch.norm(pause_surface).item()),
            "last_route_metrics": pre_metrics[-1] if pre_metrics else {},
        },
        "arms": arms,
        "continuity_advantage": {
            "surface_gap_minus_capsule_gap": float(surface_gap - capsule_gap),
            "surface_gap_ratio": surface_gap_ratio,
        },
        "uninterrupted_tail_metrics": uninterrupted_metrics[-1] if uninterrupted_metrics else {},
    }


def default_v9_five_channel(hidden_size: int) -> ExperimentalV9MessageCarrier:
    return ExperimentalV9MessageCarrier(
        hidden_size,
        initial_message_scale=0.0,
        initial_carrier_scale=0.0,
        release_policy="learned",
        release_gain=0.0,
        binding_start_step=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--pause-steps", type=int, default=64)
    parser.add_argument("--resume-steps", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="data/substrate_lab/v9_capsule_continuity_20260511/summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "experiment": "v9-capsule-continuity",
        "description": (
            "Capsule resume probe on canonical v9 and v9 five-channel; "
            "full state resume is compared with surface-only and component-only resumes."
        ),
        "runs": [
            run_capsule_probe(
                "demian_native_v9",
                lambda hidden_size: DemianNativeV9Substrate(hidden_size),
                ("fast", "slow", "control"),
                args.hidden_size,
                args.seed,
                args.pause_steps,
                args.resume_steps,
                args.device,
            ),
            run_capsule_probe(
                "v9_five_channel",
                default_v9_five_channel,
                ("fast", "slow", "control", "message", "carrier"),
                args.hidden_size,
                args.seed,
                args.pause_steps,
                args.resume_steps,
                args.device,
            ),
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for run in payload["runs"]:
        capsule = run["arms"]["full_capsule"]
        surface = run["arms"]["surface_only"]
        ratio = run["continuity_advantage"]["surface_gap_ratio"]
        ratio_text = f"{ratio:.3f}" if ratio is not None else "undefined"
        print(
            f"{run['label']}: capsule_cos={capsule['final_cosine_vs_uninterrupted']:.6f} "
            f"surface_cos={surface['final_cosine_vs_uninterrupted']:.6f} "
            f"surface_gap_ratio={ratio_text}"
        )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
