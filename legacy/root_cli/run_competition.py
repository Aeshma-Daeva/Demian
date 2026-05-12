#!/usr/bin/env python3
"""Multi-agent Mamba competition v3 — CLI entry point.

Variable population. True death (internal collapse). True fission (h inheritance).
Field communication: entropy-gradient, SEND/RECEIVE phase alternation.
H-coupling: sustained affinity unlocks bidirectional genome blend.

No predator. No external judgment. Attractor gravity is the only pressure.

Usage:
    venv/bin/python3 run_competition.py \
        --n-agents 13 \
        --rounds 500 \
        --base-steps 15 \
        --min-steps 3 \
        --machine-driver \
        --death-rdelta 1e-4 \
        --death-window 50 \
        --fission-depth 200 \
        --fission-richness 5e-3 \
        --fission-novelty 0.7 \
        --fuse-threshold 0.85 \
        --fuse-rounds 10 \
        --fuse-alpha 0.05 \
        --emit-cost 0.02 \
        --data-dir data/competition_v3

Smoke test:
    venv/bin/python3 run_competition.py \
        --n-agents 5 --rounds 10 --base-steps 5 --min-steps 2 \
        --machine-driver --data-dir data/competition_v3_smoke
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml

from legacy.demian_runtime.competition import Competition


def parse_args():
    p = argparse.ArgumentParser(description="Demian Multi-Agent Competition v3")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--n-agents", type=int, default=13)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--base-steps", type=int, default=15,
                   help="Steps for depth=1 agent; total budget = n_agents × base_steps")
    p.add_argument("--min-steps", type=int, default=3,
                   help="Floor step budget for any agent")

    # Death
    p.add_argument("--death-rdelta", type=float, default=1e-4,
                   help="rdelta below this for death_window steps → death")
    p.add_argument("--death-window", type=int, default=50,
                   help="Steps of collapse history required to trigger death")

    # Fission
    p.add_argument("--fission-depth", type=int, default=200,
                   help="Minimum step_count before agent can fission")
    p.add_argument("--fission-richness", type=float, default=5e-3,
                   help="Minimum mean rdelta to qualify for fission")
    p.add_argument("--fission-novelty", type=float, default=0.7,
                   help="Max cosine similarity to any neighbor for fission (novelty gate)")
    p.add_argument("--fission-cooldown", type=int, default=20,
                   help="Rounds between fissions per agent. Prevents every-round cloning.")

    # H-coupling
    p.add_argument("--fuse-threshold", type=float, default=0.85,
                   help="Cosine threshold for affinity tracking")
    p.add_argument("--fuse-rounds", type=int, default=10,
                   help="Consecutive rounds above threshold before h-coupling unlocks")
    p.add_argument("--fuse-alpha", type=float, default=0.05,
                   help="H-coupling blend factor (small — genome drift, not replacement)")

    # Field
    p.add_argument("--emit-cost", type=float, default=0.02,
                   help="Sender homogenization cost per broadcast")

    # Initial diversity
    p.add_argument("--init-noise", type=float, default=0.3,
                   help="Per-agent noise on initial residual (fraction of residual norm). "
                        "Breaks symmetry so agents don't all converge to same attractor.")

    # Hebbian
    p.add_argument("--hebbian", action="store_true",
                   help="Enable per-agent personal LoRA (x_proj+out_proj). "
                        "Fast weights: decay each step, blank on fission. "
                        "Budget allocation switches to fitness×(1/crowding).")
    p.add_argument("--hebbian-decay", type=float, default=0.01,
                   help="Per-round A decay rate (fast-weight forgetting). "
                        "Applied once after BCM step, not per step.")
    p.add_argument("--hebbian-eta", type=float, default=1e-5,
                   help="BCM base learning rate")
    p.add_argument("--hebbian-scale", type=float, default=0.01,
                   help="LoRA output scale")
    p.add_argument("--hebbian-rank", type=int, default=4,
                   help="LoRA rank")
    p.add_argument("--fission-weight-threshold", type=float, default=0.01,
                   help="Max ||W_personal||_F for fission (h-inheritance validity gate)")
    p.add_argument("--hebbian-layer-stride", type=int, default=8,
                   help="Patch every Nth layer. stride=8 on 64-layer Mamba → 16 adapters")

    # Infrastructure
    p.add_argument("--machine-driver", action="store_true",
                   help="Enable MachineDriver (required for omega → K phase switching)")
    p.add_argument("--offload-cache", action="store_true")
    p.add_argument("--data-dir", type=str, default="data/competition_v3")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    model_id = args.model or cfg.get("mamba_model_id", "state-spaces/mamba-2.8b-hf")

    print(f"\n  Loading: {model_id}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)  # noqa: F841
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    d_model = model.config.hidden_size
    device = str(next(model.parameters()).device)

    print(f"  d_model={d_model} | device={device}")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        used_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM: {used_gb:.1f}/{vram_gb:.1f} GB after model load")

    print(f"  Agents: {args.n_agents} (variable — grows/shrinks)")
    print(f"  Budget: {args.n_agents * args.base_steps} steps/round (depth-proportional)")
    print(f"  Death: rdelta<{args.death_rdelta} or gate_frozen | window={args.death_window}")
    print(f"  Fission: depth>{args.fission_depth} richness>{args.fission_richness} novelty<{args.fission_novelty}")
    print(f"  H-coupling: cos>{args.fuse_threshold} for {args.fuse_rounds}r → α={args.fuse_alpha}")
    print(f"  Emit cost: {args.emit_cost} | Machine driver: {args.machine_driver}")

    competition = Competition(
        n_agents=args.n_agents,
        data_dir=Path(args.data_dir),
        d_model=d_model,
        device=device,
        base_steps=args.base_steps,
        min_steps=args.min_steps,
        use_machine_driver=args.machine_driver,
        death_rdelta=args.death_rdelta,
        death_window=args.death_window,
        fission_depth=args.fission_depth,
        fission_richness=args.fission_richness,
        fission_novelty=args.fission_novelty,
        fission_weight_threshold=args.fission_weight_threshold,
        fission_cooldown=args.fission_cooldown,
        fuse_threshold=args.fuse_threshold,
        fuse_rounds=args.fuse_rounds,
        fuse_alpha=args.fuse_alpha,
        emit_cost=args.emit_cost,
        offload_cache=args.offload_cache,
        init_noise=args.init_noise,
        use_hebbian=args.hebbian,
        hebbian_decay=args.hebbian_decay,
        hebbian_eta=args.hebbian_eta,
        hebbian_scale=args.hebbian_scale,
        hebbian_rank=args.hebbian_rank,
        hebbian_layer_stride=args.hebbian_layer_stride,
    )

    competition.initialize_agents(model, device)
    competition.run(model, n_rounds=args.rounds)


if __name__ == "__main__":
    main()
