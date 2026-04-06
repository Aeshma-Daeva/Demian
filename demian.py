"""Demian -- a model that experiences its own computation."""
import argparse
import logging
import sys
import uuid
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from demian.vibration import VibrationTracker
from demian.nous import NousInjector
from demian.loop import generate_with_proprioception
from demian.rhythm import FibonacciConsolidator, InjectionScheduler
from demian.dream import DreamSynthesizer

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Demian")
    parser.add_argument("--model", default=None, help="Override model ID")
    parser.add_argument(
        "--no-inject", action="store_true",
        help="Disable proprioceptive injection (control mode)",
    )
    parser.add_argument("--temp", type=float, default=None, help="Override temperature")
    parser.add_argument(
        "--target-dim", type=int, default=None,
        help="Random projection target dimensionality",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    model_id = args.model or cfg.get("proprioceptor_model_id", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    temperature = args.temp if args.temp is not None else cfg.get("temperature", 0.7)
    max_new_tokens = cfg.get("max_new_tokens", 256)
    target_dim = args.target_dim or cfg.get("target_dim", 128)
    injection_scale = cfg.get("injection_scale", 0.1)

    log.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()

    d_model = model.config.hidden_size

    tracker = VibrationTracker(
        d_model=d_model, target_dim=target_dim,
        max_trajectory=max_new_tokens,
    )

    injector = NousInjector(
        model=model,
        tracker=tracker,
        max_memory_length=cfg.get("max_memory_length", 16),
        injection_scale=cfg.get("injection_scale", 0.1),
    )

    scheduler = InjectionScheduler(
        max_steps=max_new_tokens,
        base_gap=2,
        max_gap=13,
    )

    damping = cfg.get("injection_damping", 0.3)

    consolidator = FibonacciConsolidator(
        tracker=tracker,
        target_dim=target_dim,
        fibonacci_intervals=cfg.get("consolidation_intervals", [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]),
    )

    dream = DreamSynthesizer(consolidator)
    tendency = dream.synthesize()
    if tendency is not None:
        injector.record_step(tendency.detach().cpu())
        injector._damped_residual = tendency.detach().cpu().clone()
        summary = dream.dream_summary
        if summary:
            log.info("Dream summary:\n%s", summary)
    else:
        log.info("No prior dream state -- first session")

    engagement_id = str(uuid.uuid4())[:8]
    log.info("Session engagement_id=%s  d_model=%d  target_dim=%d", engagement_id, d_model, target_dim)

    print()
    print("=" * 60)
    print("  Demian -- the model experiences itself.")
    print()
    out_model = model_id
    print("  Model:      " + out_model)
    print("  d_model:    " + str(d_model))
    print("  target_dim: " + str(target_dim))
    inj_status = "active" if not args.no_inject else "disabled (control)"
    print("  Injection:  " + inj_status)
    if tendency is not None:
        dream_text = dream.dream_summary
        if dream_text:
            print("  Dream:      loaded")
    else:
        print("  Dream:      -- (no prior sessions)")
    print("=" * 60)
    print()

    try:
        turn_count = 0
        while True:
            user_input = input("you > ")
            if not user_input.strip():
                continue

            turn_count += 1
            text, snapshots = generate_with_proprioception(
                model=model,
                tokenizer=tokenizer,
                tracker=tracker,
                injector=injector,
                prompt=user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                proprio_inject=not args.no_inject,
                device=str(model.device),
                scheduler=scheduler,
                damping=damping,
            )

            print()
            print(text)
            print("  [" + str(len(snapshots)) + " steps of computational trajectory]")

            compressed = consolidator.on_turn(engagement_id)
            if compressed:
                print("  [Consolidation turn " + str(turn_count) + ": " + compressed.compression_type + "]")
            print()

    except KeyboardInterrupt:
        print("\nSession ended.")
        sys.exit(0)
    except EOFError:
        print("\nSession ended.")
        sys.exit(0)


if __name__ == "__main__":
    main()
