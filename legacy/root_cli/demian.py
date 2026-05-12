"""Demian -- a model that experiences its own computation."""
import argparse
import logging
import sys
import uuid
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from legacy.demian_runtime.vibration import VibrationTracker
from legacy.demian_runtime.nous import NousInjector
from legacy.demian_runtime.loop import generate_with_proprioception
from legacy.demian_runtime.rhythm import FibonacciConsolidator, InjectionScheduler
from legacy.demian_runtime.dream import DreamSynthesizer
from legacy.demian_runtime.continuity import ContinuityRunner

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
    parser.add_argument(
        "--warmup", type=int, default=0,
        help="Pre-generation proprioceptive steps before sampling",
    )
    parser.add_argument(
        "--blend-mode", default="append", choices=["append", "additive"],
        help="KV injection mode: append (new positions) or additive (perturb)",
    )
    parser.add_argument(
        "--inject-mode", default="continuous", choices=["continuous", "one", "spaced"],
        help="When to inject: continuous (every step), one (step 0 only), spaced (Fibonacci)",
    )
    parser.add_argument(
        "--random-inject", action="store_true",
        help="Inject random Gaussian vectors instead of residuals (control)",
    )
    parser.add_argument(
        "--continuity", action="store_true",
        help="Run continuity mode: repeated autos with persistent dream state",
    )
    parser.add_argument(
        "--runs", type=int, default=1000,
        help="Number of continuity runs (default: 1000)",
    )
    parser.add_argument("--prompt", default=None, help="Continuity prompt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
    model_id = args.model or cfg.get("proprioceptor_model_id", "Qwen/Qwen2.5-3B-Instruct")
    temperature = args.temp if args.temp is not None else cfg.get("temperature", 0.7)
    max_new_tokens = cfg.get("max_new_tokens", 256)
    target_dim = args.target_dim or cfg.get("target_dim", 128)
    injection_scale = cfg.get("injection_scale", 0.01)

    log.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
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
        injection_scale=cfg.get("injection_scale", 0.01),
        blend_mode=cfg.get("blend_mode", "append"),
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

    # Continuity mode
    if args.continuity:
        prompt = args.prompt or cfg.get("continuity_prompt", "Tell me about yourself")

        runner = ContinuityRunner(
            model=model,
            tokenizer=tokenizer,
            tracker=tracker,
            injector=injector,
            device=str(model.device),
        )

        def _generate(prompt, n_tokens, temperature):
            return generate_with_proprioception(
                model=model,
                tokenizer=tokenizer,
                tracker=tracker,
                injector=injector,
                prompt=prompt,
                max_new_tokens=n_tokens,
                temperature=temperature,
                proprio_inject=True,
                device=str(model.device),
                damping=cfg.get("injection_damping", 0.7),
                inject_mode=cfg.get("inject_mode", "one"),
                random_inject=args.random_inject,
                stream=False,
            )

        runner.run(
            generate_fn=_generate,
            prompt=prompt,
            initial_tokens=cfg.get("max_new_tokens", 256),
            temperature=args.temp if args.temp is not None else cfg.get("temperature", 0.7),
            n_runs=args.runs,
            dream_weight=cfg.get("dream_weight", 0.003),
            damping=cfg.get("injection_damping", 0.7),
            inject_mode=cfg.get("inject_mode", "one"),
        )
        return

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
                inject_mode=cfg.get("inject_mode", "continuous"),
                random_inject=args.random_inject,
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
