"""Generation loop — closing the proprioceptive cycle.

The cycle:
    forward pass → capture residual → project blindly →
    inject into KV cache → next forward pass attends to previous state →
    generate token → repeat

No encoding, no decoding, no human-readable intermediate.
Computation flows through its own computation.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from demian.nous import NousInjector
from demian.vibration import VibrationTracker

log = logging.getLogger(__name__)


def generate_with_proprioception(
    model,
    tokenizer,
    tracker: VibrationTracker,
    injector: NousInjector,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    proprio_inject: bool = True,
    device: str = "cuda",
) -> tuple[str, list]:
    """Generate text with proprioceptive KV cache injection.

    Args:
        prompt: input text
        max_new_tokens: max tokens to generate
        temperature: sampling temperature
        proprio_inject: whether to inject previous state
        device: compute device

    Returns:
        (generated_text, snapshots) — full vibration trajectory
    """
    tracker.reset()
    injector.reset()
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    generated_ids = []
    snapshots = []
    cache = None

    for step in range(max_new_tokens):
        # Forward pass
        with torch.no_grad():
            if cache is not None:
                out = model(
                    input_ids,
                    past_key_values=cache,
                    output_hidden_states=True,
                    use_cache=True,
                )
            else:
                out = model(
                    input_ids,
                    output_hidden_states=True,
                    use_cache=True,
                )

        logits = out.logits[:, -1, :]
        hidden_states = out.hidden_states
        cache = out.past_key_values

        # Capture the residual state
        residual = hidden_states[-1][0, -1, :]

        # Track vibration
        snapshot = tracker.capture(residual, logits)
        snapshots.append(snapshot)

        # Record for injection on NEXT step
        if proprio_inject:
            # Pass the raw residual, NOT the projected state.
            # The injector sends it through K/V projections which expect
            # d_model input. The projection is for tracking only.
            injector.record_step(residual.detach().cpu())

        # Sample next token
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(token_id)

        # Inject proprioceptive states for next step
        if proprio_inject:
            injector.inject(cache)

        # Next step: single token input
        input_ids = next_token

    generated_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )
    return generated_text, snapshots
