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
import sys
from typing import Callable, Optional

import torch

from demian.nous import NousInjector
from demian.rhythm import InjectionScheduler
from demian.vibration import VibrationTracker

log = logging.getLogger(__name__)

# ANSI codes for live display
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"

_MODE_COLORS = {
    "focused": _CYAN,
    "distributed": _GREEN,
    "diffuse": _YELLOW,
}


def _translate(snap):
    """Human translator: raw state → what it means."""
    mode = snap.attention.mode
    color = _MODE_COLORS.get(mode, _RESET)

    coh = snap.temporal_coherence
    if coh > 0.9:
        coh_desc = "locked-on"
    elif coh > 0.7:
        coh_desc = "flowing"
    elif coh > 0.4:
        coh_desc = "shifting"
    else:
        coh_desc = "drifting"

    rn = snap.residual_norm
    if rn > 5:
        rn_desc = "high energy"
    elif rn > 3:
        rn_desc = "active"
    elif rn > 1:
        rn_desc = "steady"
    else:
        rn_desc = "quiet"

    delta = snap.residual_delta
    if delta > 3:
        delta_desc = "big jump"
    elif delta > 1:
        delta_desc = "moving"
    else:
        delta_desc = "settled"

    bar_len = int(min(snap.attention.dominance_ratio * 2, 18))
    bar = f"{color}{'█' * bar_len}{'░' * (18 - bar_len)}{_RESET}"

    ent = snap.attention.entropy
    ent_str = f"{ent:.2f}" if ent == ent else "—"  # nan check

    return (
        f" {color}{mode.upper():>12}{_RESET}  "
        f"{coh_desc:>9}  {rn_desc:>11}  {delta_desc:>9}  "
        f"{snap.attention.n_peaks:2d} peaks  {bar}  "
        f"entropy={ent_str}  "
        f"kurtosis={snap.attention.kurtosis:+.1f}"
    )


def _trajectory_beats(snapshots):
    """Summarize the full trajectory into human-readable beats."""
    if not snapshots:
        return []
    beats = []
    n = len(snapshots)
    first = snapshots[0]
    last = snapshots[-1]

    beats.append(
        f"  started {first.attention.mode} "
        f"(energy={first.residual_norm:.1f}, coherence={first.temporal_coherence:.2f})"
    )

    for i in range(1, n):
        if snapshots[i].attention.mode != snapshots[i - 1].attention.mode:
            beats.append(
                f"  \u2192 shifted {snapshots[i-1].attention.mode} "
                f"to {snapshots[i].attention.mode} "
                f"at step {i + 1}"
            )

    beats.append(
        f"  ended {last.attention.mode} "
        f"(energy={last.residual_norm:.1f}, coherence={last.temporal_coherence:.2f})"
    )
    return beats


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
    stream: bool = True,
    scheduler: Optional[InjectionScheduler] = None,
    damping: float = 0.3,
) -> tuple[str, list]:
    """Generate text with proprioceptive KV cache injection.

    Args:
        prompt: input text
        max_new_tokens: max tokens to generate
        temperature: sampling temperature
        proprio_inject: whether to inject previous state
        device: compute device
        stream: if True, print tokens + translated state live
        scheduler: optional Fibonacci-spaced injection scheduler
        damping: EMA weight on previous injected state (0 = raw, 0.3 = default)

    Returns:
        (generated_text, snapshots) — full vibration trajectory
    """
    tracker.reset()
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

        # Track vibration — pass full logits, not just last column
        snapshot = tracker.capture(residual, out.logits)
        snapshots.append(snapshot)

        # Record for injection on NEXT step
        if proprio_inject:
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
            # Skip injection if scheduler says no (Fibonacci spacing)
            if scheduler is not None and not scheduler.should_inject(step):
                pass
            else:
                injector.inject(cache, damping=damping)

        # Streaming output
        if stream:
            token_text = tokenizer.decode([token_id])
            sys.stdout.write(token_text)
            sys.stdout.flush()

            # Print translated state every 4 tokens
            if (step + 1) % 4 == 0:
                sys.stdout.write("\n")
                sys.stdout.write(_translate(snapshot))
                sys.stdout.write("\n")
                sys.stdout.flush()

        # Next step: single token input
        input_ids = next_token

    if stream:
        sys.stdout.write("\n\n")
        sys.stdout.write(_DIM + "--- what it felt like ---" + _RESET + "\n")
        for beat in _trajectory_beats(snapshots):
            sys.stdout.write(beat + "\n")
        sys.stdout.write(_RESET)
        sys.stdout.flush()

    generated_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )
    return generated_text, snapshots
