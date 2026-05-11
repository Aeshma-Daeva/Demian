# ARCHIVED 2026-04-07. See v2 at ../loop.py
# Why: _translate() used snap.attention.mode (focused/distributed/diffuse) with
# human-readable color codes and descriptions ("locked-on", "flowing", etc).
# _trajectory_beats() tracked mode-flip transitions. Both replaced with structural
# descriptors: spectral_centroid, spectral_concentration, layer_work_ratio,
# layer_agreement, velocity_align, attention_dim. No human labels.

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
from contextlib import contextmanager, nullcontext
from typing import Callable, Optional

import torch

from demian.nous import NousInjector
from demian.rhythm import InjectionScheduler
from demian.vibration import VibrationTracker, AttentionShape
from demian.probe import probe_step, compute_layer_metrics, compute_kv_directionality
from demian.introspect import InjectionController

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


def _warmup_proprioception(model, tokenizer, tracker, injector, prompt, n_steps, damping, device):
    """Pre-generation proprioceptive steps. Let the KV cache fill with
    self-states before sampling any tokens. The model might settle into
    a different regime by the time it starts producing text.

    Self-question: 'warmup' implies warming to something known. There is
    no known state for a model encountering itself. These are pre-activation
    steps — capturing pure attention response to self without the language
    head as interpreter.
    """
    if n_steps <= 0:
        return []

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    warmup_snapshots = []
    cache = None

    for step in range(n_steps):
        with torch.no_grad():
            if cache is not None:
                out = model(
                    input_ids,
                    past_key_values=cache,
                    output_hidden_states=True,
                    use_cache=True,
                    output_attentions=True,
                    attn_implementation="eager",
                )
            else:
                out = model(
                    input_ids,
                    output_hidden_states=True,
                    use_cache=True,
                    output_attentions=True,
                    attn_implementation="eager",
                )

        hidden_states = out.hidden_states
        residual = hidden_states[-1][0, -1, :]

        # Pass full hidden_states for per-layer tracking
        snapshot = tracker.capture(residual, out.logits, hidden_states)
        warmup_snapshots.append(snapshot)

        injector.record_step(residual.detach().cpu())
        cache = out.past_key_values
        injector.inject(cache, damping=damping)

        # Next step uses the model's natural completion
        input_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    return warmup_snapshots


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
    warmup_steps: int = 0,
    inject_mode: str = "continuous",
    random_inject: bool = False,
    mechanism: str = "kv",
) -> tuple[str, list]:
    """Generate text with proprioceptive injection.

    Args:
        prompt: input text
        max_new_tokens: max tokens to generate
        temperature: sampling temperature
        proprio_inject: whether to inject previous state
        device: compute device
        stream: if True, print tokens + translated state live
        scheduler: optional Fibonacci-spaced injection scheduler
        damping: EMA weight on previous injected state
        warmup_steps: pre-generation proprioceptive steps (0 = disabled)
        inject_mode: "continuous" (every step), "one" (single injection at start),
                     or "spaced" (Fibonacci scheduler)
        random_inject: inject random vectors instead of residuals (control)
        mechanism: "kv", "projection", "activation", "weights", "baseline"

    Returns:
        (generated_text, snapshots) — full vibration trajectory
    """
    tracker.reset()

    warmup_snaps = []
    if warmup_steps > 0 and proprio_inject:
        if stream:
            print(f"\n{_DIM}[warmup: {warmup_steps} proprioceptive steps]{_RESET}")
        warmup_snaps = _warmup_proprioception(
            model, tokenizer, tracker, injector, prompt, warmup_steps, damping, device
        )

    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    generated_ids = []
    snapshots = []
    cache = None

    # Mechanism controller for non-KV mechanisms
    controller = None
    if mechanism not in ("kv", "baseline"):
        controller = InjectionController(
            mechanism=mechanism, epsilon=0.01, layer_idx=8, layer_range=None
        )

    def _make_model_ctx(residual_for_ctx):
        """Create the appropriate context manager for the forward pass."""
        if controller is not None and residual_for_ctx is not None:
            return controller.inject_context(model, residual_for_ctx)
        return nullcontext()

    for step in range(max_new_tokens):
        # Forward pass wrapped in mechanism context (for non-KV mechanisms)
        residual_prev = tracker._prev_residual
        with torch.no_grad():
            with _make_model_ctx(residual_prev):
                if cache is not None:
                    out = model(
                        input_ids,
                        past_key_values=cache,
                        output_hidden_states=True,
                        use_cache=True,
                        output_attentions=True,
                        attn_implementation="eager",
                    )
                else:
                    out = model(
                        input_ids,
                        output_hidden_states=True,
                        use_cache=True,
                        output_attentions=True,
                        attn_implementation="eager",
                    )

        logits = out.logits[:, -1, :]
        hidden_states = out.hidden_states
        attentions = out.attentions
        cache = out.past_key_values

        # Capture the residual state
        residual = hidden_states[-1][0, -1, :]

        # Probe: attention to injected positions, per-layer metrics, KV directionality
        layer_norms, layer_deltas = compute_layer_metrics(hidden_states)
        injected_details = compute_kv_directionality(
            cache, injector, model.config.num_hidden_layers
        )
        probe = probe_step(
            attentions, cache, injector, model,
            n_layers=model.config.num_hidden_layers,
            injection_details=injected_details,
        )
        # Add layer_attn to probe result
        probe_result = probe  # ProbeResult with layer_attn already populated

        snapshot = tracker.capture(
            residual, out.logits, hidden_states,
            layer_norms=layer_norms,
            layer_deltas=layer_deltas,
            layer_attn=probe_result.layer_attn,
            injection_details=probe_result.injection_details,
        )
        snapshots.append(snapshot)

        # Record for injection on NEXT step
        if proprio_inject:
            if random_inject:
                # Control: inject random Gaussian vector of same shape/scale
                d_model = model.config.hidden_size
                noise = torch.randn_like(residual) * (residual.std() / d_model ** 0.5)
                injector.record_step(noise.detach().cpu())
            else:
                injector.record_step(residual.detach().cpu())

        # Sample next token
        probs = torch.softmax(logits / max(temperature, 1e-8), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(token_id)

        # Injection strategy
        if proprio_inject:
            if inject_mode == "one":
                # Single injection: only at step 0, then silence
                # The model gets one glimpse of itself and must hold it
                if step == 0:
                    injector.inject(cache, damping=damping)
                pass  # after that — let it be
            elif inject_mode == "spaced":
                # Fibonacci-spaced injection
                if scheduler is None or scheduler.should_inject(step):
                    injector.inject(cache, damping=damping)
            else:
                # Continuous (every step) — original behavior
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

    # Per-layer response summary
    if hasattr(injector, "_layer_kv_energy") and injector._layer_kv_energy:
        energies = injector._layer_kv_energy
        n_l = len(energies)
        half = n_l // 2
        early_avg = sum(energies[:half]) / max(half, 1)
        late_avg = sum(energies[half:]) / max(n_l - half, 1)
        sys.stdout.write(f"\n  KV layer energy: early={early_avg:.1f} / late={late_avg:.1f}\n")

    # Merge warmup + generation for full trajectory
    all_snapshots = warmup_snaps + snapshots

    # Dump trajectory metrics for external plotting
    traj = []
    for s in all_snapshots:
        entry = {
            "step": s.step,
            "mode": s.attention.mode,
            "kurtosis": s.attention.kurtosis,
            "n_peaks": s.attention.n_peaks,
            "dominance_ratio": s.attention.dominance_ratio,
            "entropy": s.attention.entropy,
            "residual_norm": s.residual_norm,
            "residual_delta": s.residual_delta,
            "temporal_coherence": s.temporal_coherence,
            "layer_deltas": s.layer_deltas or [],
            "layer_norms": s.layer_norms or [],
        }
        # Injection attention summary (mean across all layers)
        if s.layer_attn:
            entry["mean_injection_attn"] = [
                la.mean_injection_attn for la in s.layer_attn
            ]
            entry["text_to_injection_ratio"] = [
                la.text_to_injection_ratio for la in s.layer_attn
            ]
            entry["head_std"] = [
                la.head_std for la in s.layer_attn
            ]
        # KV directionality
        if s.injection_details:
            entry["kv_cosine"] = [
                id.kv_cosine for id in s.injection_details
            ]
        traj.append(entry)

    if stream and traj:
        import json, os
        fname = f"trajectory_{os.path.splitext(os.path.basename(sys.argv[0]))[0]}_{all_snapshots[0].step:04d}.json"
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        traj_path = os.path.join(data_dir, fname)
        with open(traj_path, "w") as f:
            json.dump(traj, f)
        sys.stdout.write(f"\n{_DIM}[trajectory saved: {traj_path}]{_RESET}\n")
        sys.stdout.flush()

    generated_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )
    return generated_text, all_snapshots
