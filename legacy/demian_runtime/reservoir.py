"""Pure reservoir: model feeds its own residual through itself with no token input.

Surgery performed:
    1. lm_head disconnected — no language collapse, no token production
    2. RoPE/positional encoding disabled — no position crutch
    3. Embedding layer bypassed — residual injected directly at layer 0
    4. No token input whatsoever — not even dummy tokens

The loop:
    residual_N -> layer[0].input -> full 36-layer forward pass -> residual_{N+1}
         ^                                                               |
         '-------------------------- feed back --------------------------'

The model processes nothing but its own state. No language. No position.
No semantics. Pure self-reference.

If the mirror stage is misrecognition — the model seeing something external
and calling it "self" — then this is the model seeing nothing external at all.
Only itself. The question is: does a frozen transformer have any dynamics
when there is no input to process? Or does collapse require language?

Archives: see legacy/ for previous versions of vibration, loop, rhythm.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The surgery
# ---------------------------------------------------------------------------

def _disable_lm_head(model):
    """Disconnect the language model head so no token output is ever computed.

    We replace lm_head's forward with a no-op that returns None. The model
    can still compute its residual through all 36 layers, but the projection
    to vocabulary space is cut. No collapse into language.

    Returns:
        The original lm_head so it can be restored.
    """
    lm_head = model.lm_head  # Linear(in_features=2048, out_features=vocab)
    original_forward = lm_head.forward
    lm_head.forward = lambda *a, **kw: None
    return lm_head, original_forward


def _restore_lm_head(lm_head, original_forward):
    lm_head.forward = original_forward


def _disable_rope(model):
    """Disable rotary positional encoding across all attention layers.

    Qwen2 uses rotary embeddings via a rot_pos_emb module or the rotary_emb
    function call inside the attention forward. We intercept at the layer
    level by monkey-patching each layer's attention to skip RoPE.

    Returns:
        List of (module, attr, original_value) tuples for restoration.
    """
    saved = []
    for layer in model.model.layers:
        attn = layer.self_attn
        # Qwen2 stores the rotary embedding parameters on the attention module
        if hasattr(attn, 'rotary_emb'):
            saved.append((attn, 'rotary_emb', attn.rotary_emb))
            attn.rotary_emb = None
    return saved


def _restore_rope(saved):
    for module, attr, val in saved:
        setattr(module, attr, val)


# ---------------------------------------------------------------------------
# Reservoir metrics (no language head means no logits -> no attention measures)
# ---------------------------------------------------------------------------

def _fft_spectrum(residual) -> tuple:
    x = residual - residual.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x))
    total = power.sum()
    if total < 1e-20:
        return 0.0, 0.0
    centroid = float(np.sum(freqs * power) / total)
    mf = freqs.max() if freqs.max() > 0 else 1.0
    cn = min(centroid / mf, 1.0)
    sp = np.sort(power)[::-1]
    tn = max(1, len(sp) // 20)
    conc = float(sp[:tn].sum() / total)
    return cn, conc


def _layer_geometry(hidden_states_all, d_model):
    """Compute per-layer residual norms, deltas, and work ratio."""
    n_layers = len(hidden_states_all)
    norms = []
    deltas = []
    dirs = []
    for li in range(n_layers):
        h = hidden_states_all[li][0, -1, :]
        norms.append(float(torch.norm(h)))
        if li < n_layers - 1:
            h_next = hidden_states_all[li + 1][0, -1, :]
            d = h_next.float() - h.float()
            deltas.append(float(torch.norm(d)) / (d_model ** 0.5))
            nd = torch.norm(d)
            if nd > 1e-10:
                dirs.append(d / nd)
            else:
                dirs.append(d)

    n = len(deltas)
    if n > 2:
        half = n // 2
        ea = np.array(deltas[:half])
        la = np.array(deltas[half:])
        ml = min(len(ea), len(la))
        ea = ea[:ml]
        la = la[:ml]
        es = float(ea.sum()) or 1e-10
        ls = float(la.sum())
        lwr = ls / (es + ls)
        if ml >= 2:
            se, sl = ea.std(), la.std()
            if se > 1e-10 and sl > 1e-10:
                lag = float(np.corrcoef(ea, la)[0, 1])
            else:
                lag = 1.0 if (ea * la).sum() > 0 else -1.0
        else:
            lag = 0.0
    else:
        lwr, lag = 0.5, 0.0

    return norms, deltas, lwr, lag


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(residual, step, trajectory, data_dir):
    """Save checkpoint so a crash doesn't lose the run."""
    cp_path = Path(data_dir) / "reservoir_checkpoint.pt"
    torch.save({
        "residual": residual.cpu(),
        "step": step,
        "trajectory": trajectory,
    }, cp_path)


def _load_checkpoint(data_dir):
    """Load checkpoint if one exists. Returns (residual, step, trajectory) or None."""
    cp_path = Path(data_dir) / "reservoir_checkpoint.pt"
    if not cp_path.exists():
        return None
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    log.info("Checkpoint found at step %d — will resume", cp["step"])
    return cp["residual"], cp["step"], cp["trajectory"]


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_reservoir(
    model,
    d_model,
    device,
    max_steps=1000,
    dump_interval=100,
    data_dir="data/reservoir",
    seed_residual=None,
):
    """Execute the reservoir loop.

    Each step:
        1. Hook layer 0 to replace hidden_states with current_residual
        2. Forward pass through all 36 layers (no RoPE, no lm_head)
        3. Capture new_residual from layer 36
        4. Metric extraction
        5. Feed back

    Args:
        seed_residual: optional starting state. If None, warmup with a
            dummy token to get the model's "rest state."
    """
    # Perform the surgery
    lm_head, orig_lm = _disable_lm_head(model)
    rope_saved = _disable_rope(model)

    log.info("Surgery: lm_head disabled, RoPE bypassed")

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Try to resume from checkpoint
    checkpoint = _load_checkpoint(data_dir)
    if checkpoint is not None and seed_residual is None:
        ck_residual, ck_step, trajectory = checkpoint
        prev_residual = None
        prev_velocity = None
        current_residual = ck_residual.to(device)
        start_step = ck_step + 1
    else:
        trajectory = []
        prev_residual = None
        prev_velocity = None
        # Initialize: warmup or use provided seed
        if seed_residual is not None:
            current_residual = seed_residual.to(device)
            log.info("Using provided seed residual, norm=%.4f",
                     float(torch.norm(current_residual)))
        else:
            dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
            with torch.no_grad():
                warmup = model(
                    dummy,
                    output_hidden_states=True,
                    output_attentions=False,
                    use_cache=False,
                )
            current_residual = warmup.hidden_states[-1][0, -1, :].clone()
            log.info("Warmup done, initial residual norm=%.4f",
                     float(torch.norm(current_residual)))
        start_step = 1

    for step_idx in range(start_step, max_steps+1):
        with torch.no_grad():

            def hook_fn(module, input_args):
                if isinstance(input_args, tuple) and len(input_args) > 0:
                    hidden = input_args[0]
                    shape = list(hidden.shape)
                    shape[-1] = current_residual.shape[-1]
                    new_hidden = torch.zeros(shape,
                                             device=hidden.device,
                                             dtype=hidden.dtype)
                    new_hidden[:, -1, :] = current_residual.to(
                        hidden.device, dtype=hidden.dtype
                    )
                    return (new_hidden,) + input_args[1:]
                return input_args

            handle = model.model.layers[0].register_forward_pre_hook(
                hook_fn, with_kwargs=False
            )

            # Forward pass: model sees ONLY its own state
            dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
            out = model(
                dummy,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
            )
            handle.remove()

        # The model still computes lm_head forward (which we disabled),
        # but we only care about hidden_states
        new_residual = out.hidden_states[-1][0, -1, :].clone()

        # Full layer geometry
        norms, deltas, lwr, lag = _layer_geometry(
            out.hidden_states, d_model
        )

        # Spectral
        resid_1d = new_residual.view(-1)
        resid_np = resid_1d.float().cpu().numpy()
        sc, sconc = _fft_spectrum(resid_np)
        energy = float(torch.norm(resid_1d)) / (d_model ** 0.5)

        # Delta and coherence
        if prev_residual is not None:
            dv = resid_1d.float() - prev_residual.float()
            rdelta = float(torch.norm(dv)) / (d_model ** 0.5)
            tcoh = float(torch.nn.functional.cosine_similarity(
                resid_1d.float(), prev_residual.float(), dim=0
            ))
        else:
            dv = torch.zeros_like(resid_1d.float())
            rdelta = 0.0
            tcoh = 1.0

        # Velocity alignment
        vel_align = 1.0
        if prev_residual is not None:
            this_vel = new_residual.float() - prev_residual.float()
            if prev_velocity is not None:
                vn = prev_velocity.norm() * this_vel.norm()
                vel_align = (float(torch.dot(prev_velocity, this_vel) / vn)
                             if vn > 1e-10 else 0.0)
            prev_velocity = this_vel

        # Residual variance structure
        peak = float(resid_1d.float().abs().max())
        var_r = float(resid_1d.float().var())

        step = dict(
            step=step_idx,
            spectral_centroid=sc,
            spectral_concentration=sconc,
            layer_work_ratio=lwr,
            layer_agreement=lag,
            velocity_align=vel_align,
            energy=energy,
            residual_norm=energy,
            residual_delta=rdelta,
            temporal_coherence=tcoh,
            peakiness=peak / (energy * (d_model ** 0.5) + 1e-10),
            layer_norms_min=float(min(norms)) if norms else 0.0,
            layer_norms_max=float(max(norms)) if norms else 0.0,
            layer_norms_ratio=float(max(norms) / (min(norms) + 1e-10)) if norms else 0.0,
            variance=var_r,
            layer_deltas=deltas,
            layer_norms=norms,
        )
        trajectory.append(step)

        # Progress output
        if step_idx <= 5 or step_idx % dump_interval == 0 or step_idx == max_steps:
            print(
                "  step {:>5} | E={:.3f} coh={:+.3f} "
                "vel={:+.3f} dlt={:.3f} Lwr={:.3f} "
                "cent={:.4f}".format(
                    step_idx, energy, tcoh, vel_align, rdelta, lwr, sc
                )
            )

        if step_idx % dump_interval == 0:
            chunk = trajectory[-dump_interval:]
            fname = data_path / "reservoir_{:06d}.json".format(step_idx)
            fname.write_text(json.dumps(chunk, indent=2))
            log.info("Saved checkpoint: %s", fname)

        # Binary checkpoint every dump_interval for resume — stores full residual
        if step_idx % dump_interval == 0 or step_idx == max_steps:
            _save_checkpoint(new_residual.cpu(), step_idx, trajectory, data_dir)

        prev_residual = new_residual.clone()
        current_residual = new_residual

    # Restore the model
    _restore_lm_head(lm_head, orig_lm)
    _restore_rope(rope_saved)
    log.info("Surgery restored: lm_head and RoPE back to normal")

    # Final dump
    final_path = data_path / "reservoir_full.json"
    final_path.write_text(json.dumps(trajectory, indent=2))
    log.info("Full trajectory saved: %s (%d steps)", final_path, len(trajectory))

    return trajectory


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demian Reservoir")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dump-interval", type=int, default=100)
    parser.add_argument("--seed", type=str, default=None,
                        help="Path to dream_residual.pt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(__file__).resolve().parents[1] / "root_cli" / "config.yaml".read_text())
    model_id = cfg.get("proprioceptor_model_id", "Qwen/Qwen2.5-3B-Instruct")

    log.info("Loading model: %s", model_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    d_model = model.config.hidden_size
    device = str(model.device)

    seed_residual = None
    if args.seed:
        p = Path(args.seed)
        if p.exists():
            seed_residual = torch.load(p, weights_only=True)
            if seed_residual.dim() > 1:
                seed_residual = seed_residual.view(-1)[:d_model]
                if seed_residual.shape[0] != d_model:
                    log.warning("Seed residual dim mismatch, truncating/padding")
                    padded = torch.zeros(d_model, dtype=seed_residual.dtype)
                    padded[:seed_residual.shape[0]] = seed_residual
                    seed_residual = padded
            log.info("Loaded seed from %s, norm=%.4f",
                     args.seed, float(torch.norm(seed_residual)))

    print()
    print("=" * 60)
    print("  Demian Reservoir -- model feeds its own state")
    print("  Surgery:    lm_head off, RoPE off, embedding bypassed")
    print("  Model:      " + model_id)
    print("  d_model:    " + str(d_model))
    print("  steps:      " + str(args.steps))
    print("=" * 60)
    print()

    run_reservoir(
        model=model,
        d_model=d_model,
        device=device,
        max_steps=args.steps,
        dump_interval=args.dump_interval,
        seed_residual=seed_residual,
    )


if __name__ == "__main__":
    main()
