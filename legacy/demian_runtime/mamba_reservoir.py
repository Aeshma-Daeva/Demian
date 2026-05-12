"""Mamba reservoir: pure self-reference without attention.

Surgery:
    1. lm_head disconnected — no language collapse
    2. Layer 0 input replaced with fed-back residual (same as transformer)
    3. SSM cache optionally persisted between steps (Mamba's recurrent memory)

Unlike the transformer reservoir:
    - No RoPE to disable (Mamba uses no explicit positional encoding)
    - No attention/FFN alternation (single Mamba block per layer)
    - SSM state h accumulates history when persist_ssm_state=True

Expected dynamics vs transformer:
    - Transformer:           period-2 oscillation (attention/FFN eigenvalue alternation)
    - Mamba + SSM cache:     convergence or slow oscillation (A decay pulls h toward 0)
    - Mamba - SSM cache:     stateless baseline (each step memoryless)

The comparison question: does the period-2 limit cycle (ac2=0.963) require
attention structure, or does any self-referential loop produce it?
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
# Shared helpers — used by both run_mamba_reservoir and competition.py
# ---------------------------------------------------------------------------

def _prepare_injection(
    force_driver,
    driver_state: dict,
    current_residual: torch.Tensor,
    embed_norm: float,
    device: str,
    dtype=torch.bfloat16,
) -> torch.Tensor:
    """Compute normalized injection tensor from driver or raw residual."""
    if force_driver is not None and driver_state:
        _inj, _scale, (_ds, _df) = force_driver.step(driver_state, current_residual)
        _inj_norm = float(torch.norm(_inj.float()))
        if _inj_norm > 1e-10:
            return (_inj.float() * (embed_norm / _inj_norm)).to(dtype)
        return current_residual
    else:
        _cr = current_residual.float()
        _cr_norm = float(torch.norm(_cr))
        if _cr_norm > 1e-10:
            return (_cr * (embed_norm / _cr_norm)).to(dtype)
        return current_residual


def _register_injection_hook(layer0, cr: torch.Tensor):
    """Register layer0 pre-hook that replaces hidden_states with cr.

    cr is captured by value as a parameter — safe across sequential agent runs.
    Returns the hook handle; caller must call handle.remove() after forward pass.
    """
    def hook_fn(module, args):
        if isinstance(args, tuple) and len(args) > 0:
            hidden = args[0]
            shape = list(hidden.shape)
            shape[-1] = cr.shape[-1]
            new_hidden = torch.zeros(shape, device=hidden.device, dtype=hidden.dtype)
            new_hidden[:, -1, :] = cr.to(hidden.device, dtype=hidden.dtype)
            return (new_hidden,) + args[1:]
        return args
    return layer0.register_forward_pre_hook(hook_fn, with_kwargs=False)


def _register_gate_hooks(all_layers):
    """Register dt_proj forward hooks on every layer for Δ/E-I metrics.

    Returns a fresh (captures_list, handles_list) each call — no cross-agent
    contamination when called sequentially per agent.
    Caller must remove all handles after forward pass.
    """
    gate_captures = []
    gate_handles = []
    for _layer in all_layers:
        _mixer = getattr(_layer, "mixer", None)
        if _mixer is not None:
            _dt_proj = getattr(_mixer, "dt_proj", None)
            if _dt_proj is not None:
                def _ghook(mod, inp, out_g, _c=gate_captures):
                    _c.append(out_g.detach().float().cpu())
                gate_handles.append(_dt_proj.register_forward_hook(_ghook))
    return gate_captures, gate_handles


def _compute_gate_metrics(all_layers, gate_captures):
    """Compute (gate_mean, gate_variance, spectral_radius_ssm) from dt_proj captures.

    Returns normalized [0,1] values.
    Returns (0.5, 0.5, 0.5) if no captures available.
    """
    gate_mean_val = 0.5
    gate_var_val  = 0.5
    spec_rad_val  = 0.5
    if gate_captures:
        _delta_means = []
        _abar_maxes  = []
        for _lyr, _raw_dt in zip(all_layers, gate_captures):
            _mixer = getattr(_lyr, "mixer", None)
            if _mixer is None:
                continue
            _dt_bias = getattr(_mixer, "dt_bias", None)
            _raw = _raw_dt[0, 0]  # [d_inner]
            if _dt_bias is not None:
                _delta = torch.nn.functional.softplus(_raw + _dt_bias.detach().float().cpu())
            else:
                _delta = torch.nn.functional.softplus(_raw)
            _delta_means.append(float(_delta.mean()))
            _A_log = getattr(_mixer, "A_log", None)
            if _A_log is not None:
                _A = -torch.exp(_A_log.detach().float().cpu())
                _A_bar = torch.exp(_delta.unsqueeze(-1) * _A)
                _abar_maxes.append(float(_A_bar.max()))
        if _delta_means:
            _gm = float(np.mean(_delta_means))
            gate_mean_val = float(np.clip(_gm / 5.0, 0, 1))
            gate_var_val  = float(np.clip(float(np.std(_delta_means)) / 2.5, 0, 1))
        if _abar_maxes:
            spec_rad_val  = float(np.clip(float(np.mean(_abar_maxes)), 0, 1))
    return gate_mean_val, gate_var_val, spec_rad_val


def _build_step_metrics(
    step_idx: int,
    new_residual: torch.Tensor,
    prev_residual,
    prev_velocity,
    d_model: int,
    hidden_states_all,
    gate_mean: float,
    gate_var: float,
    spec_rad: float,
):
    """Compute all step metrics from forward-pass outputs.

    Returns (step_dict, new_velocity_tensor).
    new_velocity is None if prev_residual is None (first step).
    """
    norms, deltas, lwr, lag = _layer_geometry(hidden_states_all, d_model)

    resid_1d = new_residual.view(-1)
    resid_np  = resid_1d.float().cpu().numpy()
    sc, sconc = _fft_spectrum(resid_np)
    energy    = float(torch.norm(resid_1d)) / (d_model ** 0.5)

    if prev_residual is not None:
        dv    = resid_1d.float() - prev_residual.float()
        rdelta = float(torch.norm(dv)) / (d_model ** 0.5)
        tcoh   = float(torch.nn.functional.cosine_similarity(
            resid_1d.float(), prev_residual.float(), dim=0
        ))
    else:
        rdelta = 0.0
        tcoh   = 1.0

    new_velocity = None
    vel_align    = 1.0
    if prev_residual is not None:
        this_vel = new_residual.float() - prev_residual.float()
        if prev_velocity is not None:
            vn = prev_velocity.norm() * this_vel.norm()
            vel_align = (float(torch.dot(prev_velocity, this_vel) / vn)
                         if vn > 1e-10 else 0.0)
        new_velocity = this_vel

    peak  = float(resid_1d.float().abs().max())
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
        gate_mean=gate_mean,
        gate_variance=gate_var,
        spectral_radius_ssm=spec_rad,
    )
    return step, new_velocity


def _cache_to_device(cache_params, device):
    """Recursively move cache_params tensors to device.

    Used for optional cache offload (--offload-cache) on <12GB GPUs.
    Handles None, objects with .to(), dicts, lists, tuples.
    """
    if cache_params is None:
        return None
    if hasattr(cache_params, "to"):
        return cache_params.to(device)
    if isinstance(cache_params, dict):
        return {k: _cache_to_device(v, device) for k, v in cache_params.items()}
    if isinstance(cache_params, (list, tuple)):
        moved = [_cache_to_device(v, device) for v in cache_params]
        return type(cache_params)(moved)
    return cache_params


# ---------------------------------------------------------------------------
# Surgery
# ---------------------------------------------------------------------------

def _disable_lm_head(model):
    """Disconnect lm_head so no token output is ever computed."""
    lm_head = model.lm_head
    original_forward = lm_head.forward
    lm_head.forward = lambda *a, **kw: None
    return lm_head, original_forward


def _restore_lm_head(lm_head, original_forward):
    lm_head.forward = original_forward


def _get_layer0(model):
    """Return the first block regardless of backbone attribute name."""
    # HuggingFace Mamba uses model.backbone.layers
    # Some checkpoints use model.model.layers (like transformers >=4.40 Mamba2)
    for attr in ("backbone", "model"):
        backbone = getattr(model, attr, None)
        if backbone is not None and hasattr(backbone, "layers"):
            return backbone.layers[0], backbone
    raise AttributeError(
        "Cannot locate layer stack. Expected model.backbone.layers or model.model.layers."
    )


def _get_all_layers(model):
    for attr in ("backbone", "model"):
        backbone = getattr(model, attr, None)
        if backbone is not None and hasattr(backbone, "layers"):
            return backbone.layers
    raise AttributeError("Cannot locate layer stack.")


# ---------------------------------------------------------------------------
# Metrics (identical to transformer reservoir for direct comparison)
# ---------------------------------------------------------------------------

def _fft_spectrum(residual: np.ndarray) -> tuple:
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


def _layer_geometry(hidden_states_all, d_model: int):
    """Per-layer residual norms and work ratio — same as transformer reservoir."""
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
        ea, la = ea[:ml], la[:ml]
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
    cp_path = Path(data_dir) / "mamba_checkpoint.pt"
    torch.save({
        "residual": residual.cpu(),
        "step": step,
        "trajectory": trajectory,
    }, cp_path)


def _load_checkpoint(data_dir):
    cp_path = Path(data_dir) / "mamba_checkpoint.pt"
    if not cp_path.exists():
        return None
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    log.info("Checkpoint found at step %d — resuming", cp["step"])
    return cp["residual"], cp["step"], cp["trajectory"]


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_mamba_reservoir(
    model,
    d_model: int,
    device: str,
    max_steps: int = 1000,
    dump_interval: int = 100,
    data_dir: str = "data/mamba_reservoir",
    seed_residual=None,
    persist_ssm_state: bool = True,
    force_driver=None,
    hebbian_adapters=None,
    adapter_checkpoint: str = "data/mamba_reservoir/adapters.pt",
):
    """Execute the Mamba reservoir loop.

    Each step:
        1. Hook layer 0 to replace hidden_states with injection
        2. Forward pass through all Mamba blocks
        3. Capture new_residual from last layer
        4. Apply Hebbian weight update (if adapters active)
        5. Optionally carry SSM cache forward (persist_ssm_state=True)
        6. Feed back

    Args:
        persist_ssm_state: If True, SSM state h accumulates across steps.
        force_driver: optional CriticalityDriver. Structured force injection.
        hebbian_adapters: optional dict of HebbianAdapter. Non-frozen weights.
            BCM updates driven by force driver signals each step.
        adapter_checkpoint: path to save adapter weights periodically.
    """
    layer0, backbone = _get_layer0(model)
    all_layers = _get_all_layers(model)

    # Note: Mamba's forward calls .float() on lm_head output internally,
    # so we cannot return None from a disabled lm_head. We let it run
    # and simply ignore out.logits — overhead is negligible for 768-dim.
    log.info("SSM state persistence: %s", persist_ssm_state)

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Checkpoint resume
    # Always compute embed_norm from a clean warmup (needed for injection scaling)
    dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        _emb_out = model(dummy, output_hidden_states=True, use_cache=False)
    embed_norm = float(torch.norm(_emb_out.hidden_states[0][0, -1, :].float()))
    log.info("Embedding norm (injection scale target): %.4f", embed_norm)
    del _emb_out

    checkpoint = _load_checkpoint(data_dir)
    if checkpoint is not None and seed_residual is None:
        ck_residual, ck_step, trajectory = checkpoint
        current_residual = ck_residual.to(device)
        prev_residual = None
        prev_velocity = None
        start_step = ck_step + 1
    else:
        trajectory = []
        prev_residual = None
        prev_velocity = None
        if seed_residual is not None:
            current_residual = seed_residual.to(device)
            log.info("Using seed residual, norm=%.4f", float(torch.norm(current_residual)))
        else:
            with torch.no_grad():
                warmup = model(
                    dummy,
                    output_hidden_states=True,
                    use_cache=False,
                )
            current_residual = warmup.hidden_states[-1][0, -1, :].clone()
            log.info("Warmup done, initial norm=%.4f", float(torch.norm(current_residual)))
        start_step = 1

    # SSM cache state (None = fresh each step when persist_ssm_state=False)
    cache_params = None

    # force_driver produces injection from previous step metrics.
    # First step has no metrics yet — use raw normalized residual.
    _driver_state: dict = {}

    for step_idx in range(start_step, max_steps + 1):
        with torch.no_grad():

            _cr = _prepare_injection(
                force_driver, _driver_state, current_residual, embed_norm, device,
                dtype=current_residual.dtype,
            )
            handle = _register_injection_hook(layer0, _cr)
            _gate_captures, _gate_handles = _register_gate_hooks(all_layers)

            dummy = torch.zeros((1, 1), dtype=torch.long, device=device)

            forward_kwargs = dict(
                output_hidden_states=True,
                use_cache=persist_ssm_state,
                return_dict=True,
            )
            if persist_ssm_state and cache_params is not None:
                forward_kwargs["cache_params"] = cache_params
                forward_kwargs["cache_position"] = torch.tensor(
                    [step_idx - 1], device=device
                )

            out = model(dummy, **forward_kwargs)
            handle.remove()
            for _gh in _gate_handles:
                _gh.remove()

        if persist_ssm_state:
            cache_params = getattr(out, "cache_params", None)

        new_residual = out.hidden_states[-1][0, -1, :].clone()

        gate_mean_val, gate_var_val, spec_rad_val = _compute_gate_metrics(
            all_layers, _gate_captures
        )

        step, new_velocity = _build_step_metrics(
            step_idx, new_residual, prev_residual, prev_velocity,
            d_model, out.hidden_states, gate_mean_val, gate_var_val, spec_rad_val,
        )
        if new_velocity is not None:
            prev_velocity = new_velocity

        # Capture force driver state for next step's injection
        _driver_state = step.copy()
        _driver_state.pop("layer_deltas", None)
        _driver_state.pop("layer_norms", None)

        # Attach driver snapshot to trajectory record
        if force_driver is not None and hasattr(force_driver, "state_dict"):
            step["force_driver"] = force_driver.state_dict()

        # Hebbian weight update — BCM rule gated by force signals
        if hebbian_adapters is not None:
            if force_driver is None:
                force_signals = {}
            elif hasattr(force_driver, "_compat_forces"):
                # MachineDriver: translate geometric observables to Hebbian keys
                force_signals = force_driver._compat_forces()
            else:
                force_signals = force_driver.forces
            for ha in hebbian_adapters.values():
                ha.step(force_signals)
            if step_idx % dump_interval == 0 or step_idx == max_steps:
                from legacy.demian_runtime.hebbian import save_adapters, adapter_summary
                save_adapters(hebbian_adapters, adapter_checkpoint, step_idx)
                step["hebbian"] = adapter_summary(hebbian_adapters)

        trajectory.append(step)

        if step_idx <= 5 or step_idx % dump_interval == 0 or step_idx == max_steps:
            driver_str = ""
            if force_driver is not None and force_driver.sigil:
                driver_str = f" | {force_driver.phase} {force_driver.sigil}"
            hebb_str = ""
            if hebbian_adapters:
                norms = [ha.lora.adapter_norm() for ha in hebbian_adapters.values()]
                hebb_str = f" | W∆={sum(norms)/len(norms):.4f}"
            print(
                "  step {:>5} | E={:.3f} coh={:+.3f} "
                "vel={:+.3f} dlt={:.3f} Lwr={:.3f} "
                "cent={:.4f}{}{}".format(
                    step_idx, energy, tcoh, vel_align, rdelta, lwr, sc,
                    driver_str, hebb_str
                )
            )

        if step_idx % dump_interval == 0:
            chunk = trajectory[-dump_interval:]
            fname = data_path / "mamba_{:06d}.json".format(step_idx)
            fname.write_text(json.dumps(chunk, indent=2))

        if step_idx % dump_interval == 0 or step_idx == max_steps:
            _save_checkpoint(new_residual.cpu(), step_idx, trajectory, data_dir)

        prev_residual = new_residual.clone()
        current_residual = new_residual

    log.info("Run complete")

    final_path = data_path / "mamba_full.json"
    final_path.write_text(json.dumps(trajectory, indent=2))
    log.info("Full trajectory saved: %s (%d steps)", final_path, len(trajectory))

    return trajectory


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demian Mamba Reservoir")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dump-interval", type=int, default=100)
    parser.add_argument("--model", type=str, default=None,
                        help="Mamba model id (overrides config mamba_model_id)")
    parser.add_argument("--seed", type=str, default=None,
                        help="Path to seed residual .pt")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable SSM state persistence (stateless baseline)")
    parser.add_argument("--force-driver", action="store_true",
                        help="Enable CriticalityDriver (anthropocentric force ontology)")
    parser.add_argument("--machine-driver", action="store_true",
                        help="Enable MachineDriver (geometric observable ontology, "
                             "replaces --force-driver)")
    parser.add_argument("--target-novelty", type=float, default=0.15,
                        help="Criticality target rdelta for CriticalityDriver (default 0.15)")
    parser.add_argument("--base-scale", type=float, default=0.01,
                        help="Base injection scale for driver (default 0.01)")
    parser.add_argument("--force-scale", type=float, default=0.005,
                        help="Force tensor injection amplitude (default 0.005)")
    parser.add_argument("--hebbian", action="store_true",
                        help="Enable Hebbian LoRA adapters (non-frozen weights)")
    parser.add_argument("--hebbian-rank", type=int, default=4,
                        help="LoRA rank for Hebbian adapters (default 4)")
    parser.add_argument("--hebbian-eta", type=float, default=4e-5,
                        help="BCM base learning rate (default 4e-5)")
    parser.add_argument("--hebbian-targets", type=str, default="dt_proj",
                        help="Comma-separated module name suffixes to adapt (default: dt_proj)")
    parser.add_argument("--adapter-checkpoint", type=str, default=None,
                        help="Path to save/load adapter weights")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Output directory for trajectory/checkpoints (default: data/mamba_reservoir)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(__file__).resolve().parents[1] / "root_cli" / "config.yaml".read_text())
    model_id = args.model or cfg.get(
        "mamba_model_id", "state-spaces/mamba-2.8b-hf"
    )

    log.info("Loading Mamba model: %s", model_id)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # bfloat16: same memory footprint as float16 but float32's exponent range.
    # Mamba's SSM exp(Δ) overflows float16 when injecting residuals with
    # components outside the embedding's learned distribution. bfloat16 avoids this.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    d_model = model.config.hidden_size
    device = str(next(model.parameters()).device)

    seed_residual = None
    if args.seed:
        p = Path(args.seed)
        if p.exists():
            seed_residual = torch.load(p, weights_only=True)
            if seed_residual.dim() > 1:
                seed_residual = seed_residual.view(-1)[:d_model]

    driver = None
    if args.machine_driver:
        from legacy.demian_runtime.machine_observables import MachineDriver
        driver = MachineDriver(
            d_model=d_model,
            base_scale=args.base_scale,
            force_scale=args.force_scale,
            device=device,
        )
        log.info("MachineDriver active (geometric observables)")
    elif args.force_driver:
        from legacy.demian_runtime.force_driver import CriticalityDriver
        driver = CriticalityDriver(
            d_model=d_model,
            target_novelty=args.target_novelty,
            base_scale=args.base_scale,
            force_scale=args.force_scale,
            device=device,
        )
        log.info("CriticalityDriver active: target_novelty=%.3f", args.target_novelty)

    adapters = None
    data_dir = args.data_dir or "data/mamba_reservoir"
    adapter_ckpt = args.adapter_checkpoint or f"{data_dir}/adapters.pt"
    if args.hebbian:
        from legacy.demian_runtime.hebbian import apply_hebbian_adapters, load_adapters
        targets = [t.strip() for t in args.hebbian_targets.split(",")]
        adapters = apply_hebbian_adapters(
            model,
            rank=args.hebbian_rank,
            eta=args.hebbian_eta,
            target_module_names=targets,
        )
        load_adapters(adapters, adapter_ckpt)
        log.info("Hebbian adapters active: %d modules, rank=%d, eta=%.2e",
                 len(adapters), args.hebbian_rank, args.hebbian_eta)

    print()
    print("=" * 60)
    print("  Demian Mamba Reservoir")
    print("  Surgery:    lm_head off")
    print("  SSM cache:  " + ("disabled (stateless)" if args.no_cache else "enabled (recurrent)"))
    _driver_label = ("MachineDriver λβωSρ" if args.machine_driver else
                     "CriticalityDriver ⚛" if args.force_driver else
                     "none (passive)")
    print("  Driver:     " + _driver_label)
    print("  Weights:    " + (f"Hebbian LoRA rank={args.hebbian_rank} η={args.hebbian_eta:.0e}" if adapters else "frozen"))
    print("  Model:      " + model_id)
    print("  d_model:    " + str(d_model))
    print("  steps:      " + str(args.steps))
    print("=" * 60)
    print()

    run_mamba_reservoir(
        model=model,
        d_model=d_model,
        device=device,
        max_steps=args.steps,
        dump_interval=args.dump_interval,
        data_dir=data_dir,
        seed_residual=seed_residual,
        persist_ssm_state=not args.no_cache,
        force_driver=driver,
        hebbian_adapters=adapters,
        adapter_checkpoint=adapter_ckpt,
    )


if __name__ == "__main__":
    main()
