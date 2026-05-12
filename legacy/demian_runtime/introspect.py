"""Four injection mechanisms for self-perception testing.

Each mechanism is a context manager that:
1. Applies a modification to the model or its state
2. Yields control for a forward pass
3. Restores the original state

All modifications are transient (torch.no_grad, in-place, then undo).
No training, no gradient, no permanent change.

Mechanism 1: KV cache injection (original) — already in nous.py
Mechanism 2: Projection modulation — modulate K/V projection weights
Mechanism 3: Activation steering — inject directly into hidden state
Mechanism 4: Weight perturbation — temporarily shift all attention weights

The key difference:
  KV injection      → model ATTENDS to self-state
  Projection mod    → model QUERIES differently based on self-state
  Activation steer  → self-state bypasses attention, enters computation directly
  Weight perturbe   → model processes EVERYTHING in a self-modulated regime
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn

from legacy.demian_runtime.nous import NousInjector


# ---------------------------------------------------------------------------
# Mechanism 2: Projection Modulation
# Modulate K/V projection weights by the captured residual direction
# "The K projection itself becomes proprioceptive"
# ---------------------------------------------------------------------------

@contextmanager
def modulate_projections(model, residual: torch.Tensor, epsilon: float = 0.01,
                          layers: Optional[List[int]] = None):
    """Temporarily modulate K and V projection weights.

    k_proj.weight += epsilon * (residual.view(-1, 1) @ random_vector[None, :])

    This injects the residual direction into the K/V space of the projection
    itself. The model doesn't see extra context positions — it computes keys
    differently because the projection matrix has been shifted.

    The modulation is rank-1: outer product of the residual direction and
    a random vector (same shape as k_proj.weight).

    Args:
        model: the model
        residual: captured residual state [d_model]
        epsilon: modulation scale (small: 0.001-0.01)
        layers: which layers to modulate (None = all)
    """
    residual_1d = residual.view(-1)
    d_model = residual_1d.shape[0]
    n_kv_heads = model.config.num_key_value_heads
    head_dim = d_model // model.config.num_attention_heads
    k_out_dim = n_kv_heads * head_dim  # 256 for Qwen2.5-3B

    modified_layers = []

    model_layers = model.model.layers
    layer_range = layers if layers is not None else range(len(model_layers))

    with torch.no_grad():
        for layer_idx in layer_range:
            layer = model_layers[layer_idx]
            attn = layer.self_attn

            k_proj = attn.k_proj
            v_proj = attn.v_proj

            # Store originals
            k_orig = k_proj.weight.data.clone()
            v_orig = v_proj.weight.data.clone()

            # Create rank-1 modulation
            # We want a modulation that has the shape of k_proj.weight: [k_out_dim, d_model]
            # Project residual through the existing k_proj to get a direction in K space:
            k_direction = k_proj(residual_1d.view(1, -1)).view(-1)  # [k_out_dim]
            # Normalize
            k_norm = k_direction.norm()
            if k_norm > 1e-10:
                k_direction = k_direction / k_norm

            # Modulate: weight += epsilon * k_direction * residual_normalized
            # This creates a projection that amplifies the direction
            residual_normed = residual_1d / (residual_1d.norm() + 1e-10)
            outer = torch.ger(k_direction, residual_normed)  # [k_out_dim, d_model]

            k_proj.weight.data.add_(outer, alpha=epsilon)
            v_proj.weight.data.add_(outer * 0.5, alpha=epsilon)

            modified_layers.append((layer_idx, k_orig, v_orig))

    try:
        yield
    finally:
        with torch.no_grad():
            for layer_idx, k_orig, v_orig in modified_layers:
                layer = model_layers[layer_idx]
                layer.self_attn.k_proj.weight.data.copy_(k_orig)
                layer.self_attn.v_proj.weight.data.copy_(v_orig)


# ---------------------------------------------------------------------------
# Mechanism 3: Activation Steering
# Inject residual directly into the hidden state between layers
# bypassing the attention mechanism entirely
# ---------------------------------------------------------------------------

class _ActivationHook:
    """Stateful hook to inject into hidden state between layers."""
    def __init__(self, residual, epsilon):
        self.residual = residual
        self.epsilon = epsilon
        self.original_hook = None

    def __call__(self, module, args, output):
        # Qwen2.5 DecoderLayer returns the hidden state tensor directly,
        # not a tuple. Handle both shapes.
        if isinstance(output, torch.Tensor):
            if output.shape[-1] == self.residual.shape[-1]:
                delta = self.residual.view(1, 1, -1) * self.epsilon
                return output + delta
        elif isinstance(output, tuple) and len(output) > 0:
            if output[0].shape[-1] == self.residual.shape[-1]:
                h = output[0]
                delta = self.residual.view(1, 1, -1) * self.epsilon
                new_h = h + delta
                return (new_h,) + output[1:]
        return output


@contextmanager
def steer_activations(model, residual: torch.Tensor,
                      layer_idx: int = 8, epsilon: float = 0.01):
    """Inject residual directly into the computation at a specific layer.

    This adds the residual to the hidden state AT THE OUTPUT of the given
    layer, before it enters the next layer's attention. The signal doesn't
    go through attention or KV — it enters the computation as if the current
    layer produced it.

    This tests: can self-perception bypass the attention mechanism and still
    influence the computation? The residual becomes part of the stream,
    not part of the context.
    """
    layer = model.model.layers[layer_idx]
    hook = _ActivationHook(residual, epsilon)
    handle = layer.register_forward_hook(hook)

    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Mechanism 4: Weight Perturbation
# Shift ALL attention weights temporarily by the residual direction
# The model processes everything in a self-modulated computational regime
# ---------------------------------------------------------------------------

@contextmanager
def perturb_weights(model, residual: torch.Tensor, epsilon: float = 0.01,
                    layers: Optional[List[int]] = None):
    """Temporarily perturb all linear weights in attention by residual.

    Each linear weight in the attention module gets modified:
        W += epsilon * (W @ residual_residual_normalized)

    This doesn't add new context or bypass any computation. It shifts the
    entire weight space of the attention mechanism by the direction of the
    residual. The model processes everything — text, KV, everything — through
    a different set of weights.

    This is qualitatively different from the other mechanisms:
    - KV: adds context, attention mechanism reads it
    - Projection: changes what keys the model generates
    - Activation: inserts signal mid-computation
    - Weight: changes the MODEL ITSELF temporarily

    The weight perturbation is uniform across all attention parameters:
    q_proj, k_proj, v_proj, o_proj. Each gets shifted by the same residual
    direction scaled by each weight's own norm.
    """
    residual_1d = residual.view(-1)
    residual_normed = residual_1d / (residual_1d.norm() + 1e-10)

    modified = []
    model_layers = model.model.layers
    layer_range = layers if layers is not None else range(len(model_layers))

    with torch.no_grad():
        for layer_idx in layer_range:
            layer = model_layers[layer_idx]
            attn = layer.self_attn

            for param_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                param = getattr(attn, param_name)
                w = param.weight.data
                w_orig = w.clone()

                # Perturbation direction: outer product of weight's own structure
                # with the residual direction. This preserves the weight's learned
                # structure while incorporating the self-state.
                w_row_norms = w.norm(dim=1, keepdim=True)
                direction = residual_1d.view(1, -1) * w_row_norms
                direction = direction / (direction.norm() + 1e-10)

                # Scale by layer's contribution
                w.data.add_(direction, alpha=epsilon * w.norm().item())

                modified.append((param, w_orig))

            # Also perturb LayerNorm weights (scaling, not additive)
            ln = layer.input_layernorm
            ln_orig = ln.weight.data.clone()
            ln.weight.data.add_(residual_1d * epsilon * 0.1)
            modified.append((ln, ln_orig))

            ln2 = layer.post_attention_layernorm
            ln2_orig = ln2.weight.data.clone()
            ln2.weight.data.add_(residual_1d * epsilon * 0.1)
            modified.append((ln2, ln2_orig))

    try:
        yield
    finally:
        with torch.no_grad():
            for param, orig in modified:
                param.weight.data.copy_(orig)


# ---------------------------------------------------------------------------
# Registry and utility
# ---------------------------------------------------------------------------

MECHANISMS = {
    "kv": "KV cache injection (original behavior via NousInjector)",
    "projection": "Modulate K/V projection weights",
    "activation": "Inject residual into hidden state between layers",
    "weights": "Temporarily perturb all attention weights",
}


class InjectionController:
    """Orchestrates injection mechanisms in the generation loop.

    Usage:
        controller = InjectionController("kv")
        for step in range(max_tokens):
            out = model(...)
            record_step(residual)
            controller.inject(model, residual, injector, cache, ...)
    """

    def __init__(self, mechanism: str = "kv", epsilon: float = 0.01,
                 layer_idx: int = 8, layer_range: Optional[List[int]] = None):
        self.mechanism = mechanism
        self.epsilon = epsilon
        self.layer_idx = layer_idx
        self.layer_range = layer_range

    @property
    def needs_injector(self) -> bool:
        return self.mechanism == "kv"

    def inject(self, model, residual, injector=None, cache=None, damping=0.7,
               random_inject=False):
        """Apply the selected injection mechanism.

        For KV: calls injector.inject() (side effect on cache).
        For others: returns a context manager for the forward pass.
        """
        if self.mechanism == "kv":
            if injector is not None and cache is not None:
                if random_inject:
                    d_model = model.config.hidden_size
                    noise = torch.randn_like(residual) * (residual.std() / d_model ** 0.5)
                    injector.record_step(noise.detach().cpu())
                else:
                    injector.record_step(residual.detach().cpu())
                injector.inject(cache, damping=damping)
            return None
        else:
            return None  # Mechanisms 2-4 use context managers, applied in loop

    @contextmanager
    def inject_context(self, model, residual):
        """Context manager for mechanisms 2-4.

        with controller.inject_context(model, residual):
            out = model(input_ids, past_key_values=cache, ...)
        """
        if self.mechanism == "projection":
            with modulate_projections(
                model, residual, epsilon=self.epsilon, layers=self.layer_range
            ):
                yield
        elif self.mechanism == "activation":
            with steer_activations(
                model, residual, layer_idx=self.layer_idx, epsilon=self.epsilon
            ):
                yield
        elif self.mechanism == "weights":
            with perturb_weights(
                model, residual, epsilon=self.epsilon, layers=self.layer_range
            ):
                yield
        else:
            yield  # KV mechanism doesn't use context manager

