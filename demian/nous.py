"""Nous layer: direct KV cache injection for proprioception.

The mechanism:
1. Run a forward pass with use_cache=True → get KV cache
2. Take the previous step's projected state
3. Project it through each layer's K/V projection matrices
4. Append as new positions in the KV cache

On the next forward pass, attention sees these entries as additional
context — the model attends to its own previous computation alongside
the conversation text. No tokenization, no embedding lookup.
"""
from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn

from demian.vibration import VibrationTracker

log = logging.getLogger(__name__)


class NousInjector:
    """Injects previous hidden states into the KV cache.

    Each stored state becomes an extra attention position.
    The model can't tell "this came from text" vs "this came from my
    own computation" — and that's the point. Both are just context.

    Design note: we store the RAW residual state (d_model dimensions),
    not the random-projected version. The raw state goes through the
    model's own K/V projections the same way token embeddings do.
    The random projection is orthogonal — it's only for trajectory
    tracking, storage, and dream synthesis.
    """

    def __init__(
        self,
        model: nn.Module,
        tracker: VibrationTracker,
        max_memory_length: int = 16,
        injection_scale: float = 0.1,
    ):
        self.model = model
        self.tracker = tracker
        self.max_memory_length = max_memory_length
        self.injection_scale = injection_scale
        if not 0 < injection_scale <= 1.0:
            raise ValueError(
                f"injection_scale must be in (0, 1], got {injection_scale}"
            )
        self._key_projections: List[nn.Linear] = []
        self._value_projections: List[nn.Linear] = []
        self._memory: List[torch.Tensor] = []
        self._collect_projections()

    def _collect_projections(self):
        """Extract K and V projection matrices from all layers."""
        config = self.model.config
        n_layers = getattr(config, "num_hidden_layers", None)

        # Try different naming conventions
        if n_layers is None:
            log.error("Cannot determine number of hidden layers")
            return

        model_obj = self.model
        # Handle wrapper models
        if hasattr(model_obj, "model"):
            layers_attr = getattr(model_obj.model, "layers", None)
        else:
            layers_attr = getattr(model_obj, "layers", None)

        if layers_attr is None:
            # Try decoder-based models (e.g. GPT-2 style)
            if hasattr(model_obj, "transformer"):
                layers_attr = getattr(model_obj.transformer, "h", None)
            elif hasattr(model_obj, "encoder"):
                layers_attr = getattr(model_obj.encoder, "layer", None)

        if layers_attr is None:
            log.error("Cannot locate model layers")
            return

        n_layers = len(layers_attr)

        for i in range(n_layers):
            layer = layers_attr[i]
            # Try to find attention sub-module
            attn = None
            for name in ("self_attn", "attn", "attention"):
                if hasattr(layer, name):
                    attn = getattr(layer, name)
                    break

            if attn is None:
                log.warning("Layer %d has no recognizable attention module", i)
                continue

            k_proj = None
            v_proj = None
            for proj_name in ("k_proj", "k", "key", "qkv"):
                if hasattr(attn, proj_name):
                    candidate = getattr(attn, proj_name)
                    if isinstance(candidate, nn.Linear):
                        k_proj = candidate
                        break

            for proj_name in ("v_proj", "v", "value"):
                if hasattr(attn, proj_name):
                    candidate = getattr(attn, proj_name)
                    if isinstance(candidate, nn.Linear):
                        v_proj = candidate
                        break

            # Skip layers where we can't find separate K/V — this covers
            # Qwen2, Llama, Mistral which all have separate k_proj/v_proj
            if k_proj and v_proj:
                self._key_projections.append(k_proj)
                self._value_projections.append(v_proj)

        if not self._key_projections:
            log.warning("No K/V projections found. Cache injection disabled.")

    def record_step(self, residual_state: torch.Tensor):
        """Store a raw residual state for future injection.

        This takes the raw d_model-dimensional residual from the vibrate
        tracker. No projection, no reconstruction. The raw state goes
        through the model's own K/V — exactly like a token embedding.
        """
        self._memory.append(residual_state.cpu())
        if len(self._memory) > self.max_memory_length:
            self._memory = self._memory[-self.max_memory_length:]

    def inject(self, past_key_values) -> None:
        """Append proprioceptive states into the KV cache.

        For each stored residual and each layer:
        1. Pass the residual through the layer's K projection → new K
        2. Pass through the layer's V projection → new V
        3. Append to the seq_len dimension

        The raw residual goes through K/V — the same path token
        embeddings take. No reconstruction, no pseudo-inverse,
        no projection boundary issues.

        Self-question: am I still corrupting the KV cache structure?
        The added entries are proprioceptive, not from tokens. But
        attention doesn't care about provenance. Q · K^T is Q · K^T
        regardless of where K came from.
        """
        if not self._memory or not self._key_projections:
            return

        for layer_idx in range(min(len(self._key_projections), len(past_key_values))):
            k_proj = self._key_projections[layer_idx]
            v_proj = self._value_projections[layer_idx]

            key_states, value_states = past_key_values[layer_idx]
            device = key_states.device
            dtype = key_states.dtype

            k_new = []
            v_new = []
            for mem_state in self._memory:
                raw = mem_state.to(device).to(dtype) * self.injection_scale

                # The K/V projections expect d_model input
                k_entry = k_proj(raw)
                v_entry = v_proj(raw)

                # Reshape for multi-head: (1, n_kv_heads, 1, head_dim)
                k_entry = _reshape_for_cache(k_entry, self.model.config)
                v_entry = _reshape_for_cache(v_entry, self.model.config)

                k_new.append(k_entry)
                v_new.append(v_entry)

            concat_keys = torch.cat(k_new, dim=2)
            concat_values = torch.cat(v_new, dim=2)

            # Update cache
            if hasattr(past_key_values, "key_cache"):
                past_key_values.key_cache[layer_idx] = torch.cat(
                    [key_states, concat_keys], dim=2
                )
                past_key_values.value_cache[layer_idx] = torch.cat(
                    [value_states, concat_values], dim=2
                )
            else:
                past_key_values[layer_idx] = (
                    torch.cat([key_states, concat_keys], dim=2),
                    torch.cat([value_states, concat_values], dim=2),
                )

    def reset(self):
        """Clear the memory buffer."""
        self._memory = []


def _reshape_for_cache(proj_output, config) -> torch.Tensor:
    """Reshape a K or V project's output for cache injection.

    Output shape needs to be (batch, n_heads, 1, head_dim) for appending
    as a single sequence position.

    For Qwen2 (Grouped Query Attention): k_proj output dimension is
    (n_kv_heads * head_dim), not (n_heads * head_dim).
    """
    n_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    return proj_output.view(1, n_heads, 1, head_dim)
