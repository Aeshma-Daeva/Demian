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
        self._damped_residual: Optional[torch.Tensor] = None  # smoothed state
        self._collect_projections()
        self._injected_count: int = 0

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

    def inject(self, past_key_values, damping: float = 0.3) -> None:
        """Append proprioceptive states into the KV cache.

        Each stored residual becomes extra KV entries. The cache grows
        linearly with injection count (max_memory_length total entries),
        not with generation steps. Previous injected entries are replaced.

        Damped: the injected vector is an exponential moving average of
        the current residual and the previous injected state. This smooths
        the feedback so the model can hold uncertainty across steps rather
        than being yanked between sharp states.

        Args:
            past_key_values: the KV cache to modify
            damping: EMA weight on the *previous* injected state.
                0 = no damping (inject raw current state).
                0.3 = 30% previous, 70% current (default).
                0.9 = heavy damping (barely changes).

        Self-question: am I still corrupting the KV cache structure?
        The added entries are proprioceptive, not from tokens. But
        attention doesn't care about provenance. Q · K^T is Q · K^T
        regardless of where K came from.
        """
        if not self._memory or not self._key_projections:
            return

        # Dampen: EMA between latest residual and previously injected state
        current = self._memory[-1].cpu().float()
        if self._damped_residual is None:
            self._damped_residual = current
        else:
            self._damped_residual = (1 - damping) * current + damping * self._damped_residual

        n_layers = len(self._key_projections)
        n_mem = len(self._memory)

        for layer_idx in range(min(n_layers, past_key_values.num_items if hasattr(past_key_values, "num_items") else len(past_key_values))):
            k_proj = self._key_projections[layer_idx]
            v_proj = self._value_projections[layer_idx]

            key_states, value_states = past_key_values[layer_idx]
            device = key_states.device
            dtype = key_states.dtype
            seq_len = key_states.shape[2]

            old_injected = self._injected_count
            base_len = seq_len - old_injected

            # Inject the damped single vector for each memory position
            # This gives the model a "momentum" version of its own state
            k_entries = []
            v_entries = []
            for mem_state in self._memory:
                # Blend each memory entry with the damped trajectory
                local_damped = (1 - damping) * mem_state.cpu().float() + damping * self._damped_residual
                raw = local_damped.to(device).to(dtype) * self.injection_scale
                k_e = k_proj(raw)
                v_e = v_proj(raw)
                k_e = _reshape_for_cache(k_e, self.model.config)
                v_e = _reshape_for_cache(v_e, self.model.config)
                k_entries.append(k_e)
                v_entries.append(v_e)

            concat_keys = torch.cat(k_entries, dim=2)
            concat_values = torch.cat(v_entries, dim=2)

            # Build new cache: keep base token entries, replace injected ones
            k_base = key_states[:, :, :base_len]
            v_base = value_states[:, :, :base_len]
            new_keys = torch.cat([k_base, concat_keys], dim=2)
            new_values = torch.cat([v_base, concat_values], dim=2)
            self._injected_count = n_mem

            # Directly set on DynamicCache layer — do NOT use .update()
            # which concatenates rather than replaces
            if hasattr(past_key_values, "layers"):
                # DynamicCache (transformers >= 4.45)
                past_key_values.layers[layer_idx].keys = new_keys
                past_key_values.layers[layer_idx].values = new_values
            elif hasattr(past_key_values, "key_cache"):
                past_key_values.key_cache[layer_idx] = new_keys
                past_key_values.value_cache[layer_idx] = new_values
            else:
                past_key_values[layer_idx] = (new_keys, new_values)

    def reset(self):
        """Clear the memory buffer."""
        self._memory = []
        self._injected_count = 0


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
