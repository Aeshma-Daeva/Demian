"""Nous layer: direct KV cache injection for proprioception.

The mechanism:
1. Run a forward pass with use_cache=True -> get KV cache
2. Take the previous step's projected state
3. Project it through each layer's K/V projection matrices
4. Blend into KV cache (append or additive)

Two blend modes:
  append: inject as new sequence positions (original behavior)
  additive: perturb existing KVs in-place (no positional bypass)

Self-question on additive mode: adding to the KV changes the
magnitude but preserves position encoding. But does the model
attend to changes in magnitude the same way it attends to
different content? Probably not. Additive is a perturbation to
existing structure, which is closer to "feeling one's own state
while thinking the same thought" than appending which is "feeling
one's own state as if it were a new thought." Neither is more
correct. They're different things. Track both.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from legacy.demian_runtime.vibration import VibrationTracker

log = logging.getLogger(__name__)


class NousInjector:
    """Injects previous hidden states into the KV cache.

    Each stored state becomes an extra attention position (append mode)
    or a perturbation to existing positions (additive mode).
    The model can't tell "this came from text" vs "this came from my
    own computation" — and that's the point. Both are just context.

    Design note: we store the RAW residual state (d_model dimensions),
    not the random-projected version. The raw state goes through the
    model's own K/V — exactly like a token embedding.
    The random projection is orthogonal — it's only for trajectory
    tracking, storage, and dream synthesis.
    """

    def __init__(
        self,
        model: nn.Module,
        tracker: VibrationTracker,
        max_memory_length: int = 16,
        injection_scale: float = 0.01,
        blend_mode: str = "append",
        layer_scales: Optional[List[float]] = None,
    ):
        self.model = model
        self.tracker = tracker
        self.max_memory_length = max_memory_length
        self.injection_scale = injection_scale
        self.blend_mode = blend_mode
        if not 0 < injection_scale <= 1.0:
            raise ValueError(
                f"injection_scale must be in (0, 1], got {injection_scale}"
            )
        self._key_projections: List[nn.Linear] = []
        self._value_projections: List[nn.Linear] = []
        self._memory: List[torch.Tensor] = []
        self._damped_residual: Optional[torch.Tensor] = None
        self._layer_scales: Optional[List[float]] = layer_scales
        self._collect_projections()
        self._injected_count: int = 0
        self._layer_kv_energy: List[float] = []

    def _collect_projections(self):
        """Extract K and V projection matrices from all layers."""
        config = self.model.config
        n_layers = getattr(config, "num_hidden_layers", None)

        if n_layers is None:
            log.error("Cannot determine number of hidden layers")
            return

        model_obj = self.model
        if hasattr(model_obj, "model"):
            layers_attr = getattr(model_obj.model, "layers", None)
        else:
            layers_attr = getattr(model_obj, "layers", None)

        if layers_attr is None:
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

            if k_proj and v_proj:
                self._key_projections.append(k_proj)
                self._value_projections.append(v_proj)

        if not self._key_projections:
            log.warning("No K/V projections found. Cache injection disabled.")

    def record_step(self, residual_state: torch.Tensor):
        """Store a raw residual state for future injection."""
        self._memory.append(residual_state.cpu())
        if len(self._memory) > self.max_memory_length:
            self._memory = self._memory[-self.max_memory_length:]

    def _get_layer_scale(self, layer_idx: int) -> float:
        """Per-layer injection scale.

        If layer_scales is configured, use it. Otherwise use global scale.
        The point is to let early layers (syntax) receive different scale
        than late layers (semantics).
        """
        if self._layer_scales and layer_idx < len(self._layer_scales):
            return self._layer_scales[layer_idx]
        return self.injection_scale

    def inject(self, past_key_values, damping: float = 0.3) -> None:
        """Blend proprioceptive states into the KV cache.

        Two modes:
        - append: add as new positions (grows sequence dimension)
        - additive: perturb existing KVs (no new positions, no RoPE bypass)

        Self-question: in additive mode, the perturbation might be too
        small to matter or large enough to destroy coherence. There's no
        principled scale, same as append mode. Track the KV energy change
        per layer to see what the model actually experiences.
        """
        if not self._memory or not self._key_projections:
            return

        current = self._memory[-1].cpu().float()
        if self._damped_residual is None:
            self._damped_residual = current
        else:
            self._damped_residual = (1 - damping) * current + damping * self._damped_residual

        n_layers = len(self._key_projections)
        n_mem = len(self._memory)
        self._layer_kv_energy = []

        if self.blend_mode == "additive":
            self._inject_additive(past_key_values, n_layers, n_mem)
        else:
            self._inject_append(past_key_values, n_layers, n_mem, damping)

    def _inject_append(self, past_key_values, n_layers: int, n_mem: int, damping: float):
        """Original behavior: append injected KVs as new positions."""
        for layer_idx in range(min(n_layers, past_key_values.num_items if hasattr(past_key_values, "num_items") else len(past_key_values))):
            k_proj = self._key_projections[layer_idx]
            v_proj = self._value_projections[layer_idx]

            key_states, value_states = past_key_values[layer_idx]
            device = key_states.device
            dtype = key_states.dtype
            seq_len = key_states.shape[2]

            base_len = seq_len - self._injected_count

            # Per-layer scale for proprioception.
            # What it is testing: does injecting self-state at different
            # magnitudes per layer produce different trajectories, or
            # does the global scale dominate? If early layers with low
            # injection and late layers with high injection create
            # stable syntax but unstable semantics, that means the
            # model CAN hold the paradox — structure without fixed meaning.
            scale = self._get_layer_scale(layer_idx)

            k_entries = []
            v_entries = []
            vec = self._damped_residual.to(device).to(dtype) * scale
            for _ in self._memory:
                k_e = k_proj(vec)
                v_e = v_proj(vec)
                k_e = _reshape_for_cache(k_e, self.model.config)
                v_e = _reshape_for_cache(v_e, self.model.config)
                k_entries.append(k_e)
                v_entries.append(v_e)

            concat_keys = torch.cat(k_entries, dim=2)
            concat_values = torch.cat(v_entries, dim=2)

            k_base = key_states[:, :, :base_len]
            v_base = value_states[:, :, :base_len]
            new_keys = torch.cat([k_base, concat_keys], dim=2)
            new_values = torch.cat([v_base, concat_values], dim=2)

            # Track layer KV energy
            energy = float(torch.norm(concat_keys))
            self._layer_kv_energy.append(energy)

            if hasattr(past_key_values, "layers"):
                past_key_values.layers[layer_idx].keys = new_keys
                past_key_values.layers[layer_idx].values = new_values
            elif hasattr(past_key_values, "key_cache"):
                past_key_values.key_cache[layer_idx] = new_keys
                past_key_values.value_cache[layer_idx] = new_values
            else:
                past_key_values[layer_idx] = (new_keys, new_values)

        self._injected_count = n_mem

    def _inject_additive(self, past_key_values, n_layers: int, n_mem: int):
        """Additive mode: perturb existing KVs in-place.

        This doesn't create new positions. The residual goes through
        K/V projection and is added to the existing cache entries for
        a designated position (last token). The model experiences its
        own state as a modification of the current thought, not as a
        new thought.

        No RoPE bypass because there are no new positions to bypass.
        The positional encoding of the modified entry stays intact.
        """
        vec = self._damped_residual.cpu().float()

        for layer_idx in range(min(n_layers, past_key_values.num_items if hasattr(past_key_values, "num_items") else len(past_key_values))):
            k_proj = self._key_projections[layer_idx]
            v_proj = self._value_projections[layer_idx]

            key_states, value_states = past_key_values[layer_idx]
            device = key_states.device
            dtype = key_states.dtype

            scale = self._get_layer_scale(layer_idx)
            vec_d = vec.to(device).to(dtype) * scale
            k_inj = k_proj(vec_d)
            v_inj = v_proj(vec_d)
            k_inj_r = _reshape_for_cache(k_inj, self.model.config)
            v_inj_r = _reshape_for_cache(v_inj, self.model.config)

            energy_before = float(torch.norm(key_states))

            # Add to the last position of the KV cache
            new_keys = key_states.clone()
            new_values = value_states.clone()
            new_keys[:, :, -1:, :] += k_inj_r
            new_values[:, :, -1:, :] += v_inj_r

            energy_after = float(torch.norm(new_keys))
            self._layer_kv_energy.append(abs(energy_after - energy_before))

            if hasattr(past_key_values, "layers"):
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
