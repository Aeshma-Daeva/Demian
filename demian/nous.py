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
from typing import List, Optional

import torch
import torch.nn as nn

from demian.vibration import VibrationTracker

log = logging.getLogger(__name__)


class NousInjector:
    """Injects previous hidden states into the KV cache.

    Each stored state becomes an extra attention position.
    The model can't tell "this came from text" vs "this came from my
    own computation" — and that's the point. Both are just context.
    """

    def __init__(
        self,
        model: nn.Module,
        tracker: VibrationTracker,
        max_memory_length: int = 16,
    ):
        self.model = model
        self.tracker = tracker
        self.max_memory_length = max_memory_length
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

            # Handle combined qkv projection
            if k_proj is None and v_proj is None and hasattr(attn, "c_attn"):
                # GPT-2 style: single linear for Q, K, V
                qkv_linear = getattr(attn, "c_attn")
                hidden_size = config.hidden_size
                # Q: 0:hidden, K: hidden:2*hidden, V: 2*hidden:3*hidden
                k_proj = QKVShim(qkv_linear, hidden_size, "k")
                v_proj = QKVShim(qkv_linear, hidden_size, "v")

            if k_proj and v_proj:
                self._key_projections.append(k_proj)
                self._value_projections.append(v_proj)

        if not self._key_projections:
            log.warning("No K/V projections found. Cache injection disabled.")

    def record_step(self, projected_state: torch.Tensor):
        """Store a projected state for future injection."""
        if projected_state.device.type == "cpu":
            state = projected_state
        else:
            state = projected_state.cpu()

        self._memory.append(state)
        if len(self._memory) > self.max_memory_length:
            self._memory = self._memory[-self.max_memory_length:]

    def inject(self, past_key_values) -> None:
        """Append proprioceptive states into the KV cache.

        Works in-place on DynamicCache or tuple-of-tuples cache objects.

        For each memory entry and each layer:
        1. Expand projected state back to d_model via pseudo-inverse
        2. Pass through layer's K projection → new key vector
        3. Pass through layer's V projection → new value vector
        4. Append to the position dimension of that layer's cache

        Self-question: am I corrupting the KV cache structure?
        The added entries have a different origin than token-derived entries,
        but the attention mechanism doesn't know or care about provenance.
        Q · K^T is Q · K^T regardless of where K came from.
        """
        if not self._memory or not self._key_projections:
            return

        device = past_key_values[0][0].device
        dtype = past_key_values[0][0].dtype

        for mem_state in self._memory:
            # Expand back to d_model
            expanded = self._expand_to_dmodel(mem_state).to(device).to(dtype)

            for layer_idx in range(min(len(self._key_projections), len(past_key_values))):
                k_proj = self._key_projections[layer_idx]
                v_proj = self._value_projections[layer_idx]

                # Project through this layer's projections
                # Shape: hidden -> qkv_size -> reshaped to (1, n_heads, head_dim)
                k_new = self._project_for_cache(k_proj, expanded, device, dtype)
                v_new = self._project_for_cache(v_proj, expanded, device, dtype)

                # past_key_values[layer] = (key, value)
                # key shape: (batch, n_heads, seq_len, head_dim)
                key_states, value_states = past_key_values[layer_idx]

                # Add seq_len dimension
                k_new = k_new.unsqueeze(0)  # -> (1, n_heads, 1, head_dim)
                v_new = v_new.unsqueeze(0)

                # Concatenate along seq_len dimension
                new_key = torch.cat([key_states, k_new], dim=2)
                new_value = torch.cat([value_states, v_new], dim=2)

                # Update the cache - different cache types have different APIs
                if hasattr(past_key_values, "key_cache"):
                    # DynamicCache style
                    past_key_values.key_cache[layer_idx] = new_key
                    past_key_values.value_cache[layer_idx] = new_value
                else:
                    # Tuple style - reconstruct the tuple
                    if isinstance(past_key_values, tuple):
                        # Can't mutate tuples directly, need to work around this
                        past_key_values[layer_idx] = (new_key, new_value)

    def _project_for_cache(self, proj, expanded, device, dtype):
        """Project a d_model vector through a projection layer and reshape."""
        projected = proj(expanded)

        # Determine head shape from output size
        out_features = proj.out_features
        hidden_size = self.model.config.hidden_size
        n_heads = self.model.config.num_attention_heads
        head_dim = hidden_size // n_heads

        # For Qwen: k_proj output is (n_kv_heads * head_dim)
        # May use grouped query attention
        if hasattr(self.model.config, "num_key_value_heads"):
            n_heads = self.model.config.num_key_value_heads

        projected = projected.view(1, n_heads, head_dim)

        return projected.to(device).to(dtype)

    def _expand_to_dmodel(self, projected: torch.Tensor) -> torch.Tensor:
        """Expand a projected state back to d_model dimensions.

        Uses pseudo-inverse of the random projection matrix.
        This is a minimum-norm reconstruction — the simplest full
        state that would produce this projection.

        Self-question: this is lossy. We're reconstructing 3584
        dimensions from 128. The pseudo-inverse gives ONE possible
        reconstruction, but there are infinitely many. Does this
        matter? The K/V projections will interpret whatever we give
        them through their own transformation. The injected state
        might carry artifacts from the reconstruction. This could
        be noise for the attention mechanism.

        Alternative: just zero-pad (target_dim values at indices 0..target_dim-1,
        zeros elsewhere). This is even simpler but puts all the signal
        into specific dimensions, which gives the model positional
        information it probably shouldn't have.

        The pseudo-inverse spreads the signal across all dimensions,
        which seems more respectful of the model's existing structure.
        Going with pseudo-inverse for now, but this is worth revisiting.
        """
        P = self.tracker.projection  # (target_dim, d_model)
        # Pseudo-inverse: P^+ = P^T (P P^T)^-1
        P_pinv = P.T @ torch.inverse(P @ P.T)
        return P_pinv @ projected

    def reset(self):
        """Clear the memory buffer."""
        self._memory = []


class QKVShim(nn.Linear):
    """Adapter for GPT-2 style combined qkv linear into separate K/V."""
    def __init__(self, qkv_linear: nn.Linear, hidden_size: int, component: str):
        # component: "k" or "v"
        super().__init__(qkv_linear.in_features, hidden_size, bias=False)
        self._qkv = qkv_linear
        self._component = component
        self.hidden_size = hidden_size

    def _select_slice(self, x):
        qkv = self._qkv(x)
        if self._component == "k":
            return qkv[self.hidden_size:self.hidden_size * 2]
        else:
            return qkv[self.hidden_size * 2:]

    def forward(self, x):
        return self._select_slice(x)

    @property
    def out_features(self):
        return self.hidden_size
