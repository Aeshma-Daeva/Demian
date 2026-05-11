"""Deep introspection: attention, Q@K, per-layer analysis.

Extracts what the model actually does with injected KV positions:
- Attention weights directed at injected positions per layer/head
- Pre-softmax Q@K scores for injected vs text positions
- Per-layer residual norms (what we had was deltas, not norms)
- KV directionality: how similar injected entries are to text entries

Qwen2.5-3B specifics:
  n_layers=36, n_q_heads=16, n_kv_heads=2, head_dim=128
  GQA ratio = 8 (each KV head shared by 8 Q heads)
  K/V cache shape: [1, 2, T, 128] — 2 KV heads, not 16
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class LayerAttnInfo:
    """Attention metrics for one layer at one generation step."""
    mean_injection_attn: float   # mean attn weight to injected positions
    text_to_injection_ratio: float  # text_total_attn / injection_total_attn
    head_std: float              # std across heads (head disagreement)
    qk_dot_product: float        # Q@K pre-softmax for injected positions


@dataclass
class InjectionLayerInfo:
    """Injection details for one layer."""
    layer_idx: int
    kv_cosine: float             # cosine between injected and text KV centroids
    injection_norm: float        # magnitude of injected K
    energy_before: float         # total KV cache norm before injection
    energy_after: float          # total KV cache norm after injection


@dataclass
class ProbeResult:
    """Full introspection data for one generation step."""
    layer_attn: List[LayerAttnInfo]
    layer_norms: List[float]     # norm of each layer's residual
    layer_deltas: List[float]    # ||h_{i+1} - h_i|| per layer transition
    injection_details: List[InjectionLayerInfo]


def probe_step(
    attentions,
    cache,
    injector,
    model,
    n_layers: int,
    injection_details: Optional[List[InjectionLayerInfo]] = None,
) -> ProbeResult:
    """Probe attention, Q@K, and per-layer state for one generation step.

    Args:
        attentions: tuple of attention weight tensors from model output,
            each [1, n_q_heads, 1, T] — post-softmax
        cache: the KV cache (DynamicCache-style or tuple of tuples)
        injector: NousInjector instance
        model: the model (needed for manual Q@K computation)
        n_layers: number of transformer layers
        injection_details: pre-populated injection info from the injector
    """
    n_injected = injector._injected_count
    layer_norms = []
    layer_deltas = []
    layer_attn = []

    for layer_idx in range(n_layers):
        # Attention weights: [1, n_heads, 1, T]
        attn = attentions[layer_idx]  # [1, 16, 1, T]
        T = attn.shape[-1]

        # Injection attention: last n_injected columns
        if n_injected > 0 and T > n_injected:
            attn_injected = attn[:, :, :, -n_injected:]  # [1, 16, 1, n_inj]
            attn_text = attn[:, :, :, :-n_injected]       # [1, 16, 1, T-n_inj]

            # Mean injection attention: average over heads and injected positions
            mean_inj = float(attn_injected.mean())

            # Text vs injection ratio: sum(text attn) / sum(injection attn)
            text_total = float(attn_text.sum())
            inj_total = float(attn_injected.sum())
            ratio = text_total / inj_total if inj_total > 1e-10 else float('inf')

            # Head std: std of mean injection attention across heads
            head_means = attn_injected.squeeze(0).squeeze(-1).mean(dim=-1)  # [16]
            head_std_ = float(head_means.std())
        else:
            mean_inj = 0.0
            ratio = 0.0
            head_std_ = 0.0

        # Q@K pre-softmax for injected positions (manual computation)
        qk_injected = float('nan')
        if n_injected > 0 and T > n_injected:
            qk_injected = _compute_qk_injected(
                model, cache, layer_idx, n_injected, n_layers
            )

        layer_attn.append(LayerAttnInfo(
            mean_injection_attn=mean_inj,
            text_to_injection_ratio=ratio if ratio != float('inf') else 1e6,
            head_std=head_std_,
            qk_dot_product=qk_injected,
        ))

    return ProbeResult(
        layer_attn=layer_attn,
        layer_norms=layer_norms,
        layer_deltas=layer_deltas,
        injection_details=injection_details or [],
    )


def _compute_qk_injected(
    model, cache, layer_idx: int, n_injected: int, n_layers: int,
) -> float:
    """Manually compute Q @ K^T / sqrt(d) for injected positions.

    The KV cache has already had RoPE applied during the model's forward
    pass, so the K values are RoPE'd. We need to apply RoPE to Q to match.

    However, recomputing Q through the layer requires the pre-attention
    hidden state, which we don't have access to here. Instead, we use a
    proxy: the attention scores we already have (post-softmax) can be
    back-transformed.

    Actually: we don't need manual Q@K. The attention weights from
    `output_attentions=True` are already softmax(Q@K/sqrt(d)).
    We can extract the log-scale pre-softmax values:

        log(attn / softmax_normalizer) = Q@K/sqrt(d) - max_score

    Since the softmax subtracts the max for numerical stability, we can
    recover raw scores up to an unknown constant offset per head/layer.

    For comparing injection vs text this constant cancels out in differences.

    Approach: use log(attn) as proxy for pre-softmax scores, clipped.

    Better approach (available now): just extract from attentions
    the raw values for injected positions. But attentions are post-softmax.

    We'll stick with: use log(attn) as a relative measure.
    """
    # We'll compute this properly in a later iteration using hooks.
    # For now, return NaN as placeholder.
    # The attention weights (post-softmax) are more useful than a fake Q@K.
    return float('nan')


def compute_layer_metrics(hidden_states_all) -> tuple[List[float], List[float]]:
    """Compute per-layer residual norms and deltas from hidden states."""
    if hidden_states_all is None:
        return [], []

    layer_norms = []
    layer_deltas = []

    for layer_idx in range(len(hidden_states_all)):
        h = hidden_states_all[layer_idx][0, -1, :]  # [d_model]
        layer_norms.append(float(torch.norm(h)))

        if layer_idx < len(hidden_states_all) - 1:
            h_next = hidden_states_all[layer_idx + 1][0, -1, :]
            delta = float(torch.norm(h_next.float() - h.float()))
            layer_deltas.append(delta)

    return layer_norms, layer_deltas


def compute_kv_directionality(cache, injector, n_layers: int) -> List[InjectionLayerInfo]:
    """Compute cosine similarity between injected and text KV entries per layer."""
    n_injected = injector._injected_count
    results = []

    for layer_idx in range(n_layers):
        try:
            k_tensor, v_tensor = cache[layer_idx]
        except (IndexError, TypeError):
            continue

        seq_len = k_tensor.shape[2]
        if n_injected <= 0 or seq_len <= n_injected:
            results.append(InjectionLayerInfo(
                layer_idx=layer_idx, kv_cosine=0.0, injection_norm=0.0,
                energy_before=0.0, energy_after=0.0,
            ))
            continue

        k_text = k_tensor[:, :, :-n_injected, :]   # [1, n_kv, T-n_inj, head_dim]
        k_inj = k_tensor[:, :, -n_injected:, :]     # [1, n_kv, n_inj, head_dim]

        # Centroids
        k_text_mean = k_text.mean(dim=2, keepdim=True)  # [1, n_kv, 1, head_dim]
        k_inj_mean = k_inj.mean(dim=2, keepdim=True)

        # Cosine similarity (sum over batch, KV heads, head_dim — just direction)
        cosine = float(F.cosine_similarity(
            k_text_mean.flatten(), k_inj_mean.flatten(), dim=0
        ))

        # Injection energy
        inj_norm = float(torch.norm(k_inj))

        results.append(InjectionLayerInfo(
            layer_idx=layer_idx,
            kv_cosine=cosine,
            injection_norm=inj_norm,
            energy_before=float(torch.norm(k_text)),
            energy_after=float(torch.norm(k_tensor)),
        ))

    return results
