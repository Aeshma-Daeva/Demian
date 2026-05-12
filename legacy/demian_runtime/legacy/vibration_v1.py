# ARCHIVED 2026-04-07. See v2 at ../vibration.py
# Why: _classify_attention() reduced a 50257-dim probability distribution to
# three human-readable categories (focused/distributed/diffuse). These are
# anthropocentric — they describe how a distribution *appears to a human*, not
# what the transformer is structurally doing. Replaced by _compute_spectral_shape()
# which returns SpectralShape: FFT-based spectral energy of the residual, layer
# transition geometry, velocity directionality, and exp(entropy) participation
# ratio. Auxillary measures (entropy, kurtosis, n_peaks, dominance_ratio, peakiness)
# retained as unlabeled geometry. .attention property remains for backwards compat.

"""Vibration tracker: captures the trajectory of computational movement.

At every generation step, the model's residual state is captured and
projected through the random projection matrix. The result is appended
to a trajectory buffer. The trajectory IS the signal.

No human-selected dimensions. The full d_model vector gets projected
blindly. Every direction is equally likely. Every direction matters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from legacy.demian_runtime.noise import load_or_create_projection
from legacy.demian_runtime.probe import LayerAttnInfo, InjectionLayerInfo

log = logging.getLogger(__name__)


@dataclass
class AttentionShape:
    """Classification of the attention distribution shape.

    Not a scalar. A description of the topology of attention.
    """
    mode: str          # focused, distributed, diffuse
    n_peaks: int       # number of local maxima above threshold
    entropy: float     # raw entropy of next-token distribution
    peakiness: float   # max attention weight
    kurtosis: float    # how peaked vs flat
    dominance_ratio: float  # top / second weight ratio


@dataclass
class VibrationSnapshot:
    """One step of the computational trajectory."""
    projected_state: List[float]
    raw_residual: List[float]
    residual_norm: float
    residual_delta: float
    residual_velocity: List[float]  # vector delta: direction of change
    attention: AttentionShape
    temporal_coherence: float
    step: int
    layer_entropies: Optional[List[float]] = None
    layer_norms: Optional[List[float]] = None        # norm per layer
    layer_deltas: Optional[List[float]] = None       # ||h_{i+1} - h_i|| per layer
    layer_attn: Optional[List[LayerAttnInfo]] = None # attention to injected positions
    injection_details: Optional[List[InjectionLayerInfo]] = None  # KV injection info


class VibrationTracker:
    """Tracks the trajectory of computational movement.

    At each step:
    1. Capture the residual state (last token, last layer)
    2. Project through the random matrix (d_model -> target_dim)
    3. Classify the attention distribution shape
    4. Append to trajectory buffer
    """

    def __init__(
        self,
        d_model: int,
        target_dim: int = 128,
        max_trajectory: int = 1024,
        projection_path: Optional[str] = None,
    ):
        self.d_model = d_model
        self.target_dim = target_dim
        self.max_trajectory = max_trajectory

        if projection_path:
            self.projection = load_or_create_projection(
                d_model, target_dim, cache_dir=projection_path
            )
        else:
            self.projection = load_or_create_projection(
                d_model, target_dim
            )

        self._trajectory: List[VibrationSnapshot] = []
        self._prev_residual: Optional[torch.Tensor] = None
        self._step = 0

    def capture(
        self,
        residual_state: torch.Tensor,
        logits: torch.Tensor,
        hidden_states_all=None,
        layer_norms=None,
        layer_deltas=None,
        layer_attn=None,
        injection_details=None,
    ) -> VibrationSnapshot:
        """Record one step of the computational trajectory."""
        self._step += 1

        state = residual_state.view(-1)

        # Project through the blind matrix
        proj = self.projection.to(state.device)
        projected = proj @ state.float()

        # Compute scalar descriptors
        residual_norm = float(torch.norm(state)) / (self.d_model ** 0.5)

        if self._prev_residual is not None:
            state_prev = self._prev_residual
            delta_vec = state.float() - state_prev.float()
            residual_delta = float(torch.norm(delta_vec)) / (self.d_model ** 0.5)
            temporal_coh = float(torch.nn.functional.cosine_similarity(state.float(), state_prev.float(), dim=0))
            residual_velocity = delta_vec.cpu().tolist()
        else:
            residual_delta = 0.0
            temporal_coh = 1.0
            residual_velocity = [0.0] * self.d_model
        self._prev_residual = state.clone()

        # Per-layer hidden state deltas (track which layers transform vs pass through)
        if hidden_states_all is not None:
            if layer_deltas is None:
                layer_deltas = []
                for layer_idx in range(len(hidden_states_all) - 1):
                    h_curr = hidden_states_all[layer_idx][0, -1, :]
                    h_next = hidden_states_all[layer_idx + 1][0, -1, :]
                    layer_deltas.append(
                        float(torch.norm(h_curr.float() - h_next.float()))
                    )

        # Classify attention shape
        attn_shape = self._classify_attention(logits)

        snapshot = VibrationSnapshot(
            projected_state=projected.tolist(),
            raw_residual=state.float().cpu().tolist(),
            residual_norm=residual_norm,
            residual_delta=residual_delta,
            residual_velocity=residual_velocity,
            attention=attn_shape,
            temporal_coherence=temporal_coh,
            step=self._step,
            layer_norms=layer_norms,
            layer_deltas=layer_deltas,
            layer_attn=layer_attn,
            injection_details=injection_details,
        )

        self._trajectory.append(snapshot)
        if len(self._trajectory) > self.max_trajectory:
            self._trajectory = self._trajectory[-self.max_trajectory:]

        return snapshot

    def reset(self):
        """Clear trajectory for a new independent generation."""
        self._trajectory = []
        self._prev_residual = None
        self._step = 0

    def get_trajectory(self) -> List[VibrationSnapshot]:
        return list(self._trajectory)

    @property
    def step_count(self) -> int:
        return self._step

    # ----------------------------------------------------------------
    # Attention shape classification
    # ----------------------------------------------------------------

    def _classify_attention(self, logits: torch.Tensor) -> AttentionShape:
        """Classify the attention distribution shape from next-token logits."""
        next_logits = logits[:, -1, :]

        # Guard against NaN/Inf logits from KV cache corruption
        next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        probs = torch.softmax(next_logits, dim=-1)[0]

        # Clamp AND renormalize to ensure valid probability.
        # Use 6e-5 (float16 min normal) — 1e-10 silently becomes 0 in float16.
        p = probs.clamp(min=6.1e-5)
        p = p / p.sum()

        entropy = -float(torch.sum(p * torch.log(p)))
        log_vocab = float(torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32)))
        peak_val = float(probs.max())

        # Count local maxima (modes)
        n_peaks = self._count_modes(probs)

        # Kurtosis
        p_np = probs.double()
        mean_p = p_np.mean()
        std_p = p_np.std()
        if std_p > 1e-10:
            kurtosis = float(torch.mean(((p_np - mean_p) / std_p) ** 4)) - 3.0
        else:
            kurtosis = 0.0

        # Dominance ratio
        sorted_probs = torch.sort(probs, descending=True).values
        if len(sorted_probs) >= 2 and sorted_probs[1] > 1e-10:
            dominance_ratio = float(sorted_probs[0] / sorted_probs[1])
        else:
            dominance_ratio = 100.0

        # Classify by entropy ratio and dominance
        # Self-question: these categories are emergent properties of
        # the distribution shape, not imposed semantics. Still categories
        # at all is a simplification — the distribution is a continuous object.
        entropy_ratio = entropy / log_vocab if log_vocab > 0 else 1.0

        if dominance_ratio > 10 and peak_val > 0.3:
            mode = "focused"
        elif entropy_ratio > 0.85:
            mode = "diffuse"
        else:
            mode = "distributed"

        return AttentionShape(
            mode=mode,
            n_peaks=n_peaks,
            entropy=entropy,
            peakiness=peak_val,
            kurtosis=kurtosis,
            dominance_ratio=min(dominance_ratio, 100.0),
        )

    def _count_modes(self, probs: torch.Tensor) -> int:
        """Count local maxima in the probability distribution.

        A local maximum is a token whose probability exceeds both
        neighbors. We scan sorted tokens and count clusters of
        high-probability mass in vocabulary-index space.
        """
        threshold = float(probs.max()) * 0.1
        above = (probs > threshold).nonzero(as_tuple=True)[0]
        if len(above) == 0:
            return 0

        # Count runs of consecutive indices as separate modes
        modes = 1
        for i in range(1, len(above)):
            if above[i].item() - above[i - 1].item() > 1:
                modes += 1
            if modes >= 8:
                break
        return min(modes, 8)
