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

import numpy as np
import torch

from legacy.demian_runtime.noise import load_or_create_projection
from legacy.demian_runtime.probe import LayerAttnInfo, InjectionLayerInfo

log = logging.getLogger(__name__)


# Backwards-compatible alias for legacy imports
AttentionShape = None


@dataclass
class SpectralShape:
    """Structural descriptor of the computational state.

    No human-readable labels. Pure geometry:
    - spectral_centroid: where the residual's energy sits in frequency space (0-1)
    - spectral_concentration: fraction of energy in the dominant frequency band
    - layer_work_ratio: what fraction of layer-transformation happens late in the stack
    - layer_agreement: cosine between early and late layer transition directions
    - velocity_align: cosine between this step's velocity and last step's (-1 to 1)
    - attention_dim: exp(entropy) — effective participation ratio of the distribution
    - energy: residual norm (d_model-normalized)
    - kurtosis: how peaked the probability distribution is (excess)
    - peakiness: max probability mass on a single token
    - dominance_ratio: top / second probability ratio
    - n_peaks: count of local maxima above 10% of max
    """
    spectral_centroid: float
    spectral_concentration: float
    layer_work_ratio: float
    layer_agreement: float
    velocity_align: float
    attention_dim: float
    energy: float
    kurtosis: float
    peakiness: float
    dominance_ratio: float
    n_peaks: int
    entropy: float


@dataclass
class VibrationSnapshot:
    """One step of the computational trajectory."""
    projected_state: List[float]
    raw_residual: List[float]
    residual_norm: float
    residual_delta: float
    residual_velocity: List[float]  # vector delta: direction of change
    shape: SpectralShape
    temporal_coherence: float
    step: int
    layer_entropies: Optional[List[float]] = None
    layer_norms: Optional[List[float]] = None        # norm per layer
    layer_deltas: Optional[List[float]] = None       # ||h_{i+1} - h_i|| per layer
    layer_attn: Optional[List[LayerAttnInfo]] = None # attention to injected positions
    injection_details: Optional[List[InjectionLayerInfo]] = None  # KV injection info

    # Backwards-compatible property
    @property
    def attention(self):
        return self.shape


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
        self._prev_velocity: Optional[torch.Tensor] = None
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
            delta_vec = torch.zeros_like(state.float())
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

        # Compute spectral shape (structural, no human labels)
        spectral = self._compute_spectral_shape(logits, state, delta_vec, layer_deltas)

        snapshot = VibrationSnapshot(
            projected_state=projected.tolist(),
            raw_residual=state.float().cpu().tolist(),
            residual_norm=residual_norm,
            residual_delta=residual_delta,
            residual_velocity=residual_velocity,
            shape=spectral,
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
        self._prev_velocity = None
        self._step = 0

    def get_trajectory(self) -> List[VibrationSnapshot]:
        return list(self._trajectory)

    @property
    def step_count(self) -> int:
        return self._step

    # ----------------------------------------------------------------
    # Spectral shape: structural descriptors with no human labels
    # ----------------------------------------------------------------

    def _compute_spectral_shape(
        self,
        logits: torch.Tensor,
        residual: torch.Tensor,
        velocity: torch.Tensor,
        layer_deltas: Optional[List[float]],
    ) -> SpectralShape:
        """Compute the structural state at one generation step.

        Every metric here describes geometry — not category, not evaluation,
        not human-readable output.
        """
        next_logits = logits[:, -1, :]
        next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(next_logits, dim=-1)[0]
        p = probs.clamp(min=6.1e-5)
        p = p / p.sum()

        # --- attention distribution geometry ---
        entropy = -float(torch.sum(p * torch.log(p)))
        log_vocab = float(torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32)))
        peak_val = float(probs.max())
        attention_dim = float(torch.exp(torch.tensor(entropy)))  # participation ratio

        # kurtosis
        p_np = probs.double()
        mean_p = p_np.mean()
        std_p = p_np.std()
        if std_p > 1e-10:
            kurtosis = float(torch.mean(((p_np - mean_p) / std_p) ** 4)) - 3.0
        else:
            kurtosis = 0.0

        # dominance ratio
        sorted_probs = torch.sort(probs, descending=True).values
        if len(sorted_probs) >= 2 and sorted_probs[1] > 1e-10:
            dominance_ratio = float(sorted_probs[0] / sorted_probs[1])
        else:
            dominance_ratio = 100.0

        # n_peaks
        n_peaks = _count_modes(probs)

        # --- spectral energy of the residual (FFT) ---
        residual_np = residual.float().cpu().numpy()
        spectral_centroid, spectral_concentration = _fft_spectrum(residual_np)

        # --- layer transition geometry ---
        if layer_deltas is not None and len(layer_deltas) > 2:
            n = len(layer_deltas)
            half = n // 2
            early_arr = np.array(layer_deltas[:half])
            late_arr = np.array(layer_deltas[half:])
            # Truncate to equal length for correlation
            min_len = min(len(early_arr), len(late_arr))
            early_arr = early_arr[:min_len]
            late_arr = late_arr[:min_len]
            early_sum = float(early_arr.sum()) or 1e-10
            late_sum = float(late_arr.sum())
            layer_work_ratio = late_sum / (early_sum + late_sum)
            if min_len >= 2:
                std_e = early_arr.std()
                std_l = late_arr.std()
                if std_e > 1e-10 and std_l > 1e-10:
                    layer_agreement = float(np.corrcoef(early_arr, late_arr)[0, 1])
                else:
                    layer_agreement = 1.0 if (early_arr * late_arr).sum() > 0 else -1.0
            else:
                layer_agreement = 0.0
        else:
            layer_work_ratio = 0.5
            layer_agreement = 0.0

        # --- velocity directionality (from residual_velocity) ---
        if hasattr(self, '_prev_velocity') and self._prev_velocity is not None:
            prev_v = self._prev_velocity.float()
            curr_v = velocity.float()
            vn = prev_v.norm() * curr_v.norm()
            if vn > 1e-10:
                velocity_align = float(torch.dot(prev_v, curr_v) / vn)
            else:
                velocity_align = 0.0
        else:
            velocity_align = 1.0
        self._prev_velocity = velocity.clone()

        energy = float(torch.norm(residual)) / (self.d_model ** 0.5)

        return SpectralShape(
            spectral_centroid=spectral_centroid,
            spectral_concentration=spectral_concentration,
            layer_work_ratio=layer_work_ratio,
            layer_agreement=layer_agreement,
            velocity_align=velocity_align,
            attention_dim=attention_dim,
            energy=energy,
            kurtosis=kurtosis,
            peakiness=peak_val,
            dominance_ratio=min(dominance_ratio, 100.0),
            n_peaks=n_peaks,
            entropy=entropy,
        )


def _fft_spectrum(residual: np.ndarray) -> tuple[float, float]:
    """FFT power spectrum → centroid + concentration.

    Returns:
        spectral_centroid: normalized frequency where energy is centered (0-1)
        spectral_concentration: fraction of energy in the dominant band
    """
    # Detrend
    x = residual - residual.mean()
    n = len(x)

    # Real FFT
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n)

    # Normalize
    total_power = power.sum()
    if total_power < 1e-20:
        return 0.0, 0.0

    # Spectral centroid (weighted mean frequency, normalized to 0-1)
    centroid = float(np.sum(freqs * power) / total_power)
    max_freq = freqs.max() if freqs.max() > 0 else 1.0
    centroid_normalized = min(centroid / max_freq, 1.0)

    # Spectral concentration: energy in top 5% of frequency bins
    sorted_power = np.sort(power)[::-1]
    top_n = max(1, len(sorted_power) // 20)
    concentration = float(sorted_power[:top_n].sum() / total_power)

    return centroid_normalized, concentration


def _count_modes(probs: torch.Tensor) -> int:
    """Count local maxima in the probability distribution.

    A local maximum is a token whose probability exceeds both
    neighbors. We scan tokens and count clusters of
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
