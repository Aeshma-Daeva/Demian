"""Force driver: structural self-description for reservoir dynamics.

Translates trajectory metrics → ForceVector (khaos_sigmata ontology) → injection tensor.
The system describes its own dynamical phase to itself. No human language. No output pressure.

Reference: /home/xenith/Documents/Xenith/recursion_engine/analysis/khaos_sigmata.py
Full 53-force ontology lives there. This module uses the 15 forces relevant to SSM dynamics.

Driver target: Criticality ⚛ — the phase transition threshold.
Not minimum energy (Stasis). Not maximum novelty (Chaos). The edge between them.

Replication forces (Memory⟳, Resonance〰, Rhythm♪) → Δ_slow scale.
Curiosity forces (Anomaly✶, Emergence△, Prediction_Error ε, Far-from-Equilibrium⚠) → Δ_fast scale.

Injection structure:
    injection = scale_t * residual_component + force_scale * force_tensor
Where:
    scale_t       = adaptive injection strength (criticality driver)
    force_tensor  = weighted sum of stable direction vectors per force
                    NOT random noise — structured phase encoding
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Force ontology subset — SSM-relevant forces
# Full: khaos_sigmata.py (53 forces, 7 categories)
# ---------------------------------------------------------------------------

DYNAMIC_FORCES: Dict[str, Dict] = {
    # TEMPORAL
    "Emergence":              {"sigil": "△", "category": "TEMPORAL"},
    "Decay":                  {"sigil": "▽", "category": "TEMPORAL"},
    "Memory":                 {"sigil": "⟳", "category": "TEMPORAL"},
    "Forgetting":             {"sigil": "⊘", "category": "TEMPORAL"},
    "Rhythm":                 {"sigil": "♪", "category": "TEMPORAL"},
    # THERMODYNAMIC
    "Intensity":              {"sigil": "⚡", "category": "THERMODYNAMIC"},
    "Dissipation":            {"sigil": "∿", "category": "THERMODYNAMIC"},
    "Resonance":              {"sigil": "〰", "category": "THERMODYNAMIC"},
    "Criticality":            {"sigil": "⚛", "category": "THERMODYNAMIC"},
    "Far_from_Equilibrium":   {"sigil": "⚠", "category": "THERMODYNAMIC"},
    # RELATIONAL / INFORMATION
    "Coherence":              {"sigil": "◈", "category": "RELATIONAL"},
    "Conflict":               {"sigil": "✕", "category": "RELATIONAL"},
    "Integration":            {"sigil": "⊕", "category": "INFORMATION"},
    "Differentiation":        {"sigil": "⊖", "category": "INFORMATION"},
    "Prediction_Error":       {"sigil": "ε",  "category": "INFORMATION"},
}

N_FORCES = len(DYNAMIC_FORCES)
FORCE_NAMES = list(DYNAMIC_FORCES.keys())

# Target force profile — what the system sustains.
# Criticality dominant. Memory/Forgetting balanced (selective retention).
# Non-zero Emergence (never fully static). Low Conflict, low Decay.
TARGET_PROFILE: Dict[str, float] = {
    "Criticality":          0.80,
    "Intensity":            0.50,
    "Resonance":            0.50,
    "Coherence":            0.45,
    "Integration":          0.45,
    "Memory":               0.40,
    "Prediction_Error":     0.40,
    "Differentiation":      0.35,
    "Forgetting":           0.30,
    "Emergence":            0.30,
    "Far_from_Equilibrium": 0.25,
    "Rhythm":               0.25,
    "Dissipation":          0.10,
    "Conflict":             0.10,
    "Decay":                0.10,
}

# Phase labels — structural only, no language judgement
PHASES = ["CRITICALITY", "GROWTH", "DECAY", "EQUILIBRIUM", "MUTATION", "COLLAPSE", "TRANSITION"]


# ---------------------------------------------------------------------------
# Stable force directions in residual space
# ---------------------------------------------------------------------------

def _stable_seed(name: str) -> int:
    """Deterministic seed from force name. Stable across Python runs."""
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % (2 ** 31)


def build_force_directions(d_model: int, dtype=torch.float32) -> Dict[str, torch.Tensor]:
    """Generate stable, approximately orthogonal direction vectors per force.

    Each force gets a fixed unit vector in residual space.
    Deterministic: same force name → same direction across all runs/processes.
    Approximately orthogonal by JL lemma (high-dim random vectors near-orthogonal).

    Returns dict: force_name → unit tensor of shape (d_model,)
    """
    directions = {}
    for fname in FORCE_NAMES:
        seed = _stable_seed(fname)
        rng = np.random.RandomState(seed)
        v = rng.randn(d_model).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        directions[fname] = torch.tensor(v, dtype=dtype)
    return directions


# ---------------------------------------------------------------------------
# Phase detection — structural, no language
# ---------------------------------------------------------------------------

def detect_phase(window: List[dict]) -> str:
    """Detect dynamical phase from trajectory window.

    Structural metrics only. No text quality evaluation.
    """
    if len(window) < 2:
        return "TRANSITION"

    energies  = [s["energy"] for s in window]
    rdeltas   = [s["residual_delta"] for s in window]
    tcohs     = [s["temporal_coherence"] for s in window]
    vel_aligns = [s.get("velocity_align", 0.0) for s in window]

    e_last    = energies[-1]
    e_first   = energies[0]
    e_trend   = e_last - e_first
    rdelta_arr = np.array(rdeltas, dtype=np.float32)
    rdelta_mean = float(rdelta_arr.mean())
    rdelta_last = float(rdelta_arr[-1])
    tcoh_mean = float(np.mean(tcohs))
    vel_std   = float(np.std(vel_aligns)) if len(vel_aligns) > 2 else 0.0

    # COLLAPSE: energy near zero
    if e_last < 0.05:
        return "COLLAPSE"

    # MUTATION: sudden spike in rdelta (> 3σ above window mean)
    if len(rdeltas) > 4:
        rdelta_std = float(rdelta_arr[:-1].std())
        if rdelta_std > 1e-6 and rdelta_last > rdelta_arr[:-1].mean() + 3 * rdelta_std:
            return "MUTATION"

    # EQUILIBRIUM: very low rdelta, high coherence — system froze
    if rdelta_mean < 0.03 and tcoh_mean > 0.92:
        return "EQUILIBRIUM"

    # CRITICALITY: rdelta moderate, vel_align oscillating — active edge
    if 0.04 < rdelta_mean < 0.35 and vel_std > 0.20:
        return "CRITICALITY"

    # GROWTH: rising energy, falling coherence — new patterns forming
    if e_trend > 0.10 and tcoh_mean < 0.50:
        return "GROWTH"

    # DECAY: falling energy
    if e_trend < -0.20:
        return "DECAY"

    return "TRANSITION"


# ---------------------------------------------------------------------------
# Metrics → ForceVector
# ---------------------------------------------------------------------------

def metrics_to_forces(
    window: List[dict],
    target_novelty: float = 0.15,
    prev_residual: Optional[torch.Tensor] = None,
    current_residual: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Translate trajectory statistics into force strengths [0, 1].

    Each force maps to a measurable property of the SSM dynamics.
    No semantic content. No language evaluation.

    Args:
        window:           recent trajectory steps (dicts with metric keys)
        target_novelty:   rdelta target for Criticality scoring
        prev_residual:    previous h_t for Prediction_Error
        current_residual: current h_t for Prediction_Error
    """
    if not window:
        return {f: 0.0 for f in FORCE_NAMES}

    recent   = window[-1]
    energies = [s["energy"] for s in window]
    rdeltas  = [s["residual_delta"] for s in window]
    tcohs    = [s["temporal_coherence"] for s in window]
    vel_aligns = [s.get("velocity_align", 0.0) for s in window]

    e          = recent["energy"]
    rdelta     = recent["residual_delta"]
    tcoh       = recent["temporal_coherence"]
    vel_align  = recent.get("velocity_align", 0.0)
    sconc      = recent.get("spectral_concentration", 0.5)
    variance   = recent.get("variance", 0.0)
    e_delta    = energies[-1] - energies[0] if len(energies) > 1 else 0.0

    def clamp(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    f: Dict[str, float] = {}

    # Emergence △: rdelta above target → new patterns forming
    f["Emergence"] = clamp(rdelta / (target_novelty * 2.0)) if rdelta > target_novelty else 0.0

    # Decay ▽: energy falling
    f["Decay"] = clamp(-e_delta / (energies[0] + 1e-10))

    # Memory ⟳: high coherence → patterns persisting
    f["Memory"] = clamp((tcoh + 1.0) / 2.0)

    # Forgetting ⊘: low coherence → patterns releasing
    f["Forgetting"] = clamp(1.0 - (tcoh + 1.0) / 2.0)

    # Rhythm ♪: autocorrelation of vel_align → periodic oscillation
    if len(vel_aligns) > 4:
        try:
            ac = float(np.corrcoef(vel_aligns[:-1], vel_aligns[1:])[0, 1])
            f["Rhythm"] = clamp(abs(ac))
        except Exception:
            f["Rhythm"] = 0.0
    else:
        f["Rhythm"] = 0.0

    # Intensity ⚡: raw energy (normalized ~0-5 range from experiments)
    f["Intensity"] = clamp(e / 5.0)

    # Dissipation ∿: energy falling trend (signed)
    f["Dissipation"] = clamp(-e_delta / 2.0)

    # Resonance 〰: positive vel_align → movement sustaining direction
    f["Resonance"] = clamp((vel_align + 1.0) / 2.0)

    # Criticality ⚛: proximity to target novelty
    # Maximum when rdelta == target_novelty; falls off in both directions
    f["Criticality"] = clamp(1.0 - abs(rdelta - target_novelty) / (target_novelty + 1e-10))

    # Far_from_Equilibrium ⚠: rdelta well above window mean → dissipative structure
    rdelta_mean = float(np.mean(rdeltas)) if rdeltas else 0.0
    if rdelta_mean > 1e-10:
        f["Far_from_Equilibrium"] = clamp((rdelta / rdelta_mean) - 1.0)
    else:
        f["Far_from_Equilibrium"] = 0.0

    # Coherence ◈: spectral concentration → state energy concentrated in few dimensions
    f["Coherence"] = clamp(sconc)

    # Conflict ✕: negative vel_align → direction reversal
    f["Conflict"] = clamp(-vel_align)

    # Integration ⊕: spectral concentration (same signal, information perspective)
    f["Integration"] = clamp(sconc)

    # Differentiation ⊖: variance relative to energy → complex internal structure
    f["Differentiation"] = clamp(variance / (e + 1e-10))

    # Prediction_Error ε: deviation of h_t from linear prediction
    if prev_residual is not None and current_residual is not None:
        with torch.no_grad():
            p = prev_residual.float()
            c = current_residual.float()
            pred_err = float(torch.norm(c - p) / (torch.norm(p) + 1e-10))
        f["Prediction_Error"] = clamp(pred_err)
    else:
        f["Prediction_Error"] = clamp(rdelta)

    return f


def forces_to_sigil(forces: Dict[str, float], top_n: int = 6) -> str:
    """Encode force strengths as sigil notation.

    Output: '⚛0.82 ⚡0.51 〰0.48 ...'
    No human words. Structural self-description.
    """
    ranked = sorted(forces.items(), key=lambda x: -x[1])[:top_n]
    parts = []
    for fname, strength in ranked:
        if strength > 0.05 and fname in DYNAMIC_FORCES:
            sigil = DYNAMIC_FORCES[fname]["sigil"]
            parts.append(f"{sigil}{strength:.2f}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Force deviation from target — how far from sustaining Criticality
# ---------------------------------------------------------------------------

def compute_target_deviation(forces: Dict[str, float]) -> float:
    """Mean absolute deviation of current forces from TARGET_PROFILE."""
    total = 0.0
    for fname in FORCE_NAMES:
        total += abs(forces.get(fname, 0.0) - TARGET_PROFILE.get(fname, 0.0))
    return total / N_FORCES


# ---------------------------------------------------------------------------
# CriticalityDriver — stateful, step-by-step
# ---------------------------------------------------------------------------

class CriticalityDriver:
    """Drives reservoir dynamics toward Criticality target ⚛.

    Each step:
        1. Update window with new metrics
        2. Compute ForceVector from window
        3. Compute adaptive injection signal
        4. Return (injection_tensor, injection_scale, delta_modulation)

    Injection structure:
        injection = scale * h_t + force_scale * force_tensor

    force_tensor: weighted sum of stable force direction vectors.
    scale: adaptive — phase-dependent, driven by curiosity/replication balance.
    delta_modulation: (slow_scale, fast_scale) for dual-Δ implementation.
    """

    def __init__(
        self,
        d_model: int,
        target_novelty: float = 0.15,
        window_size: int = 20,
        base_scale: float = 0.01,
        force_scale: float = 0.005,
        dtype=torch.float32,
        device: str = "cpu",
    ):
        self.d_model        = d_model
        self.target_novelty = target_novelty
        self.window_size    = window_size
        self.base_scale     = base_scale
        self.force_scale    = force_scale
        self.dtype          = dtype
        self.device         = device

        # Stable direction vectors for each force in residual space
        self._directions = build_force_directions(d_model, dtype)

        self._window: List[dict] = []
        self._prev_residual: Optional[torch.Tensor] = None

        # Readable state (for logging / trajectory record)
        self.forces: Dict[str, float]  = {f: 0.0 for f in FORCE_NAMES}
        self.phase: str                 = "TRANSITION"
        self.sigil: str                 = ""
        self.target_deviation: float    = 1.0
        self.injection_scale: float     = base_scale

        log.info(
            "CriticalityDriver: d_model=%d target_novelty=%.3f window=%d "
            "base_scale=%.4f force_scale=%.5f",
            d_model, target_novelty, window_size, base_scale, force_scale,
        )

    # ------------------------------------------------------------------

    def step(
        self,
        step_metrics: dict,
        current_residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, Tuple[float, float]]:
        """Process one reservoir step.

        Args:
            step_metrics:      dict from reservoir loop (energy, rdelta, tcoh, ...)
            current_residual:  h_t at this step (before next forward pass)

        Returns:
            (injection, scale, (slow_scale, fast_scale))
            injection: tensor of shape (d_model,) — inject as layer-0 input next step
            scale:     float — adaptive injection magnitude logged for trajectory
            (slow, fast): Δ modulation scalars
        """
        # Update window
        self._window.append(step_metrics)
        if len(self._window) > self.window_size:
            self._window.pop(0)

        cr = current_residual.detach()

        # Compute forces
        self.forces = metrics_to_forces(
            self._window,
            target_novelty=self.target_novelty,
            prev_residual=self._prev_residual,
            current_residual=cr,
        )
        self.phase            = detect_phase(self._window)
        self.sigil            = forces_to_sigil(self.forces)
        self.target_deviation = compute_target_deviation(self.forces)

        # Adaptive injection scale
        self.injection_scale = self._adaptive_scale()

        # Build force tensor in residual space
        force_tensor = self._build_force_tensor()

        # Final injection: residual component + structural force encoding
        c_norm = cr.float().norm().item()
        if c_norm > 1e-10:
            normalized_cr = (cr.float() * (1.0 / c_norm)).to(self.dtype)
        else:
            normalized_cr = cr.to(self.dtype)

        injection = (
            self.injection_scale * normalized_cr.to(self.device)
            + self.force_scale   * force_tensor.to(self.device)
        )

        self._prev_residual = cr.clone()

        return injection, self.injection_scale, self.delta_modulation()

    # ------------------------------------------------------------------

    def _build_force_tensor(self) -> torch.Tensor:
        """Weighted sum of stable force directions."""
        t = torch.zeros(self.d_model, dtype=self.dtype)
        for fname, strength in self.forces.items():
            if strength > 0.01:
                t += strength * self._directions[fname]
        n = t.norm().item()
        if n > 1e-10:
            t = t / n
        return t

    def _adaptive_scale(self) -> float:
        """Phase-aware injection scale.

        Curiosity pressure:   stasis/collapse → push harder
        Replication pressure: mutation/growth → ease off
        Criticality target:   maintain when already there
        """
        # Phase modifier
        phase_mod = {
            "COLLAPSE":     3.0,   # Dying — inject hard
            "EQUILIBRIUM":  2.5,   # Frozen — push
            "DECAY":        1.5,   # Fading — nudge
            "TRANSITION":   1.0,   # Neutral
            "CRITICALITY":  1.0,   # Target reached — maintain
            "GROWTH":       0.6,   # Expanding — ease off
            "MUTATION":     0.3,   # Perturbed — ease way off
        }.get(self.phase, 1.0)

        # Curiosity (novelty pressure) vs Replication (stability pressure)
        curiosity = (
            self.forces.get("Prediction_Error",     0.0) * 0.40 +
            self.forces.get("Far_from_Equilibrium", 0.0) * 0.35 +
            self.forces.get("Emergence",            0.0) * 0.25
        )
        replication = (
            self.forces.get("Memory",    0.0) * 0.45 +
            self.forces.get("Resonance", 0.0) * 0.35 +
            self.forces.get("Rhythm",    0.0) * 0.20
        )

        # Target deviation: far from target profile → increase drive
        deviation_boost = 1.0 + self.target_deviation * 0.5

        balance = 1.0 + 0.5 * curiosity - 0.3 * replication
        scale = self.base_scale * phase_mod * float(np.clip(balance, 0.1, 4.0)) * deviation_boost

        return float(np.clip(scale, 1e-4, 0.50))

    def delta_modulation(self) -> Tuple[float, float]:
        """Dual-Δ scales for long/short memory separation.

        slow_scale: high when replication forces dominant
            → low Δ_slow → Ā_slow ≈ 1 → long memory preserved
        fast_scale: high when curiosity forces dominant
            → high Δ_fast → Ā_fast << 1 → short memory, novelty-sensitive

        Returns (slow_scale, fast_scale) in [0, 1].
        """
        slow = (
            self.forces.get("Memory",    0.0) * 0.40 +
            self.forces.get("Resonance", 0.0) * 0.35 +
            self.forces.get("Rhythm",    0.0) * 0.25
        )
        fast = (
            self.forces.get("Prediction_Error",     0.0) * 0.40 +
            self.forces.get("Far_from_Equilibrium", 0.0) * 0.30 +
            self.forces.get("Emergence",            0.0) * 0.30
        )
        return float(np.clip(slow, 0.05, 1.0)), float(np.clip(fast, 0.05, 1.0))

    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serializable snapshot for trajectory logging."""
        return {
            "phase":            self.phase,
            "sigil":            self.sigil,
            "target_deviation": self.target_deviation,
            "injection_scale":  self.injection_scale,
            "forces":           dict(self.forces),
            "delta_slow":       self.delta_modulation()[0],
            "delta_fast":       self.delta_modulation()[1],
        }
