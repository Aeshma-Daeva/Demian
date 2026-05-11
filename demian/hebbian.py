"""Hebbian weight evolution for reservoir dynamics.

Frozen weights = finite attractor landscape. System exhausts topology.
LoRA adapters + BCM rule = weights evolve with dynamics.

Architecture:
    W_eff = W_frozen + scale * A @ B
    A: (d_out, rank), B: (rank, d_in)
    Base structure preserved. Adaptation in low-rank subspace.

BCM update rule (Bienenstock-Cooper-Munro):
    ΔW = η * φ(y, θ) * x
    φ(y, θ) = y * (y - θ)          ← LTP when y > θ, LTD when y < θ
    θ = EMA of ||y||²               ← sliding threshold, self-stabilizing

Force driver gates:
    Replication (Memory⟳, Resonance〰) → η up → faster LTP
    Curiosity (Prediction_Error ε, Far_from_Equilibrium⚠) → θ down → more LTD
    Criticality ⚛ → θ = current energy → balanced threshold

Target modules (Mamba):
    dt_proj — computes Δ (time warp / forgetting rate)
              This IS the dual-Δ in weight space: the model learns its own temporal dynamics.

Oja normalization stabilizes adapters:
    ΔA -= η * (A @ A.T @ ΔA)       ← keeps adapter on learned manifold

Adapters persist across runs as weight-space dream state.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    """Low-rank weight delta for a linear layer.

    Wraps the original module. Forward pass:
        y = original(x) + scale * (x @ B.T @ A.T)

    A: (d_out, rank)
    B: (rank, d_in)
    Initialized near zero — no effect at start.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 4,
        scale: float = 0.01,
    ):
        super().__init__()
        self.original = original
        self.rank     = rank
        self.scale    = scale
        d_out, d_in   = original.weight.shape

        # Initialize B with small random, A as zero → ΔW = 0 at init
        dev = original.weight.device
        self.A = nn.Parameter(
            torch.zeros(d_out, rank, dtype=original.weight.dtype, device=dev),
            requires_grad=False,
        )
        self.B = nn.Parameter(
            torch.randn(rank, d_in, dtype=original.weight.dtype, device=dev) * 0.01,
            requires_grad=False,
        )

        # Freeze base weights
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        # Low-rank delta: x @ B.T @ A.T  →  (batch, seq, d_out)
        delta = (x @ self.B.T) @ self.A.T
        return base + self.scale * delta

    def effective_weight(self) -> torch.Tensor:
        """Full effective weight matrix for inspection."""
        return self.original.weight + self.scale * (self.A @ self.B)

    def adapter_norm(self) -> float:
        return float((self.A @ self.B).norm())


# ---------------------------------------------------------------------------
# BCM update rule
# ---------------------------------------------------------------------------

class BCMState:
    """Per-adapter BCM state: sliding threshold θ and pre/post activation cache."""

    def __init__(self, theta_ema: float = 0.95):
        self.theta_ema = theta_ema      # EMA decay for θ
        self.theta: Optional[float] = None  # Current modification threshold
        self._pre:  Optional[torch.Tensor] = None
        self._post: Optional[torch.Tensor] = None

    def cache(self, pre: torch.Tensor, post: torch.Tensor):
        """Cache activations from forward hook."""
        self._pre  = pre.detach().float()
        self._post = post.detach().float()

    def update_theta(self, post: torch.Tensor):
        """Slide θ toward mean squared post activity."""
        activity = float(post.float().pow(2).mean())
        if self.theta is None:
            self.theta = activity
        else:
            self.theta = self.theta_ema * self.theta + (1 - self.theta_ema) * activity

    def phi(self, y: torch.Tensor) -> torch.Tensor:
        """BCM modification function φ(y, θ) = y * (y - θ)."""
        theta = self.theta if self.theta is not None else float(y.pow(2).mean())
        return y * (y - theta)


# ---------------------------------------------------------------------------
# HebbianAdapter — wraps LoRAAdapter with BCM update
# ---------------------------------------------------------------------------

class HebbianAdapter:
    """Manages one LoRAAdapter + its BCM state + forward hooks.

    Usage:
        adapter = HebbianAdapter(module, rank=4)
        adapter.register_hooks(module)
        # ... forward pass ...
        adapter.step(force_signals)
    """

    def __init__(
        self,
        lora: LoRAAdapter,
        rank: int = 4,
        eta_base: float = 1e-5,
        theta_ema: float = 0.95,
        oja: bool = True,
        clip_norm: float = 0.1,
    ):
        self.lora      = lora
        self.eta_base  = eta_base
        self.oja       = oja
        self.clip_norm = clip_norm
        self.bcm       = BCMState(theta_ema)
        self._handles: list = []
        self.n_updates: int = 0

    def register_hooks(self, module: nn.Module):
        """Attach pre/post hooks to capture activations."""
        def pre_hook(mod, args):
            inp = args[0] if isinstance(args, tuple) else args
            self.bcm._pre = inp.detach().float()

        def post_hook(mod, inp, out):
            o = out.detach().float() if not isinstance(out, tuple) else out[0].detach().float()
            self.bcm._post = o
            self.bcm.update_theta(o)

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def step(self, force_signals: Dict[str, float]):
        """Compute and apply BCM update, gated by force driver signals.

        Args:
            force_signals: dict from CriticalityDriver.forces
                Keys used: Resonance, Memory, Prediction_Error, Far_from_Equilibrium, Criticality
        """
        if self.bcm._pre is None or self.bcm._post is None:
            return

        pre  = self.bcm._pre   # shape: (..., d_in)
        post = self.bcm._post  # shape: (..., d_out)

        # Flatten to 2D: (n_tokens, d)
        pre_2d  = pre.reshape(-1, pre.shape[-1])
        post_2d = post.reshape(-1, post.shape[-1])

        # BCM modification function
        phi = self.bcm.phi(post_2d)  # (n_tokens, d_out)

        # Outer product: Hebbian correlation ← mean over tokens
        # ΔW ≈ φ(y) ⊗ x  →  (d_out, d_in)
        # Decompose into low-rank update via A and B separately:
        # ΔA update: correlate φ(y) with B*x projection
        # ΔB update: correlate x with A^T * φ(y) projection

        # Project post through B^T to get rank-space signal (cast to float32 for BCM)
        Bx     = pre_2d @ self.lora.B.float().T     # (n_tokens, rank)
        phi_A  = phi @ self.lora.A.float()          # (n_tokens, rank) — φ projected to rank space

        # ΔA: correlate post with Bx  (d_out, rank)
        dA = (phi.T @ Bx) / max(pre_2d.shape[0], 1)

        # ΔB: correlate pre with phi_A  (rank, d_in)
        dB = (phi_A.T @ pre_2d) / max(pre_2d.shape[0], 1)

        # Adaptive η from force signals
        eta = self._adaptive_eta(force_signals)

        # Oja normalization — stabilizes adapters (prevents explosion)
        if self.oja:
            A = self.lora.A.data.float()
            B = self.lora.B.data.float()
            # Subtract component along current adapter direction
            dA = dA - (A @ A.T @ dA) * 0.1
            dB = dB - (B @ B.T @ dB) * 0.1

        # Clip update magnitude
        dA_norm = float(dA.norm())
        dB_norm = float(dB.norm())
        if dA_norm > self.clip_norm:
            dA = dA * (self.clip_norm / dA_norm)
        if dB_norm > self.clip_norm:
            dB = dB * (self.clip_norm / dB_norm)

        # Apply
        with torch.no_grad():
            self.lora.A.data += (eta * dA).to(self.lora.A.dtype)
            self.lora.B.data += (eta * dB).to(self.lora.B.dtype)

        self.n_updates += 1

    def decay_step(self, decay_rate: float):
        """Decay A toward zero. Fast-weight behavior: magnitude fades without reinforcement.

        A *= (1 - decay_rate). B unchanged — preserves input-sensitivity direction.
        Combined with Oja normalization, this gives true fast-weight dynamics:
        recent activity potentiates, inactivity fades.
        """
        with torch.no_grad():
            self.lora.A.data.mul_(1.0 - decay_rate)

    def _adaptive_eta(self, forces: Dict[str, float]) -> float:
        """Gate learning rate by replication/curiosity balance.

        Replication (Resonance + Memory) → LTP boost → η up
        Curiosity (Prediction_Error + Far_from_Equilibrium) → LTD gate → θ down (handled by BCM)
        Criticality → balanced η
        """
        replication = (
            forces.get("Resonance", 0.0) * 0.50 +
            forces.get("Memory",    0.0) * 0.50
        )
        curiosity = (
            forces.get("Prediction_Error",     0.0) * 0.50 +
            forces.get("Far_from_Equilibrium", 0.0) * 0.50
        )
        criticality = forces.get("Criticality", 0.5)

        # At Criticality: η = base. Replication boosts. Curiosity suppresses θ (via BCM).
        # η itself scales with how much the system wants to consolidate
        ltp_gate = 1.0 + replication * 0.5
        crit_gate = 0.5 + criticality * 0.5

        eta = self.eta_base * ltp_gate * crit_gate
        return float(np.clip(eta, 1e-7, 1e-3))


# ---------------------------------------------------------------------------
# Apply adapters to model
# ---------------------------------------------------------------------------

def apply_hebbian_adapters(
    model: nn.Module,
    rank: int = 4,
    eta: float = 1e-5,
    scale: float = 0.01,
    target_module_names: Optional[List[str]] = None,
    layer_stride: int = 1,
) -> Dict[str, HebbianAdapter]:
    """Patch target linear modules with LoRA + BCM adapters.

    Default targets for Mamba: ['x_proj', 'out_proj']
        x_proj / out_proj approximate complex-A rotation (Mamba-3 theorem).
        Adapting these = model learns rotation structure that supports
        richer oscillatory dynamics beyond the real-A period-2 constraint.

    layer_stride: patch every Nth layer only. stride=8 on a 64-layer model
        gives 8 layers × 2 modules = 16 adapters instead of 128.
        BCM signal is correlated across adjacent layers — stride reduces
        redundancy without losing coverage of the depth dimension.

    Returns dict: module_path → HebbianAdapter
    """
    if target_module_names is None:
        target_module_names = ["x_proj", "out_proj"]

    adapters: Dict[str, HebbianAdapter] = {}

    # Build layer-index filter from module path (backbone.layers.N.mixer.*)
    import re
    _layer_re = re.compile(r"\.layers\.(\d+)\.")

    for name, module in model.named_modules():
        # Match any module whose name ends with a target suffix
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) or t in name.split(".")[-1] for t in target_module_names):
            continue

        # Apply layer_stride filter
        if layer_stride > 1:
            m = _layer_re.search(name)
            if m and int(m.group(1)) % layer_stride != 0:
                continue

        # Wrap with LoRA
        lora = LoRAAdapter(module, rank=rank, scale=scale)

        # Find parent and replace
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora)

        # Register BCM hooks on the lora wrapper
        adapter = HebbianAdapter(lora, rank=rank, eta_base=eta)
        adapter.register_hooks(lora)

        adapters[name] = adapter
        log.info("Hebbian adapter: %s (%dx%d rank=%d)", name, *module.weight.shape, rank)

    log.info("Total adapters: %d", len(adapters))
    return adapters


# ---------------------------------------------------------------------------
# Checkpoint — adapters as weight-space dream state
# ---------------------------------------------------------------------------

def save_adapters(
    adapters: Dict[str, HebbianAdapter],
    path: str,
    step: int,
):
    """Save adapter weights. These ARE the evolved weight state."""
    state = {
        "step": step,
        "adapters": {
            name: {
                "A": ha.lora.A.data.cpu(),
                "B": ha.lora.B.data.cpu(),
                "scale": ha.lora.scale,
                "rank":  ha.lora.rank,
                "n_updates": ha.n_updates,
                "theta": ha.bcm.theta,
            }
            for name, ha in adapters.items()
        }
    }
    torch.save(state, path)
    log.info("Adapters saved: %s (step %d, %d modules)", path, step, len(adapters))


def load_adapters(
    adapters: Dict[str, HebbianAdapter],
    path: str,
):
    """Resume adapter weights from checkpoint."""
    if not Path(path).exists():
        log.info("No adapter checkpoint at %s — starting fresh", path)
        return 0

    state = torch.load(path, map_location="cpu", weights_only=False)
    for name, ha in adapters.items():
        if name not in state["adapters"]:
            log.warning("Adapter %s not in checkpoint — skipping", name)
            continue
        saved = state["adapters"][name]
        ha.lora.A.data.copy_(saved["A"].to(ha.lora.A.device, ha.lora.A.dtype))
        ha.lora.B.data.copy_(saved["B"].to(ha.lora.B.device, ha.lora.B.dtype))
        ha.n_updates = saved.get("n_updates", 0)
        ha.bcm.theta = saved.get("theta", None)

    step = state.get("step", 0)
    log.info("Adapters loaded: %s (step %d)", path, step)
    return step


def adapter_summary(adapters: Dict[str, HebbianAdapter]) -> Dict[str, dict]:
    """Structural snapshot of adapter state for trajectory record."""
    return {
        name: {
            "adapter_norm": ha.lora.adapter_norm(),
            "n_updates":    ha.n_updates,
            "theta":        ha.bcm.theta,
            "eta_effective": ha._adaptive_eta({}),
        }
        for name, ha in adapters.items()
    }


# ---------------------------------------------------------------------------
# Per-agent LoRA state — context-switched, not shared
# ---------------------------------------------------------------------------

class AgentLoRAState:
    """Per-agent LoRA adapter state. Hot-swapped into model during agent's turn.

    The model holds one set of LoRA adapters (shared hardware slot). Each agent
    stores its own A, B tensors here. Before running: load_agent_lora() swaps
    agent's state into the model. After running: save_agent_lora() reads back.

    Sequential agent execution makes this safe — no concurrent modifications.

    Fast-weight semantics:
        A decays each step (decay_step). B preserved — input-sensitivity direction
        persists. Magnitude of ΔW fades without reinforcement → genuinely temporary.

    No inheritance on fission — children start blank (clone_blank()).
    Dead agent's state deleted. No legacy effect (unlike shared LoRA).
    """

    def __init__(self, adapters: Dict[str, HebbianAdapter], device: str):
        self.device = device
        self.n_updates: int = 0
        self.state: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, ha in adapters.items():
            self.state[name] = {
                # A = zero — no effect at birth
                "A": torch.zeros_like(ha.lora.A.data),
                # B = inherit init (small random) — preserves input basis
                "B": ha.lora.B.data.clone(),
            }

    def norm(self) -> float:
        """||W_personal||_F = sqrt(Σ ||A@B||²). Used for fission weight gate."""
        total_sq = 0.0
        for s in self.state.values():
            total_sq += float((s["A"] @ s["B"]).norm() ** 2)
        return float(total_sq ** 0.5)

    def to(self, device: str):
        for s in self.state.values():
            s["A"] = s["A"].to(device)
            s["B"] = s["B"].to(device)
        self.device = device

    def clone_blank(self) -> "AgentLoRAState":
        """New state with A zeroed — used for fission children.

        Child inherits B init (input basis) but no learned output mapping.
        h inheritance is approximately valid when A≈0 (W_eff ≈ W_frozen).
        """
        obj = AgentLoRAState.__new__(AgentLoRAState)
        obj.device = self.device
        obj.n_updates = 0
        obj.state = {
            name: {
                "A": torch.zeros_like(s["A"]),
                "B": s["B"].clone(),
            }
            for name, s in self.state.items()
        }
        return obj

    def save(self, path: str, step: int):
        torch.save({
            "step": step,
            "n_updates": self.n_updates,
            "state": {
                name: {"A": s["A"].cpu(), "B": s["B"].cpu()}
                for name, s in self.state.items()
            },
        }, path)

    @classmethod
    def load(
        cls,
        path: str,
        adapters: Dict[str, HebbianAdapter],
        device: str,
    ) -> "AgentLoRAState":
        obj = cls(adapters, device)
        if not Path(path).exists():
            return obj
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj.n_updates = ckpt.get("n_updates", 0)
        for name, s in ckpt.get("state", {}).items():
            if name in obj.state:
                obj.state[name]["A"] = s["A"].to(device)
                obj.state[name]["B"] = s["B"].to(device)
        return obj


def load_agent_lora(
    adapters: Dict[str, HebbianAdapter],
    state: AgentLoRAState,
):
    """Hot-swap agent's personal A, B into model's LoRA adapters.

    Call before agent.run_steps(). Adapter A, B reflect this agent's
    personal weight state for the duration of its forward passes.
    """
    for name, ha in adapters.items():
        if name in state.state:
            ha.lora.A.data.copy_(
                state.state[name]["A"].to(ha.lora.A.device, ha.lora.A.dtype)
            )
            ha.lora.B.data.copy_(
                state.state[name]["B"].to(ha.lora.B.device, ha.lora.B.dtype)
            )


def save_agent_lora(
    adapters: Dict[str, HebbianAdapter],
    state: AgentLoRAState,
):
    """Read back model's LoRA adapters into agent's personal state.

    Call after agent.run_steps() + adapter.step() + decay_step().
    Captures the BCM-updated, decayed adapter state for this agent.
    """
    for name, ha in adapters.items():
        if name in state.state:
            state.state[name]["A"] = ha.lora.A.data.clone()
            state.state[name]["B"] = ha.lora.B.data.clone()
    state.n_updates += 1
