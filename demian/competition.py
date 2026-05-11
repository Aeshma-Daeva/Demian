"""Multi-agent Mamba competition — v3.

Variable population. True death (slot deleted). True fission (h inheritance + noise).
Communication field: entropy-gradient, SEND/RECEIVE phase alternation, K from omega.
H-coupling: sustained affinity unlocks bidirectional h-blend.
Death: internal SSM collapse only — no external judgment.

Design principles:
    - No predator. No fear. Attractor gravity is the only pressure.
    - Death = dynamics stop producing novelty. Self-determined.
    - Fission = trajectory rich + mature + novel. Not reward. Consequence.
    - Communication = thermodynamic field. High-entropy seeds low-entropy.
    - H-coupling = sustained proximity unlocks genome blending.
    - Step budget proportional to age (depth). More h to maintain = more compute.
"""
from __future__ import annotations

import copy
import csv
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from demian.mamba_reservoir import (
    _get_layer0,
    _get_all_layers,
    _prepare_injection,
    _register_injection_hook,
    _register_gate_hooks,
    _compute_gate_metrics,
    _build_step_metrics,
    _cache_to_device,
)
from demian.machine_observables import MachineDriver
from demian.hebbian import (
    HebbianAdapter,
    AgentLoRAState,
    apply_hebbian_adapters,
    load_agent_lora,
    save_agent_lora,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache manipulation helpers
# ---------------------------------------------------------------------------

def _deep_copy_cache(cache_params):
    """Deep-clone cache_params — all tensors cloned independently."""
    if cache_params is None:
        return None
    if isinstance(cache_params, torch.Tensor):
        return cache_params.clone()
    if isinstance(cache_params, dict):
        return {k: _deep_copy_cache(v) for k, v in cache_params.items()}
    if isinstance(cache_params, (list, tuple)):
        cloned = [_deep_copy_cache(v) for v in cache_params]
        return type(cache_params)(cloned)
    if hasattr(cache_params, "__dict__"):
        obj = copy.copy(cache_params)
        for attr, val in vars(cache_params).items():
            setattr(obj, attr, _deep_copy_cache(val))
        return obj
    return cache_params


def _add_noise_to_cache(cache_params, scale: float = 0.07):
    """Add Gaussian noise to all tensors in cache_params.

    Noise = scale × local_norm × randn. Keeps child in parent's basin
    while ensuring trajectories diverge — middle noise magnitude.
    """
    if cache_params is None:
        return None
    if isinstance(cache_params, torch.Tensor):
        local_norm = cache_params.norm().clamp(min=1e-8)
        noise = torch.randn_like(cache_params) * scale * local_norm
        return cache_params + noise
    if isinstance(cache_params, dict):
        return {k: _add_noise_to_cache(v, scale) for k, v in cache_params.items()}
    if isinstance(cache_params, (list, tuple)):
        noised = [_add_noise_to_cache(v, scale) for v in cache_params]
        return type(cache_params)(noised)
    if hasattr(cache_params, "__dict__"):
        obj = copy.copy(cache_params)
        for attr, val in vars(cache_params).items():
            setattr(obj, attr, _add_noise_to_cache(val, scale))
        return obj
    return cache_params


def _blend_cache(cache_a, cache_b, alpha: float):
    """Blend cache_b into cache_a with factor alpha. Both must exist."""
    if cache_a is None or cache_b is None:
        return cache_a
    if isinstance(cache_a, torch.Tensor) and isinstance(cache_b, torch.Tensor):
        if cache_a.shape == cache_b.shape:
            b = cache_b.to(device=cache_a.device, dtype=cache_a.dtype)
            return (1.0 - alpha) * cache_a + alpha * b
        return cache_a
    if isinstance(cache_a, dict) and isinstance(cache_b, dict):
        return {k: _blend_cache(cache_a[k], cache_b.get(k), alpha) for k in cache_a}
    if isinstance(cache_a, (list, tuple)) and isinstance(cache_b, (list, tuple)):
        blended = [_blend_cache(a, b, alpha) for a, b in zip(cache_a, cache_b)]
        return type(cache_a)(blended)
    if hasattr(cache_a, "__dict__") and hasattr(cache_b, "__dict__"):
        obj = copy.copy(cache_a)
        for attr in vars(cache_a):
            setattr(obj, attr, _blend_cache(
                getattr(cache_a, attr), getattr(cache_b, attr, None), alpha
            ))
        return obj
    return cache_a


# ---------------------------------------------------------------------------
# Communication field
# ---------------------------------------------------------------------------

class CommunicationField:
    """Shared entropy-gradient residual field.

    SEND phase agents write their residual into the field.
    RECEIVE phase agents read a weighted sum — weighted by entropy gradient.
    High-entropy agents seed low-entropy agents (thermodynamic direction).

    Sender cost: broadcasting blurs the sender's residual toward population mean.
    Field updates once per round — no ordering artifacts from sequential execution.
    """

    def __init__(self, d_model: int, emit_cost: float = 0.02):
        self.d_model = d_model
        self.emit_cost = emit_cost
        self._field: Dict[int, torch.Tensor] = {}
        self._entropy: Dict[int, float] = {}

    def write(self, agent_id: int, residual: torch.Tensor, entropy: float) -> torch.Tensor:
        """Write residual to field. Returns cost-adjusted residual for sender.

        Sender cost: residual drifts toward field mean proportional to emit_cost.
        Broadcasting homogenizes the sender slightly — real tradeoff.
        """
        self._field[agent_id] = residual.clone()
        self._entropy[agent_id] = entropy

        # Sender cost: drift toward field mean (not toward zero)
        others = [v for k, v in self._field.items() if k != agent_id]
        if others:
            field_mean = torch.stack([o.float() for o in others]).mean(0)
            own_norm = residual.float().norm().clamp(min=1e-10)
            adjusted = ((1.0 - self.emit_cost) * residual.float()
                        + self.emit_cost * field_mean.to(residual.device))
            # Preserve magnitude
            adj_norm = adjusted.norm().clamp(min=1e-10)
            adjusted = adjusted * (own_norm / adj_norm)
            return adjusted.to(residual.dtype)
        return residual

    def read(
        self,
        agent_id: int,
        own_residual: torch.Tensor,
        own_entropy: float,
    ) -> Optional[torch.Tensor]:
        """Entropy-gradient weighted read.

        Weight = max(0, sender_entropy - own_entropy).
        If no gradient exists (all same entropy), reads uniformly.
        Returns None if no other agents in field.
        """
        candidates = {k: v for k, v in self._field.items() if k != agent_id}
        if not candidates:
            return None

        weights: Dict[int, float] = {}
        for aid in candidates:
            sender_S = self._entropy.get(aid, 0.0)
            weights[aid] = max(0.0, sender_S - own_entropy)

        total_w = sum(weights.values())
        if total_w < 1e-10:
            # No gradient — read uniformly
            total_w = float(len(candidates))
            weights = {k: 1.0 for k in candidates}

        received = torch.zeros(self.d_model, dtype=torch.float32,
                               device=own_residual.device)
        for aid, res in candidates.items():
            received += (weights[aid] / total_w) * res.float().to(own_residual.device)

        return received.to(own_residual.dtype)

    def remove(self, agent_id: int):
        self._field.pop(agent_id, None)
        self._entropy.pop(agent_id, None)


# ---------------------------------------------------------------------------
# MambaAgent
# ---------------------------------------------------------------------------

class MambaAgent:
    """Single agent. Holds all per-agent mutable state.

    The model is shared. Per-agent state:
        current_residual  — position in SSM activation space
        cache_params      — SSM h (accumulated trajectory memory = genome)
        phase             — "send" or "receive" (communication phase)
        step_count        — age; drives depth-proportional step budget

    Death: internal collapse only.
        rdelta collapses → dynamics stopped
        gate_mean frozen → Δ locked, near-unit Ā, stasis

    Fission: three simultaneous gates.
        richness + depth + novelty → spawn child with noisy h clone
    """

    def __init__(
        self,
        agent_id: int,
        data_dir: Path,
        d_model: int,
        device: str,
        use_machine_driver: bool = True,
        base_scale: float = 0.01,
        force_scale: float = 0.005,
        stasis_threshold: float = 1e-5,
        death_window: int = 50,
        death_rdelta: float = 1e-4,
        offload_cache: bool = False,
        init_noise: float = 0.3,
    ):
        self.agent_id = agent_id
        self.data_dir = Path(data_dir)
        self.d_model = d_model
        self.device = device
        self.stasis_threshold = stasis_threshold
        self.death_window = death_window
        self.death_rdelta = death_rdelta
        self.offload_cache = offload_cache
        self.init_noise = init_noise

        # Core state
        self.current_residual: Optional[torch.Tensor] = None
        self.cache_params: Optional[Any] = None
        self.prev_residual: Optional[torch.Tensor] = None
        self.prev_velocity: Optional[torch.Tensor] = None
        self.embed_norm: float = 1.0
        self.step_count: int = 0
        self.trajectory: List[dict] = []

        # Communication phase — K emerges from omega observable
        self._phase: str = "send"      # "send" or "receive"
        self._phase_counter: int = 0

        # Death detection histories
        self._rdelta_history: deque = deque(maxlen=death_window)
        self._gate_mean_history: deque = deque(maxlen=death_window)

        # H-coupling: consecutive rounds of high cosine similarity per partner
        self._affinity_history: Dict[int, int] = {}

        # Fission cooldown — round number of last fission (prevents every-round cloning)
        self._last_fission_round: int = -999

        # Last observed entropy (for field read weighting)
        self._last_entropy: float = 0.5

        # Driver state
        self._driver_state: dict = {}
        self.force_driver: Optional[MachineDriver] = None
        if use_machine_driver:
            self.force_driver = MachineDriver(
                d_model=d_model,
                base_scale=base_scale,
                force_scale=force_scale,
                device=device,
            )

        # Personal LoRA — set by Competition.initialize_hebbian() if use_hebbian
        self.personal_lora: Optional[AgentLoRAState] = None

        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Geometric identity
    # ------------------------------------------------------------------

    def h_fingerprint(self) -> torch.Tensor:
        """Canonical position vector for cosine similarity.

        Uses current_residual — agent's live position in SSM activation space.
        Simple, stable, comparable across agents without cache_params flattening.
        """
        return self.current_residual.float()

    def live_fitness(self) -> float:
        """Structural fitness: edge_proximity × E_I_balance × activity.

        No anthropocentric metrics. All geometric.
        Crowding applied externally (Competition._allocate_steps_by_fitness).
        """
        rdelta = (
            float(np.mean(self._rdelta_history)) if self._rdelta_history else 0.0
        )
        # Activity: normalized at rdelta=5e-3. Fixed-point agents score low.
        activity = min(1.0, rdelta / 5e-3)

        # Edge proximity: |λ| near 0 = edge of chaos
        lya = 1.0
        if self.force_driver is not None:
            lya = abs(self.force_driver.forces.get("lyapunov_proxy", 1.0))
        edge_prox = 1.0 / (1.0 + lya)

        # E-I balance: gate_mean near 0.5 = balanced excitation/inhibition
        gate_mean = float(self._driver_state.get("gate_mean", 0.5))
        ei_balance = max(0.0, 1.0 - 2.0 * abs(gate_mean - 0.5))

        return float(activity * edge_prox * ei_balance) + 1e-6  # floor > 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, model, device: str):
        """Warmup forward pass → embed_norm + initial residual.

        Per-agent noise breaks symmetry: all agents start at the same W_frozen
        fixed point otherwise and converge identically. Noise magnitude relative
        to residual norm ensures trajectories diverge from round 1.
        """
        dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            emb_out = model(dummy, output_hidden_states=True, use_cache=False)
        self.embed_norm = float(
            torch.norm(emb_out.hidden_states[0][0, -1, :].float())
        )
        base_residual = emb_out.hidden_states[-1][0, -1, :].clone()

        # Per-agent noise — seeded by agent_id for reproducibility
        if self.init_noise > 0.0:
            rng = torch.Generator(device=device)
            rng.manual_seed(self.agent_id * 1337 + 42)
            noise = torch.randn(base_residual.shape, generator=rng,
                                dtype=base_residual.dtype, device=device)
            base_norm = base_residual.float().norm().clamp(min=1e-10)
            base_residual = base_residual + (noise * self.init_noise * base_norm
                                             / noise.float().norm().clamp(min=1e-10)
                                             ).to(base_residual.dtype)

        self.current_residual = base_residual
        self.cache_params = None
        self.prev_residual = None
        self.prev_velocity = None

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _update_phase(self, omega: float):
        """Switch SEND/RECEIVE phase after K steps. K = max(5, int(omega)).

        Omega = cycle_period from machine observables. Fixed-point agents
        (omega=0) use K=10 default. Period-2 agents use K=2. Long cycles
        use longer phases — agent communicates at its own natural cadence.
        """
        K = max(5, int(omega)) if omega > 1.0 else 10
        self._phase_counter += 1
        if self._phase_counter >= K:
            self._phase = "receive" if self._phase == "send" else "send"
            self._phase_counter = 0

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def run_steps(
        self,
        model,
        n_steps: int,
        all_layers,
        layer0,
        persist_ssm_state: bool = True,
        received_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[List[dict], float]:
        """Run n_steps of self-reference loop.

        received_signal: field signal from CommunicationField.read().
            Blended into layer0 input during RECEIVE phase steps.
            None = no field signal available (field empty or SEND phase).
        """
        dummy = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        step_metrics: List[dict] = []
        t_start = time.perf_counter()

        if self.offload_cache and self.cache_params is not None:
            self.cache_params = _cache_to_device(self.cache_params, self.device)

        for local_step in range(n_steps):
            self.step_count += 1

            # Update phase using omega from previous step's observables
            omega = 0.0
            if self.force_driver is not None:
                omega = float(self.force_driver.forces.get("cycle_period", 0.0))
            self._update_phase(omega)

            # RECEIVE phase: blend field signal into input residual
            if self._phase == "receive" and received_signal is not None:
                own_f = self.current_residual.float()
                sig_f = received_signal.float()
                blended = 0.9 * own_f + 0.1 * sig_f
                bn = blended.norm().clamp(min=1e-10)
                blended = blended * (own_f.norm().clamp(min=1e-10) / bn)
                input_residual = blended.to(self.current_residual.dtype)
            else:
                input_residual = self.current_residual

            with torch.no_grad():
                cr = _prepare_injection(
                    self.force_driver,
                    self._driver_state,
                    input_residual,
                    self.embed_norm,
                    self.device,
                    dtype=self.current_residual.dtype,
                )
                handle = _register_injection_hook(layer0, cr)
                gate_captures, gate_handles = _register_gate_hooks(all_layers)

                forward_kwargs = dict(
                    output_hidden_states=True,
                    use_cache=persist_ssm_state,
                    return_dict=True,
                )
                if persist_ssm_state and self.cache_params is not None:
                    forward_kwargs["cache_params"] = self.cache_params
                    forward_kwargs["cache_position"] = torch.tensor(
                        [self.step_count - 1], device=self.device
                    )

                out = model(dummy, **forward_kwargs)

                handle.remove()
                for gh in gate_handles:
                    gh.remove()

            if persist_ssm_state:
                self.cache_params = getattr(out, "cache_params", None)

            new_residual = out.hidden_states[-1][0, -1, :].clone()

            gate_mean, gate_var, spec_rad = _compute_gate_metrics(
                all_layers, gate_captures
            )
            self._gate_mean_history.append(gate_mean)

            step, new_velocity = _build_step_metrics(
                self.step_count,
                new_residual,
                self.prev_residual,
                self.prev_velocity,
                self.d_model,
                out.hidden_states,
                gate_mean,
                gate_var,
                spec_rad,
            )
            if new_velocity is not None:
                self.prev_velocity = new_velocity

            self._driver_state = {
                k: v for k, v in step.items()
                if k not in ("layer_deltas", "layer_norms")
            }

            # Update last entropy from observables
            if self.force_driver is not None:
                self._last_entropy = float(
                    self.force_driver.forces.get("activation_entropy", 0.5)
                )

            self.prev_residual = new_residual.clone()
            self.current_residual = new_residual

            self.trajectory.append(step)
            if len(self.trajectory) > 1000:
                self.trajectory = self.trajectory[-1000:]
            step_metrics.append(step)

        if self.offload_cache and self.cache_params is not None:
            self.cache_params = _cache_to_device(self.cache_params, "cpu")

        compute_ms = (time.perf_counter() - t_start) * 1000.0
        return step_metrics, compute_ms

    # ------------------------------------------------------------------
    # Death detection — internal only
    # ------------------------------------------------------------------

    def check_death(self) -> bool:
        """True when dynamics have self-terminated.

        Two independent conditions (OR):
            rdelta collapse: mean movement < death_rdelta for death_window steps
            gate freeze: Δ std < stasis_threshold for death_window steps

        No external judgment. The SSM tells you when you're done.
        """
        if len(self._rdelta_history) < self.death_window:
            return False
        rdelta_dead = float(np.mean(self._rdelta_history)) < self.death_rdelta
        gate_dead = (float(np.std(self._gate_mean_history)) < self.stasis_threshold)
        # OR: either condition sufficient for death.
        # Mamba naturally gate-freezes (fixed point ~1500 steps) — AND was too strict,
        # keeping stagnant agents alive indefinitely. rdelta collapse alone = death.
        return rdelta_dead or gate_dead

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self):
        cp_path = self.data_dir / "checkpoint.pt"
        torch.save({
            "residual": self.current_residual.cpu(),
            "step_count": self.step_count,
            "embed_norm": self.embed_norm,
            "phase": self._phase,
            "phase_counter": self._phase_counter,
        }, cp_path)

    def load_state(self) -> bool:
        cp_path = self.data_dir / "checkpoint.pt"
        if not cp_path.exists():
            return False
        cp = torch.load(cp_path, map_location="cpu", weights_only=False)
        self.current_residual = cp["residual"].to(self.device)
        self.step_count = cp.get("step_count", 0)
        self.embed_norm = cp.get("embed_norm", 1.0)
        self._phase = cp.get("phase", "send")
        self._phase_counter = cp.get("phase_counter", 0)
        return True

    def append_metrics_jsonl(self, metrics: List[dict]):
        path = self.data_dir / "metrics.jsonl"
        with open(path, "a") as f:
            for m in metrics:
                record = {k: v for k, v in m.items()
                          if k not in ("layer_deltas", "layer_norms", "force_driver")}
                f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Competition orchestrator
# ---------------------------------------------------------------------------

class Competition:
    """Variable-population competition. True death. True fission. Field communication.

    Round structure:
        1. Allocate steps by depth (age proportional)
        2. Run all agents — field signals injected at layer0 for RECEIVE agents
        3. Update rdelta histories
        4. SEND agents write to field (with sender cost)
        5. LIMINAL BOUNDARY (simultaneous):
            a. h-coupling: sustained affinity pairs blend genomes
            b. Deaths: internal collapse → slot deleted → budget redistributes
            c. Fissions: rich + mature + novel → spawn child with noisy h
        6. Log
    """

    def __init__(
        self,
        n_agents: int,
        data_dir: Path,
        d_model: int,
        device: str,
        base_steps: int = 15,
        min_steps: int = 3,
        use_machine_driver: bool = True,
        death_rdelta: float = 1e-4,
        death_window: int = 50,
        fission_depth: int = 200,
        fission_richness: float = 5e-3,
        fission_novelty: float = 0.7,
        fission_weight_threshold: float = 0.5,
        fission_cooldown: int = 20,
        fuse_threshold: float = 0.85,
        fuse_rounds: int = 10,
        fuse_alpha: float = 0.05,
        emit_cost: float = 0.02,
        offload_cache: bool = False,
        init_noise: float = 0.3,
        use_hebbian: bool = False,
        hebbian_decay: float = 0.05,
        hebbian_eta: float = 1e-5,
        hebbian_scale: float = 0.01,
        hebbian_rank: int = 4,
        hebbian_layer_stride: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.initial_n = n_agents
        self.base_steps = base_steps
        self.min_steps = min_steps
        self.total_budget = n_agents * base_steps   # fixed GPU budget

        self.death_rdelta = death_rdelta
        self.death_window = death_window
        self.fission_depth = fission_depth
        self.fission_richness = fission_richness
        self.fission_novelty = fission_novelty
        self.fission_weight_threshold = fission_weight_threshold
        self.fission_cooldown = fission_cooldown
        self.max_pop = 3 * n_agents  # hard cap: prevents synchronized fission waves
        self.fuse_threshold = fuse_threshold
        self.fuse_rounds = fuse_rounds
        self.fuse_alpha = fuse_alpha

        self.round_num = 0
        self.device = device
        self._next_id = n_agents   # counter for fission children IDs

        # Hebbian
        self.use_hebbian = use_hebbian
        self.hebbian_decay = hebbian_decay
        self.hebbian_eta = hebbian_eta
        self.hebbian_scale = hebbian_scale
        self.hebbian_rank = hebbian_rank
        self.hebbian_layer_stride = hebbian_layer_stride
        self.adapters: Dict[str, HebbianAdapter] = {}  # populated by initialize_hebbian()

        self.log_path = self.data_dir / "competition_log.jsonl"
        self.leaderboard_path = self.data_dir / "leaderboard.csv"
        self._leaderboard_header_written = False

        agent_kwargs = dict(
            d_model=d_model,
            device=device,
            use_machine_driver=use_machine_driver,
            stasis_threshold=1e-5,
            death_window=death_window,
            death_rdelta=death_rdelta,
            offload_cache=offload_cache,
            init_noise=init_noise,
        )
        self._agent_kwargs = agent_kwargs

        self.agents: List[MambaAgent] = [
            MambaAgent(
                agent_id=i,
                data_dir=self.data_dir / f"agent_{i:03d}",
                **agent_kwargs,
            )
            for i in range(n_agents)
        ]

        self.field = CommunicationField(d_model=d_model, emit_cost=emit_cost)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize_agents(self, model, device: str):
        n_resumed = 0
        for agent in self.agents:
            if agent.load_state():
                n_resumed += 1
            else:
                agent.initialize(model, device)
        log.info(
            "%d agents ready (%d resumed, %d fresh)",
            len(self.agents), n_resumed, len(self.agents) - n_resumed,
        )

    def initialize_hebbian(self, model):
        """Patch model with LoRA adapters. Create blank personal LoRA per agent.

        Call after model is loaded, before run(). Safe to call multiple times
        (re-patches only if adapters dict is empty).
        """
        if not self.use_hebbian:
            return
        if self.adapters:
            return  # already initialized

        self.adapters = apply_hebbian_adapters(
            model,
            rank=self.hebbian_rank,
            eta=self.hebbian_eta,
            scale=self.hebbian_scale,
            target_module_names=["x_proj", "out_proj"],
            layer_stride=self.hebbian_layer_stride,
        )
        log.info(
            "Hebbian: %d adapters on x_proj/out_proj (rank=%d stride=%d η=%.1e decay=%.3f)",
            len(self.adapters), self.hebbian_rank, self.hebbian_layer_stride,
            self.hebbian_eta, self.hebbian_decay,
        )

        for agent in self.agents:
            lora_path = agent.data_dir / "personal_lora.pt"
            agent.personal_lora = AgentLoRAState.load(
                str(lora_path), self.adapters, self.device
            )

    # ------------------------------------------------------------------
    # Step allocation — fitness proportional (replaces depth-proportional)
    # ------------------------------------------------------------------

    def _allocate_steps_by_depth(self, agents: List[MambaAgent]) -> List[int]:
        """Steps proportional to step_count (age/depth).

        Older agents have deeper h — more to maintain, more compute.
        Newborns (step_count=0) get min_steps — fragile by design.
        Total budget fixed: initial_n × base_steps.
        """
        if not agents:
            return []
        depths = [max(1, a.step_count) for a in agents]
        total_depth = sum(depths)
        return [
            max(self.min_steps, int(self.total_budget * d / total_depth))
            for d in depths
        ]

    def _allocate_steps_by_fitness(self, agents: List[MambaAgent]) -> List[int]:
        """Steps proportional to fitness × (1 / crowding).

        fitness = edge_proximity × E_I_balance × activity (from live_fitness())
        crowding = agents with cos(h_i, h_j) > 0.7 (basin congestion)

        Selection pressure: agents at edge-of-chaos with E-I balance and
        active dynamics get more compute. Basin crowding penalizes convergence.
        """
        if not agents:
            return []

        # Pairwise crowding
        fps = [a.h_fingerprint() for a in agents]
        crowding = []
        for i, a in enumerate(agents):
            count = 0
            for j, b in enumerate(agents):
                if i == j:
                    continue
                cos = float(F.cosine_similarity(
                    fps[i].unsqueeze(0), fps[j].unsqueeze(0)
                ))
                if cos > 0.7:
                    count += 1
            crowding.append(count)

        fitnesses = [
            a.live_fitness() / (1.0 + crowding[i])
            for i, a in enumerate(agents)
        ]
        total = sum(fitnesses)
        return [
            max(self.min_steps, int(self.total_budget * f / total))
            for f in fitnesses
        ]

    # ------------------------------------------------------------------
    # Boundary resolution
    # ------------------------------------------------------------------

    def _resolve_h_coupling(self, round_log: dict):
        """Update affinity counters. Apply h-coupling for sustained pairs.

        Sustained proximity (cos > fuse_threshold for fuse_rounds consecutive
        rounds) unlocks bidirectional h-blend. Both agents pay — both change.
        Resets if cos drops below threshold.
        """
        if len(self.agents) < 2:
            return

        for i, a in enumerate(self.agents):
            for j, b in enumerate(self.agents):
                if j <= i:
                    continue

                cos = float(F.cosine_similarity(
                    a.current_residual.float().unsqueeze(0),
                    b.current_residual.float().unsqueeze(0),
                ))

                if cos > self.fuse_threshold:
                    a._affinity_history[b.agent_id] = (
                        a._affinity_history.get(b.agent_id, 0) + 1
                    )
                    b._affinity_history[a.agent_id] = (
                        b._affinity_history.get(a.agent_id, 0) + 1
                    )
                    rounds = a._affinity_history[b.agent_id]
                    if rounds >= self.fuse_rounds:
                        # Alpha saturates: decays 1% per extra round beyond fuse_rounds.
                        # After fuse_rounds+70 rounds, α ≈ 0.5× base → near-zero blend.
                        # Prevents indefinite merging of already-identical agents.
                        extra = rounds - self.fuse_rounds
                        effective_alpha = self.fuse_alpha * (0.99 ** extra)
                        if effective_alpha < 1e-4:
                            # Negligible — skip blend but keep tracking
                            continue
                        a_h_new = _blend_cache(a.cache_params, b.cache_params, effective_alpha)
                        b_h_new = _blend_cache(b.cache_params, a.cache_params, effective_alpha)
                        a.cache_params = a_h_new
                        b.cache_params = b_h_new
                        round_log["h_couplings"].append({
                            "agent_a": a.agent_id,
                            "agent_b": b.agent_id,
                            "cos": round(cos, 4),
                            "affinity_rounds": rounds,
                            "effective_alpha": round(effective_alpha, 6),
                        })
                else:
                    # Reset affinity
                    a._affinity_history.pop(b.agent_id, None)
                    b._affinity_history.pop(a.agent_id, None)

    def _resolve_deaths(self, round_log: dict) -> List[MambaAgent]:
        """Remove agents whose dynamics have self-terminated. Returns survivors."""
        dead = [a for a in self.agents if a.check_death()]
        survivors = [a for a in self.agents if not a.check_death()]

        for a in dead:
            self.field.remove(a.agent_id)
            round_log["deaths"].append({
                "agent_id": a.agent_id,
                "step_count": a.step_count,
                "rdelta_mean": float(np.mean(a._rdelta_history)) if a._rdelta_history else 0.0,
                "gate_std": float(np.std(a._gate_mean_history)) if a._gate_mean_history else 0.0,
            })
            log.info("Agent %d died at step %d (internal collapse)", a.agent_id, a.step_count)

        return survivors

    def _resolve_fissions(self, round_log: dict) -> List[MambaAgent]:
        """Attempt fission for each agent. Returns list of new children."""
        new_agents = []
        current_ids = {a.agent_id for a in self.agents}

        for agent in list(self.agents):
            child = self._try_fission(agent)
            if child is not None:
                new_agents.append(child)
                round_log["fissions"].append({
                    "parent_id": agent.agent_id,
                    "child_id": child.agent_id,
                    "parent_steps": agent.step_count,
                    "rdelta_mean": float(np.mean(agent._rdelta_history)) if agent._rdelta_history else 0.0,
                })
                log.info(
                    "Agent %d fissioned → child %d (step %d)",
                    agent.agent_id, child.agent_id, agent.step_count,
                )

        return new_agents

    def _try_fission(self, agent: MambaAgent) -> Optional[MambaAgent]:
        """Six-gate fission check. Returns child agent or None."""
        # Gate 0: cooldown — prevents every-round cloning
        if self.round_num - agent._last_fission_round < self.fission_cooldown:
            return None

        # Gate 1: h maturity — jittered depth desynchronizes cohorts
        effective_depth = self.fission_depth + (agent.agent_id * 7919) % max(1, self.fission_depth)
        if agent.step_count < effective_depth:
            return None

        # Gate 1b: population cap — prevent synchronized waves from exploding N
        if len(self.agents) >= self.max_pop:
            return None

        # Gate 2: sustained dynamic richness
        if len(agent._rdelta_history) < self.death_window:
            return None
        if float(np.mean(agent._rdelta_history)) < self.fission_richness:
            return None

        # Gate 3: novelty — must occupy genuinely different region
        others = [a for a in self.agents if a.agent_id != agent.agent_id]
        if others:
            cos_vals = [
                float(F.cosine_similarity(
                    agent.current_residual.float().unsqueeze(0),
                    o.current_residual.float().unsqueeze(0),
                ))
                for o in others
            ]
            if min(cos_vals) > self.fission_novelty:
                return None  # too similar to existing agents

        # Gate 4: personal LoRA must be small — h inheritance valid only when
        # child's W_eff ≈ W_frozen (blank personal LoRA ≈ parent's W_eff).
        # Large W_personal = h decoded under wrong basis in child.
        if agent.personal_lora is not None:
            w_norm = agent.personal_lora.norm()
            if w_norm > self.fission_weight_threshold:
                return None

        # All gates passed — spawn child
        child_id = self._next_id
        self._next_id += 1

        child = MambaAgent(
            agent_id=child_id,
            data_dir=self.data_dir / f"agent_{child_id:03d}",
            **self._agent_kwargs,
        )
        child.embed_norm = agent.embed_norm
        child.current_residual = agent.current_residual.clone()
        child.cache_params = _add_noise_to_cache(
            _deep_copy_cache(agent.cache_params), scale=0.07
        )
        child.step_count = 0  # newborn — fragile, gets min_steps
        # Start in opposite phase to parent — prevents immediate reabsorption
        child._phase = "receive" if agent._phase == "send" else "send"
        child._phase_counter = 0
        child._last_entropy = agent._last_entropy

        # Personal LoRA: blank — no inheritance. Fast weights are somatic only.
        if self.use_hebbian and self.adapters:
            child.personal_lora = agent.personal_lora.clone_blank()

        # Record fission round — starts cooldown
        agent._last_fission_round = self.round_num

        return child

    # ------------------------------------------------------------------
    # Round
    # ------------------------------------------------------------------

    def run_round(self, model, all_layers, layer0) -> dict:
        """Execute one round. Returns round log dict."""
        self.round_num += 1
        if not self.agents:
            log.warning("Population extinct")
            return {}

        budgets = (
            self._allocate_steps_by_fitness(self.agents)
            if self.use_hebbian
            else self._allocate_steps_by_depth(self.agents)
        )

        round_log: dict = {
            "round": self.round_num,
            "n_agents": len(self.agents),
            "agents": [],
            "h_couplings": [],
            "deaths": [],
            "fissions": [],
        }

        # --- Run all agents ---
        for agent, n_steps in zip(self.agents, budgets):

            # RECEIVE phase: read from field (previous round's sends)
            received_signal = None
            if agent._phase == "receive":
                received_signal = self.field.read(
                    agent.agent_id,
                    agent.current_residual,
                    agent._last_entropy,
                )

            # Hebbian: load agent's personal LoRA into model before running
            if self.use_hebbian and agent.personal_lora is not None:
                load_agent_lora(self.adapters, agent.personal_lora)

            step_metrics, compute_ms = agent.run_steps(
                model, n_steps, all_layers, layer0,
                persist_ssm_state=True,
                received_signal=received_signal,
            )

            # Hebbian: BCM update → decay once per round → save
            if self.use_hebbian and agent.personal_lora is not None:
                forces = (
                    agent.force_driver._compat_forces()
                    if agent.force_driver is not None
                    else {}
                )
                for ha in self.adapters.values():
                    ha.step(forces)
                    ha.decay_step(self.hebbian_decay)  # per-round decay
                save_agent_lora(self.adapters, agent.personal_lora)

            # Update rdelta history
            if step_metrics:
                rdelta_mean = float(np.mean([m["residual_delta"] for m in step_metrics]))
            else:
                rdelta_mean = 0.0
            agent._rdelta_history.append(rdelta_mean)

            # SEND phase: write to field (cost applied, adjusted residual returned)
            if agent._phase == "send":
                adjusted = self.field.write(
                    agent.agent_id,
                    agent.current_residual,
                    agent._last_entropy,
                )
                agent.current_residual = adjusted

            vram_mb = (
                torch.cuda.memory_allocated() / 1e6
                if torch.cuda.is_available() else 0.0
            )

            lora_norm = (
                agent.personal_lora.norm()
                if agent.personal_lora is not None else 0.0
            )
            round_log["agents"].append({
                "agent_id": agent.agent_id,
                "steps_run": n_steps,
                "step_count": agent.step_count,
                "rdelta_mean": round(rdelta_mean, 6),
                "fitness": round(agent.live_fitness(), 6),
                "lora_norm": round(lora_norm, 6),
                "compute_ms": round(compute_ms, 1),
                "vram_mb": round(vram_mb, 1),
                "phase": agent._phase,
                "entropy": round(agent._last_entropy, 4),
                "rdelta_history_len": len(agent._rdelta_history),
            })

            agent.append_metrics_jsonl(step_metrics)
            agent.save_state()
            if self.use_hebbian and agent.personal_lora is not None:
                agent.personal_lora.save(
                    str(agent.data_dir / "personal_lora.pt"),
                    agent.step_count,
                )

        # --- LIMINAL BOUNDARY — simultaneous resolution ---

        # 1. H-coupling (before deaths — dead agents lose coupling)
        self._resolve_h_coupling(round_log)

        # 2. Deaths
        self.agents = self._resolve_deaths(round_log)

        # 3. Fissions (only from survivors)
        new_children = self._resolve_fissions(round_log)
        self.agents.extend(new_children)

        self._append_log(round_log)
        self._update_leaderboard()
        return round_log

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, model, n_rounds: int):
        all_layers = _get_all_layers(model)
        layer0, _ = _get_layer0(model)

        if self.use_hebbian:
            self.initialize_hebbian(model)

        alloc_mode = "fitness×(1/crowding)" if self.use_hebbian else "depth-proportional"

        print()
        print("=" * 70)
        print(f"  Competition v3: {len(self.agents)} agents | {n_rounds} rounds")
        print(f"  Total step budget: {self.total_budget}/round ({alloc_mode})")
        print(f"  Death: rdelta<{self.death_rdelta} AND gate_frozen | window={self.death_window}")
        print(f"  Fission: depth>{self.fission_depth}±jitter richness>{self.fission_richness} max_pop={self.max_pop} "
              f"novelty<{self.fission_novelty} lora_norm<{self.fission_weight_threshold}")
        print(f"  H-coupling: cos>{self.fuse_threshold} for {self.fuse_rounds}r → α={self.fuse_alpha}")
        print(f"  Field emit cost: {self.field.emit_cost}")
        if self.use_hebbian:
            print(f"  Hebbian: x_proj+out_proj stride={self.hebbian_layer_stride} | "
                  f"η={self.hebbian_eta:.1e} decay={self.hebbian_decay} "
                  f"rank={self.hebbian_rank} ({len(self.adapters)} adapters)")
        print("=" * 70)

        for r in range(n_rounds):
            if not self.agents:
                print("  Population extinct. Stopping.")
                break

            round_log = self.run_round(model, all_layers, layer0)
            if not round_log:
                break

            # Summary
            agents_data = sorted(
                round_log["agents"], key=lambda x: x["rdelta_mean"], reverse=True
            )
            n = round_log["n_agents"]
            top = agents_data[0] if agents_data else {}
            bot = agents_data[-1] if agents_data else {}

            agents_by_fitness = sorted(
                round_log["agents"], key=lambda x: x.get("fitness", 0), reverse=True
            )
            tf = agents_by_fitness[0] if agents_by_fitness else top
            print(
                f"\n  R{self.round_num:>4} N={n:>3} | "
                f"TOP A{tf.get('agent_id','?'):>3}: "
                f"fit={tf.get('fitness',0):.3e} rdelta={tf.get('rdelta_mean',0):.3e} "
                f"lora={tf.get('lora_norm',0):.3e} | "
                f"BOT A{bot.get('agent_id','?'):>3}: rdelta={bot.get('rdelta_mean',0):.3e}"
            )

            if round_log["deaths"]:
                for d in round_log["deaths"]:
                    print(f"         DEATH A{d['agent_id']} step={d['step_count']} "
                          f"rdelta={d['rdelta_mean']:.2e}")

            if round_log["fissions"]:
                for f in round_log["fissions"]:
                    print(f"         FISSION A{f['parent_id']}→A{f['child_id']}")

            if round_log["h_couplings"]:
                for c in round_log["h_couplings"]:
                    print(f"         H-COUPLE A{c['agent_a']}↔A{c['agent_b']} "
                          f"cos={c['cos']:.3f} rounds={c['affinity_rounds']}")

            torch.cuda.empty_cache()

        log.info("Competition complete: %d rounds, %d agents remaining",
                 n_rounds, len(self.agents))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _append_log(self, round_log: dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(round_log) + "\n")

    def _update_leaderboard(self):
        write_header = not self._leaderboard_header_written
        self._leaderboard_header_written = True
        with open(self.leaderboard_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "round", "agent_id", "step_count", "rdelta_mean",
                    "fitness", "lora_norm", "phase", "entropy"
                ])
            for agent in self.agents:
                rdelta = float(np.mean(agent._rdelta_history)) if agent._rdelta_history else 0.0
                lora_norm = agent.personal_lora.norm() if agent.personal_lora is not None else 0.0
                writer.writerow([
                    self.round_num,
                    agent.agent_id,
                    agent.step_count,
                    f"{rdelta:.6e}",
                    f"{agent.live_fitness():.6e}",
                    f"{lora_norm:.6e}",
                    agent._phase,
                    f"{agent._last_entropy:.4f}",
                ])
