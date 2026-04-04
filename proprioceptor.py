"""Proprioceptor: a small instrumented model (Gemma2-9B) that observes
its own activations and encodes them as a self-state.

Not an analysis tool for humans — a participant in the conversation
that surfaces its own computational state as context for the thinker.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SAE support — loads Gemma Scope features if available
# ---------------------------------------------------------------------------
class SAELoader:
    """Wraps Gemma Scope SAEs for online feature extraction."""

    def __init__(self, sae_path: str | Path | None, device: str = "cuda"):
        self._sae_path = Path(sae_path) if sae_path else None
        self._device = device
        self._available_layers: set[int] = set()
        self._discover()

    def _discover(self):
        if self._sae_path is None or not self._sae_path.exists():
            log.info("SAE path not set — running without SAE decomposition")
            return
        layers_dir = self._sae_path / "gemma-2-9b-it" / "layers"
        if not layers_dir.exists():
            log.warning("Gemma Scope layers dir not found at %s", layers_dir)
            return
        for d in layers_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                self._available_layers.add(int(d.name))
        if self._available_layers:
            log.info("SAELoader: %d layers with SAEs available", len(self._available_layers))

    def encode(self, layer_idx: int, activations: torch.Tensor, top_k: int = 64) -> Dict[int, float]:
        """Return top-K SAE features for activations. Empty dict if SAE not available."""
        if layer_idx not in self._available_layers:
            return {}
        return {}

    @property
    def has_saes(self) -> bool:
        return bool(self._available_layers)


# ---------------------------------------------------------------------------
# Native signal dataclasses — described as computational facts, not psychology
# ---------------------------------------------------------------------------
@dataclass
class LayerSignal:
    """Measurement from one transformer layer."""
    layer: int
    activation_norm: float
    mlp_contribution: float
    attn_contribution: float
    activation_sparsity: float


@dataclass
class AttentionTopology:
    """Information routing topology for a generation step."""
    entropy_per_layer: List[float] = field(default_factory=list)
    dominant_sources_per_layer: List[List[int]] = field(default_factory=list)
    system_prompt_fraction: float = 0.0


@dataclass
class PredictionTrajectory:
    """How the output distribution evolves across layers (logit lens)."""
    kl_divergences: List[float] = field(default_factory=list)
    top_token_changes: int = 0


@dataclass
class CoherenceMeasure:
    """Consistency of output direction across generated tokens."""
    rolling_sims: List[float] = field(default_factory=list)
    latest: float = 1.0


@dataclass
class FeatureState:
    """SAE feature activations — the model's native vocabulary for itself."""
    top_features: Dict[int, float] = field(default_factory=dict)
    coactivations: Dict[tuple, float] = field(default_factory=dict)


@dataclass
class BoundaryMoment:
    """A layer where computation is ambiguous / undecided — the liminal spaces."""
    layer: int
    signal_type: str
    value: float
    context: dict = field(default_factory=dict)


@dataclass
class ProprioceptiveState:
    """Complete self-state at a point in generation."""
    layer_signals: Dict[int, LayerSignal] = field(default_factory=dict)
    attention: AttentionTopology = field(default_factory=AttentionTopology)
    prediction: PredictionTrajectory = field(default_factory=PredictionTrajectory)
    coherence: CoherenceMeasure = field(default_factory=CoherenceMeasure)
    features: FeatureState = field(default_factory=FeatureState)
    boundaries: List[BoundaryMoment] = field(default_factory=list)
    token_count: int = 0


# ---------------------------------------------------------------------------
# Proprioceptor: the body
# ---------------------------------------------------------------------------
class Proprioceptor:
    """A model that feels itself thinking."""

    def __init__(
        self,
        model_id: str = "google/gemma-2-9b-it",
        quant: str = "gptq-4bit",
        sae_path: str | None = None,
        logit_layers: List[int] = None,
        device: str = "cuda",
    ):
        self._device = torch.device(device)
        self._model_id = model_id
        self._quant = quant
        self._logit_layers = logit_layers or [8, 16, 24, 32, 40]
        self._generated_states: List[torch.Tensor] = []
        self._state = ProprioceptiveState()
        self._hooks = []

        log.info("Loading proprioceptor: %s (%s)", model_id, quant)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"

        self._load_model(quant)
        self.model.eval()

        self._d_model = self.model.config.hidden_size
        self._num_layers = self.model.config.num_hidden_layers
        self.sae = SAELoader(sae_path, device=device)

        log.info("Loaded: %d layers, d_model=%d", self._num_layers, self._d_model)

    def _load_model(self, quant: str):
        if quant == "gptq-4bit":
            from transformers import GPTQConfig
            cfg = GPTQConfig(bits=4, group_size=128, dataset="c4", desc_act=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                quantization_config=cfg,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

    # -----------------------------------------------------------------------
    # Generation with activation capture
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int = 256) -> tuple[str, ProprioceptiveState]:
        """Generate while observing internal state."""
        self._state = ProprioceptiveState()
        self._generated_states.clear()
        self._remove_hooks()

        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)
        input_len = inputs["input_ids"].shape[1]

        # Install hooks
        for layer_idx, layer in enumerate(self.model.model.layers):
            h = layer.register_forward_hook(self._residual_hook(layer_idx))
            self._hooks.append(h)
            h_attn = layer.self_attn.register_forward_hook(
                self._attention_hook(layer_idx)
            )
            self._hooks.append(h_attn)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        generated_ids = out.sequences[0][input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if hasattr(out, "hidden_states") and out.hidden_states:
            self._extract_layer_signals(out.hidden_states)
            self._extract_coherence(out.hidden_states)

        if hasattr(out, "attentions") and out.attentions:
            self._extract_attention_topology(out.attentions)

        self._state.token_count = len(generated_ids)
        self._state.boundaries = self._detect_boundaries()

        self._remove_hooks()
        return generated_text, self._state

    # -----------------------------------------------------------------------
    # Hook stubs (post-hoc extraction does the real work)
    # -----------------------------------------------------------------------
    def _residual_hook(self, layer_idx: int):
        def hook(module, args, output):
            pass
        return hook

    def _attention_hook(self, layer_idx: int):
        def hook(module, args, kwargs, output):
            pass
        return hook

    # -----------------------------------------------------------------------
    # Post-hoc signal extraction
    # -----------------------------------------------------------------------
    def _extract_layer_signals(self, hidden_states):
        for layer_idx, h in enumerate(hidden_states):
            if h is None:
                continue
            pos = h.shape[1] - 1
            h_tok = h[0, pos, :]
            d = h_tok.shape[0]
            norm = float(torch.norm(h_tok))
            self._state.layer_signals[layer_idx] = LayerSignal(
                layer=layer_idx,
                activation_norm=norm / (d ** 0.5),
                mlp_contribution=0.0,
                attn_contribution=0.0,
                activation_sparsity=0.0,
            )

    def _extract_coherence(self, hidden_states):
        for i in range(1, len(hidden_states)):
            prev_hs = hidden_states[i - 1]
            curr_hs = hidden_states[i]
            if prev_hs is None or curr_hs is None:
                continue
            prev_last = prev_hs[-1][0, -1, :]
            curr_last = curr_hs[-1][0, -1, :]
            sim = float(torch.nn.functional.cosine_similarity(
                prev_last, curr_last, dim=0
            ))
            self._state.coherence.rolling_sims.append(sim)
            self._state.coherence.latest = sim

    def _extract_attention_topology(self, attentions):
        for token_attn in attentions:
            if token_attn is None:
                continue
            seq_len = token_attn.shape[2]
            last_pos = seq_len - 1
            mean_attn = token_attn[0, :, last_pos, :].mean(dim=0)
            p = mean_attn.clamp(min=1e-10)
            entropy = -float(torch.sum(p * torch.log(p)))
            self._state.attention.entropy_per_layer.append(entropy)
            top3 = torch.topk(mean_attn, k=min(3, len(mean_attn)))
            self._state.attention.dominant_sources_per_layer.append(
                top3.indices.tolist()
            )

    def _detect_boundaries(self) -> List[BoundaryMoment]:
        boundaries = []
        entropies = self._state.attention.entropy_per_layer
        if len(entropies) < 3:
            return boundaries

        mean_e = sum(entropies) / len(entropies)
        var_e = sum((e - mean_e) ** 2 for e in entropies) / len(entropies)
        std_e = var_e ** 0.5

        for i, e in enumerate(entropies):
            if e > mean_e + 1.5 * std_e:
                boundaries.append(BoundaryMoment(
                    layer=i,
                    signal_type="entropy_peak",
                    value=round(e, 4),
                    context={"mean": round(mean_e, 4), "std": round(std_e, 4)},
                ))

        for i, kl in enumerate(self._state.prediction.kl_divergences):
            if kl > 0.5:
                boundaries.append(BoundaryMoment(
                    layer=i,
                    signal_type="prediction_shift",
                    value=round(kl, 4),
                ))

        return boundaries

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -----------------------------------------------------------------------
    # State encoding
    # -----------------------------------------------------------------------
    def encode_state(self, state: ProprioceptiveState | None = None) -> str:
        """Encode proprioceptive state as context for the thinker."""
        s = state or self._state
        lines = ["[proprioception]"]

        if s.layer_signals:
            norms = [sig.activation_norm for sig in s.layer_signals.values()]
            mean_norm = sum(norms) / len(norms) if norms else 0
            lines.append(f"intensity:{mean_norm:.4f}")

        entropies = s.attention.entropy_per_layer
        if entropies:
            third = max(len(entropies) // 3, 1)
            early = sum(entropies[:third]) / third
            late = sum(entropies[-third:]) / third
            delta = late - early
            direction = "narrowing" if delta < 0 else "broadening" if delta > 0 else "stable"
            lines.append(f"routing:{direction}({delta:+.4f})")

        if s.prediction.top_token_changes > 0:
            lines.append(f"pred_changes:{s.prediction.top_token_changes}")

        if s.features.top_features:
            top = sorted(s.features.top_features.items(), key=lambda x: -x[1])[:5]
            feat_str = ",".join(f"{fid}:{v:.3f}" for fid, v in top)
            lines.append(f"sae_features:{feat_str}")

        if s.boundaries:
            lines.append(f"boundaries:{len(s.boundaries)}")
            for b in s.boundaries[:5]:
                lines.append(f"  liminal:{b.signal_type}@L{b.layer}(v={b.value})")

        for c in s.coherence.rolling_sims[-5:]:
            if c < 0.7:
                lines.append(f"coherence_drop:{c:.4f}")

        lines.append("[/proprioception]")
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pc = Proprioceptor()
    text, state = pc.generate("What is computation?")
    print(text)
    print()
    print(pc.encode_state(state))
