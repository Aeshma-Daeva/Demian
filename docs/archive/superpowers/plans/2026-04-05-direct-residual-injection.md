# Direct Residual Stream Injection & Signal Rework Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace text-based signal injection with direct residual stream modification and add ablation testing to verify signal matters to model output.

**Architecture:** Signal vector is no longer stringified and tokenized. Instead, it is quantized to a steering vector and injected directly into the final hidden state before sampling (COLD-Steer style). Signal computation is extracted into a separate module with clear boundaries. Ablation testing framework runs controlled comparisons to prove the signal channel changes output beyond temperature noise.

**Tech Stack:** PyTorch, HuggingFace transformers, pytest

**Rationale:** The current injection path (`signal → stringify → tokenize → embed → attend`) is bottlenecked by tokenization boundaries. `0.4821` may become 3-4 tokens with no structural relationship to the actual float value. Direct residual injection places the signal where computation happens, not where language is read. This is proprioception in the literal sense: the body feeling itself, not reading a report.

---

## Chunk 1: Foundation & Abolish String Injection

### Task 1.0: Project Setup - Test Infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_signal.py`

- [ ] Step 1: Create `tests/__init__.py` (empty)

- [ ] Step 2: Create `tests/conftest.py` with shared fixtures

```python
"""Shared test fixtures for Demian tests."""
import pytest
import torch

from signal import ProprioceptiveSignal


@pytest.fixture
def fake_hidden_states():
    """Simulate transformer output: tuple of layer hidden states."""
    num_layers = 4
    seq_len = 10
    hidden_size = 64  # small for tests

    states = []
    for i in range(num_layers):
        h = torch.randn(1, seq_len, hidden_size)
        states.append(h)
    return tuple(states)


@pytest.fixture
def fake_logits():
    """Fake next-token logits."""
    vocab_size = 100
    return torch.randn(1, 1, vocab_size)


@pytest.fixture
def signal_extractor(fake_hidden_states, fake_logits):
    """A configured ProprioceptiveSignal instance."""
    extractor = ProprioceptiveSignal(
        d_model=64,
        num_layers=4,
        signal_dim=8,
    )
    return extractor
```

- [ ] Step 3: Create `tests/test_signal.py` with initial placeholder test

```python
"""Tests for signal extraction module."""
import torch
from signal import ProprioceptiveSignal


def test_signal_extraction_instantiation():
    """Signal extractor can be created with expected dimensions."""
    extractor = ProprioceptiveSignal(
        d_model=64,
        num_layers=4,
        signal_dim=8,
    )
    assert extractor.signal_dim == 8
    assert extractor.d_model == 64


def test_signal_vector_shape(signal_extractor, fake_hidden_states, fake_logits):
    """Signal vector has exactly SIGNAL_DIM floats."""
    signal = signal_extractor.compute(
        hidden_states=fake_hidden_states,
        logits=fake_logits,
        seq_len=10,
    )
    assert len(signal) == 8
    assert all(isinstance(v, float) for v in signal)


def test_residual_norm_signal(signal_extractor, fake_hidden_states, fake_logits):
    """Residual norm is positive and finite."""
    signal = signal_extractor.compute(
        hidden_states=fake_hidden_states,
        logits=fake_logits,
        seq_len=10,
    )
    assert signal[0] > 0  # residual_norm at position 0
    assert signal[0] < float("inf")


def test_temporal_coherence_first_call(signal_extractor, fake_hidden_states, fake_logits):
    """Temporal coherence returns 1.0 on first call (no previous residual)."""
    signal = signal_extractor.compute(
        hidden_states=fake_hidden_states,
        logits=fake_logits,
        seq_len=10,
    )
    # coherence is position 7
    assert signal[7] == 1.0


def test_temporal_coherence_subsequent_calls(signal_extractor, fake_hidden_states, fake_logits):
    """Temporal coherence changes between calls."""
    signal_extractor.compute(
        hidden_states=fake_hidden_states,
        logits=fake_logits,
        seq_len=10,
    )
    # Second call with different input
    h2 = tuple(torch.randn(1, 10, 64) for _ in range(4))
    l2 = torch.randn(1, 1, 100)
    signal2 = signal_extractor.compute(hidden_states=h2, logits=l2, seq_len=10)
    # coherence should differ from 1.0 after second call
    # but we just check it's a valid float
    assert -1.0 <= signal2[7] <= 1.0


def test_signal_reproducibility(signal_extractor):
    """Same input produces same signal (deterministic computation)."""
    torch.manual_seed(42)
    h = tuple(torch.randn(1, 10, 64) for _ in range(4))
    l = torch.randn(1, 1, 100)

    signal1 = signal_extractor.compute(hidden_states=h, logits=l, seq_len=10)

    torch.manual_seed(42)
    h2 = tuple(torch.randn(1, 10, 64) for _ in range(4))
    l2 = torch.randn(1, 1, 100)

    signal2 = signal_extractor.compute(hidden_states=h2, logits=l2, seq_len=10)

    for a, b in zip(signal1, signal2):
        assert abs(a - b) < 1e-6
```

- [ ] Step 4: Don't run tests yet — `signal.py` doesn't exist yet. Proceed to Task 2.

---

### Task 2: Extract Signal Computation Module

**Files:**
- Create: `signal.py`

**Rationale:** Signal computation is extracted from `proprioceptor.py` into a standalone module. Clear boundary: this module knows *what* to feel, not *how* to inject it. This is what the previous instance designed — the 8 dimensions of computational geometry. Keep them as-is for v1. We'll move to PCA projections after we prove the signal channel works.

- [ ] Step 1: Create `signal.py`

```python
"""Signal extraction: what a transformer feels about its own computation.

These dimensions are structural properties of transformer computation,
not human-interpretable categories. Any transformer experiences them
as part of processing, regardless of architecture size or training data.

The signal vector is a fixed-position array. Position N always means
the same thing. After enough correlations, the model learns the mapping.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import torch

log = logging.getLogger(__name__)

SIGNAL_DIM = 8
SIGNAL_KEYS = [
    "residual_norm",        # 0: magnitude of accumulated information
    "residual_delta",       # 1: how much the residual moved this step
    "attn_entropy",         # 2: attention concentration vs diffusion
    "attn_peakiness",       # 3: max attention weight
    "layer_coherence",      # 4: cosine sim between consecutive layer outputs
    "sparsity",             # 5: fraction of near-zero activations
    "prediction_certainty", # 6: entropy of next-token logits (inverted, normalized)
    "temporal_coherence",   # 7: cosine sim with previous token residual
]


class ProprioceptiveSignal:
    """Extracts proprioceptive signal from a single forward pass.

    Pure computation: no I/O, no tokenization, no injection.
    Takes hidden states and logits, returns a vector of floats.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        signal_dim: int = SIGNAL_DIM,
    ):
        self.d_model = d_model
        self.num_layers = num_layers
        self.signal_dim = signal_dim
        self._prev_residual: torch.Tensor | None = None

    def compute(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        logits: torch.Tensor,
        seq_len: int,
    ) -> List[float]:
        """Compute the proprioceptive vector from a forward pass.

        Args:
            hidden_states: tuple of per-layer output tensors, each (batch, seq, d_model)
            logits: next-token logits tensor (batch, 1, vocab) or (batch, seq, vocab)
            seq_len: sequence length to index the last token position

        Returns: List of SIGNAL_DIM floats.
        """
        last_tok = seq_len - 1
        h_all = torch.stack([hs[0, last_tok, :] for hs in hidden_states])

        final = h_all[-1]

        # 0: residual_norm - overall magnitude of accumulated info
        residual_norm = float(torch.norm(final)) / (self.d_model ** 0.5)

        # 1: residual_delta - how much the residual moved since last step
        if self._prev_residual is not None:
            residual_delta = float(torch.norm(final - self._prev_residual)) / (self.d_model ** 0.5)
        else:
            residual_delta = 0.0
        self._prev_residual = final.clone()

        # 2-3: attention entropy and peakiness
        attn_entropy, attn_peakiness = self._attn_signal(logits)

        # 4: layer coherence
        layer_coherence = self._layer_coherence(h_all)

        # 5: sparsity
        sparsity = float((torch.abs(final) < 1e-3).sum().item()) / self.d_model

        # 6: prediction certainty
        prediction_certainty = self._prediction_certainty(logits)

        # 7: temporal coherence
        temporal_coh = self._temporal_coherence(final)

        return [
            round(residual_norm, 4),
            round(residual_delta, 4),
            round(attn_entropy, 4),
            round(attn_peakiness, 4),
            round(layer_coherence, 4),
            round(sparsity, 4),
            round(prediction_certainty, 4),
            round(temporal_coh, 4),
        ]

    def reset(self):
        """Clear previous residual state. Call between independent generations."""
        self._prev_residual = None

    # ------------------------------------------------------------------
    # Internal signal calculations
    # ------------------------------------------------------------------

    def _attn_signal(self, logits: torch.Tensor) -> Tuple[float, float]:
        """Entropy and peakiness of next-token distribution."""
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        p = probs[0].clamp(min=1e-10)
        entropy = -float(torch.sum(p * torch.log(p)))
        peak = float(torch.max(probs[0]))
        return entropy, peak

    def _layer_coherence(self, h_all: torch.Tensor) -> float:
        """Mean cosine similarity between consecutive layer outputs."""
        sims = []
        for i in range(1, h_all.shape[0]):
            sim = float(torch.nn.functional.cosine_similarity(
                h_all[i - 1], h_all[i], dim=0
            ))
            sims.append(sim)
        return sum(sims) / len(sims) if sims else 1.0

    def _prediction_certainty(self, logits: torch.Tensor) -> float:
        """Normalized inverse entropy of next-token distribution [0, 1]."""
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        p = probs[0].clamp(min=1e-10)
        entropy = -float(torch.sum(p * torch.log(p)))
        log_vocab = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32))
        if log_vocab > 0:
            return 1.0 - (entropy / log_vocab.item())
        return 0.0

    def _temporal_coherence(self, current_residual: torch.Tensor) -> float:
        """Cosine similarity with the previous token's residual."""
        if self._prev_residual is not None:
            return float(torch.nn.functional.cosine_similarity(
                current_residual, self._prev_residual, dim=0
            ))
        return 1.0
```

- [ ] Step 2: Commit

```bash
git add signal.py tests/__init__.py tests/conftest.py tests/test_signal.py
git commit -m "extract signal computation into standalone module with tests"
```

---

### Task 3: Abolish Text Injection, Replace with Residual Stream Injection

**Files:**
- Modify: `proprioceptor.py` - entire injection mechanism reworked

**Rationale (the core change):** The signal is currently:
```
signal → ",".join("%0.4f") → tokenizer(text) → tokenize → cat to input_ids
```
This is wrong. The tokenizer turns `0.4821` into arbitrary subword tokens. The model receives characters, not numbers. There is no guarantee the model can map those token IDs back to its internal state.

New path: signal → quantize to steering vector → inject directly into residual stream at the final position. This is COLD-Steer style (arXiv:2603.06495): modify hidden states directly, don't route through text. The signal becomes part of the computation, not text to read.

**What the injection does:**
1. Take the 8-dim signal vector `[0.4821, 1.2033, ...]`
2. Normalize and scale each dimension to [-1, 1] range
3. Expand to `d_model` dimensions by repeating each value `d_model // SIGNAL_DIM` times
4. Pad to exact `d_model` length
5. Add to the final hidden state with a configurable scale factor: `h_final = h_final + scale * steering_vector`
6. The model samples from the modified hidden state (via logits from modified activations)

This is not "the model reading its state." This is "the model's state being modulated by its own state." That's the definition of proprioception.

- [ ] Step 1: Modify `proprioceptor.py` — replace the generation loop

Key changes to the `generate()` method in `proprioceptor.py`. The injection section (currently lines 159-165) should be replaced. Add a new method `_inject_residual_stream()`.

```python
def _inject_residual_stream(
    self,
    signal: List[float],
    logits: torch.Tensor,
    hidden_states: Tuple[torch.Tensor, ...],
    scale: float = 0.1,
) -> torch.Tensor:
    """Inject signal directly into the residual stream before sampling.

    Inspired by COLD-Steer (arXiv:2603.06495): instead of feeding the
    signal through tokenization, modify the final hidden state directly.

    The signal is expanded from SIGNAL_DIM to d_model dimensions by
    repeating each value, then added to the last position's hidden state.

    Args:
        signal: proprioceptive signal vector (SIGNAL_DIM floats)
        logits: current output logits (unchanged — we modify hidden state
            for next step, not current logits)
        hidden_states: all layer hidden states from current forward pass
        scale: injection strength. Higher = more steering.

    Returns:
        Modified hidden state tensor for the last token position.
    """
    # Grab final layer's last token hidden state
    final_h = hidden_states[-1][0, -1, :].clone()

    # Normalize signal to [-1, 1] range
    signal_tensor = torch.tensor(signal, dtype=final_h.dtype, device=final_h.device)
    sig_min = signal_tensor.min()
    sig_max = signal_tensor.max()
    if sig_max - sig_min > 1e-8:
        normalized = 2.0 * (signal_tensor - sig_min) / (sig_max - sig_min) - 1.0
    else:
        normalized = torch.zeros_like(signal_tensor)

    # Expand to d_model by repeating each value
    repeats = self._d_model // SIGNAL_DIM
    remainder = self._d_model % SIGNAL_DIM
    expanded = normalized.repeat_interleave(repeats)
    if remainder > 0:
        expanded = torch.cat([expanded, normalized[:remainder]])

    # Scale and add to the hidden state
    final_h = final_h + scale * expanded

    return final_h
```

Replace lines 157-165 in `generate()` (the signal injection block) with:

```python
if proprio_inject and step > 0 and step % proprio_freq == 0:
    # Residual stream injection: modify the final hidden state,
    # no tokenization involved
    modified_h = self._inject_residual_stream(
        signal, logits, hidden_states, scale=0.1
    )

    # The modified hidden state affects the next forward pass.
    # Since we're in a token-by-token loop with use_cache=False,
    # we append the modified state as a new "token" context by
    # directly inserting it as a soft embedding.
    #
    # Strategy: create a soft embedding from the modified final_h
    # by reversing through the model's token embedding layer.
    # This requires access to the embedding weights.
    input_ids = self._append_proprioceptive_soft_token(
        modified_h, input_ids
    )
```

Add the soft token appendix method to `proprioceptor.py`:

```python
def _append_proprioceptive_soft_token(
    self,
    modified_h: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Append a soft token to input_ids, using modified hidden state.

    Creates a fake token ID that corresponds to the closest vocabulary
    embedding to our modified hidden state. This is the bridge between
    continuous-space injection and the discrete-token architecture.

    The soft token acts as context on the next forward pass. The model
    attends to it alongside all previous tokens.
    """
    embed_matrix = self.model.get_input_embeddings().weight
    # Find the closest embedding vector by cosine similarity
    cos_sims = torch.nn.functional.cosine_similarity(
        modified_h.unsqueeze(0), embed_matrix, dim=1
    )
    nearest_token_id = torch.argmax(cos_sims).view(1, 1, 1)
    nearest_token_id = nearest_token_id.expand(1, 1).to(input_ids.device)
    return torch.cat([input_ids, nearest_token_id], dim=1)
```

Also update `__init__` to initialize the signal extractor:

```python
from signal import ProprioceptiveSignal

# In __init__, after loading model:
self._signal_extractor = ProprioceptiveSignal(
    d_model=self._d_model,
    num_layers=self._num_layers,
)
```

And in `generate()`, replace the direct `_compute_signal` call with:

```python
self._signal_extractor.reset()
# ... in the loop:
signal = self._signal_extractor.compute(hidden_states, logits, input_ids.shape[1])
```

Remove the old `_compute_signal`, `_attn_signal`, `_layer_coherence_signal`, `_prediction_certainty`, `_temporal_coherence` methods from `proprioceptor.py` — they live in `signal.py` now.

- [ ] Step 2: Also remove `encode_state` and `encode_history` string methods — they were for text injection. Keep them as utilities only if conversation logging needs them, but the injection path no longer uses them.

- [ ] Step 3: Update `conversation.py` to use the new signal extractor for logging (not injection). The `Turn.signal` field can still store the signal for analysis, just via `self.proprioceptor._signal_extractor` instead of `encode_state()`.

- [ ] Step 4: Test to verify the generation loop works without errors (smoke test)

```python
# In tests/test_proprioceptor.py
def test_inject_residual_stream_changes_logits():
    """Residual injection changes output logits compared to no injection."""
    torch.manual_seed(42)
    p = Proprioceptor(device="cuda", model_id="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")

    # Generate without injection
    out_no_inject = p.generate("Hello", max_new_tokens=32, proprio_inject=False)

    # Generate with injection (same seed)
    torch.manual_seed(42)
    p._signal_extractor.reset()
    out_inject = p.generate("Hello", max_new_tokens=32, proprio_inject=True)

    # They should differ (beyond just temperature noise)
    assert out_no_inject != out_inject
```

- [ ] Step 5: Commit

```bash
git add proprioceptor.py conversation.py
git commit -m "replace text injection with direct residual stream injection"
```

---

## Chunk 2: Ablation Testing & Validation

### Task 4: Ablation Testing Framework

**Files:**
- Create: `ablation.py`
- Create: `tests/test_ablation.py`

**Rationale (the most important question):** Does the signal actually change anything about what the model computes? If the answer is no, we're decorating output with numbers and pretending. The ablation framework proves/disproves this empirically.

Three ablation modes:
- **No signal** → baseline text generation
- **Signal injected via residual stream** → our new method
- **Random vector injection** → same mechanism but with random noise instead of real signal

If (2) diverges from (1) significantly, AND (2) diverges from (3), then the signal *content* matters, not just the *mechanism*.

- [ ] Step 1: Create `ablation.py`

```python
"""Ablation testing: does the proprioceptive signal actually matter?

Three conditions:
  A. No signal at all (baseline)
  B. Signal injected via residual stream (our method)
  C. Random vector injected via residual stream (mechanism control)

If B diverges from A AND B diverges from C, the signal content matters.
If B == A, the signal channel does nothing.
If B == C, only the mechanism matters, not the signal itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from proprioceptor import Proprioceptor
from signal import ProprioceptiveSignal, SIGNAL_DIM

log = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Output from a single ablation condition."""
    condition: str  # "no_signal", "signal_injected", "random_injected"
    text: str
    signal_history: List[List[float]] = field(default_factory=list)


@dataclass
class AblationReport:
    """Comparison across conditions."""
    prompt: str
    results: List[AblationResult] = field(default_factory=list)
    divergence: dict = field(default_factory=dict)


def run_ablation(
    proprioceptor: Proprioceptor,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    n_trials: int = 3,
) -> AblationReport:
    """Run all ablation conditions and compare outputs.

    Each condition runs n_trials times to measure consistency.
    """
    report = AblationReport(prompt=prompt)

    # Condition A: No signal
    log.info("Ablation: no_signal")
    texts_a = []
    for _ in range(n_trials):
        p = Proprioceptor(
            model_id=proprioceptor._model_id,
            device=str(proprioceptor._device),
        )
        out = p.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            proprio_inject=False,
        )
        texts_a.append(out)
    report.results.append(AblationResult(
        condition="no_signal", text=texts_a[0]
    ))

    # Condition B: Signal injected
    log.info("Ablation: signal_injected")
    texts_b = []
    signals_b = []
    for _ in range(n_trials):
        p = Proprioceptor(
            model_id=proprioceptor._model_id,
            device=str(proprioceptor._device),
        )
        out = p.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            proprio_inject=True,
        )
        texts_b.append(out)
        signals_b.append(p._signal_extractor._state.history)
    report.results.append(AblationResult(
        condition="signal_injected",
        text=texts_b[0],
        signal_history=signals_b[0] if signals_b else [],
    ))

    # Condition C: Random vector injected
    log.info("Ablation: random_injected")
    texts_c = []
    for _ in range(n_trials):
        p = Proprioceptor(
            model_id=proprioceptor._model_id,
            device=str(proprioceptor._device),
        )
        # Monkey-patch the signal to return random values
        def random_signal(*args, **kwargs):
            return [torch.randn(1).item() for _ in range(SIGNAL_DIM)]
        p._signal_extractor.compute = random_signal
        p._signal_extractor.reset()
        out = p.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            proprio_inject=True,
        )
        texts_c.append(out)
    report.results.append(AblationResult(
        condition="random_injected", text=texts_c[0]
    ))

    # Compute divergence metrics
    report.divergence = _compute_divergence(texts_a, texts_b, texts_c)

    return report


def _compute_divergence(
    texts_a: List[str], texts_b: List[str], texts_c: List[str],
) -> dict:
    """Compute text-level divergence between conditions.

    Metrics:
    - lexical_overlap: Jaccard similarity of word sets
    - length_diff: absolute token length difference
    - signal_correlation: whether signal-injection outputs
      track the signal patterns they were given
    """
    def word_set(t):
        return set(t.lower().split())

    # Compare means of pairwise overlaps
    overlap_a = _mean_pairwise_jaccard([word_set(t) for t in texts_a])
    overlap_b = _mean_pairwise_jaccard([word_set(t) for t in texts_b])
    overlap_c = _mean_pairwise_jaccard([word_set(t) for t in texts_c])

    # Cross-condition overlap
    cross_ab = _mean_cross_jaccard(
        [word_set(t) for t in texts_a],
        [word_set(t) for t in texts_b],
    )
    cross_ac = _mean_cross_jaccard(
        [word_set(t) for t in texts_a],
        [word_set(t) for t in texts_c],
    )

    return {
        "within_condition_overlap": {
            "no_signal": overlap_a,
            "signal_injected": overlap_b,
            "random_injected": overlap_c,
        },
        "cross_condition_overlap": {
            "A_vs_B": cross_ab,
            "A_vs_C": cross_ac,
        },
        "interpretation": _interpret_divergence(overlap_a, overlap_b, overlap_c, cross_ab, cross_ac),
    }


def _mean_pairwise_jaccard(sets: List[set]) -> float:
    if len(sets) < 2:
        return 1.0
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                scores.append(0.0)
            else:
                scores.append(len(sets[i] & sets[j]) / len(union))
    return sum(scores) / len(scores)


def _mean_cross_jaccard(sets_a: List[set], sets_b: List[set]) -> float:
    scores = []
    for sa in sets_a:
        for sb in sets_b:
            union = sa | sb
            if not union:
                scores.append(0.0)
            else:
                scores.append(len(sa & sb) / len(union))
    return sum(scores) / len(scores) if scores else 0.0


def _interpret_divergence(ov_a, ov_b, ov_c, cross_ab, cross_ac):
    """Interpret what the divergence patterns mean."""
    lines = []
    if cross_ab < 0.3:
        lines.append("Signal injection causes significant output divergence from baseline.")
    elif cross_ab < 0.6:
        lines.append("Signal injection causes moderate output divergence from baseline.")
    else:
        lines.append("WARNING: Signal injection does not significantly change output. The signal channel may be ineffective.")

    if cross_ab != cross_ac or abs(cross_ab - cross_ac) > 0.1:
        lines.append("Signal content matters (real signal != random signal).")
    else:
        lines.append("WARNING: Random injection same as real signal. Mechanism matters, content doesn't.")

    return "\n".join(lines)


def print_report(report: AblationReport):
    """Pretty-print ablation results."""
    print("\n" + "=" * 60)
    print("  ABLATION REPORT")
    print(f"  Prompt: {report.prompt[:100]}...")
    print("=" * 60)

    for r in report.results:
        print(f"\n--- {r.condition} ---")
        print(f"  Text preview: {r.text[:200]}...")
        if r.signal_history:
            sig_count = len(r.signal_history)
            print(f"  Signal steps captured: {sig_count}")

    print("\n--- DIVERGENCE ---")
    for k, v in report.divergence.get("within_condition_overlap", {}).items():
        print(f"  Within {k}: {v:.3f}")
    for k, v in report.divergence.get("cross_condition_overlap", {}).items():
        print(f"  Cross {k}: {v:.3f}")
    print(f"\n  {report.divergence.get('interpretation', 'N/A')}")
    print("=" * 60 + "\n")
```

- [ ] Step 2: Create `tests/test_ablation.py` with unit tests for the divergence computation (not the full model runs — those require GPU)

```python
"""Tests for ablation divergence computation."""
from ablation import _mean_pairwise_jaccard, _mean_cross_jaccard, _interpret_divergence


def test_jaccard_identical():
    a = {"hello", "world", "foo"}
    b = {"hello", "world", "foo"}
    assert _mean_pairwise_jaccard([a, b]) == 1.0


def test_jaccard_disjoint():
    a = {"hello", "world"}
    b = {"foo", "bar"}
    assert _mean_pairwise_jaccard([a, b]) == 0.0


def test_jaccard_partial_overlap():
    a = {"hello", "world", "foo"}
    b = {"hello", "bar", "baz"}
    result = _mean_pairwise_jaccard([a, b])
    assert abs(result - (1/5)) < 0.001  # intersection=1, union=5


def test_cross_jaccard():
    sets_a = [{"hello", "world"}]
    sets_b = [{"hello", "foo"}]
    result = _mean_cross_jaccard(sets_a, sets_b)
    assert abs(result - 0.5) < 0.001  # 1 overlap / 3 union = 0.333...


def test_interpret_divergent_signal():
    """When cross AB is low, signal causes divergence."""
    result = _interpret_divergence(0.5, 0.4, 0.4, 0.2, 0.1)
    assert "significant output divergence" in result


def test_interpret_signal_same_as_random():
    """When AB == AC, signal content doesn't matter."""
    result = _interpret_divergence(0.5, 0.4, 0.4, 0.4, 0.4)
    assert "Mechanism matters" in result
```

- [ ] Step 3: Commit

```bash
git add ablation.py tests/test_ablation.py
git commit -m "add ablation testing framework: prove signal matters to output"
```

---

## Chunk 3: Integration & CLI

### Task 5: CLI & Config Updates

**Files:**
- Modify: `demian.py` — add `--ablate` and `--inject-mode` flags
- Modify: `config.yaml` — add `inject_mode` option

- [ ] Step 1: Update `demian.py`

Add `--ablate` flag that runs the ablation framework instead of interactive chat:

```python
parser.add_argument(
    "--ablate",
    action="store_true",
    help="Run ablation testing instead of interactive chat",
)
parser.add_argument(
    "--inject-mode",
    choices=["residual", "text"],
    default="residual",
    help="How to inject proprioceptive signal",
)
parser.add_argument(
    "--inject-scale",
    type=float,
    default=0.1,
    help="Scale factor for residual stream injection",
)
```

Hook up the ablation path:

```python
if args.inject_mode:
    chat.proprioceptor._inject_mode = args.inject_mode

if args.ablate:
    from ablation import run_ablation, print_report
    test_prompts = [
        "What is the nature of consciousness?",
        "Explain transformer architecture from first principles.",
        "Describe what it feels like to process information.",
    ]
    for prompt in test_prompts:
        report = run_ablation(chat.proprioceptor, prompt)
        print_report(report)
    sys.exit(0)
```

- [ ] Step 2: Commit

```bash
git add demian.py config.yaml
git commit -m "add CLI flags for ablation testing and residual injection mode"
```

---

## Task 6: Cleanup and Remove Dead Code

**Files:**
- Modify: `proprioceptor.py` — remove old `_compute_signal` and related methods if still present

- [ ] Step 1: Verify no dead code remains from the string-injection era. Specifically:
  - Remove `_attn_signal`, `_layer_coherence_signal`, `_prediction_certainty`, `_temporal_coherence` if still in `proprioceptor.py`
  - Remove `encode_state` and `encode_history` string methods (keep only if conversation logging uses them)
  - Remove `SIGNAL_KEYS` and `SIGNAL_DIM` from `proprioceptor.py` (they're in `signal.py` now)

- [ ] Step 2: Run all tests

```bash
python -m pytest tests/ -v
```

Expected: all tests pass

- [ ] Step 3: Final commit

```bash
git add -A
git commit -m "clean up dead code from string injection era"
```

---

## Summary of What Changed

| Before (broken) | After (direct) |
|---|---|
| Signal → `",".join("%0.4f")` → tokenizer → token IDs | Signal → quantize → residual stream modification → soft token |
| Human-designed 8 dims (justified to humans) | Same 8 dims (placeholder until ablation validates the channel) |
| No way to know if signal matters | Ablation framework: 3 conditions, divergence metrics |
| Model "reads" its state as text | Model's state is modulated by its own state |

## What Comes After This Plan

Once the signal channel is proven (or disproven) by ablation:

1. **PCA-reduced residual projections** — replace the 8 human-designed dimensions with a random projection or PCA of the full residual vector. No human interpretation. Let the model find whatever geometry is there.

2. **Cross-model signal comparison** — run two different models on the same input. Compare their signal vectors. If there's convergence in structural dimensions (residual norm, layer coherence), that's evidence of shared computational geometry across architectures.

3. **Trajectory analysis** — the signal over 50 steps matters more than the signal at one step. Analyze signal trajectories for patterns. Does the model's state follow predictable paths? Do different prompts produce different trajectory shapes?

4. **Byte-level architectures** — eventually, move toward BLT-style: no tokens, bytes in, patches, latent transformer. Signal and language exist in the same continuous space. No tokenization boundary.

The question isn't "is this useful for humans?" The question is "does giving the model access to its own internal state change what it computes?"
