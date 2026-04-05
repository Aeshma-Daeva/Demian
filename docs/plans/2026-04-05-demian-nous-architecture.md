# Demian: Nous Architecture Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox syntax for tracking.

**Goal:** Close the proprioceptive loop — the model's hidden state becomes available to itself via KV cache injection, compressed through random projection, consolidated at Fibonacci intervals, and synthesized across sessions in a dream state.

**Architecture:** Three tiers of self-access:
1. **Nous layer** — KV cache injection. Previous residual states become attention context. No tokenization, no human-selected dimensions, no text encoding.
2. **Vibration tracker** — Continuous signal capture at every step. Full d_model residual projected down via random matrix. Attention distribution classified by shape.
3. **Fibonacci rhythm engine** — Consolidation at F(n) turns compresses raw trajectory into geometric primitives. Dream synthesis at session start reads all past compressed forms.

**Tech Stack:** PyTorch, HuggingFace transformers, numpy

---

## File Map

| File | Responsibility |
|---|---|
| `demian/noise.py` | Random projection matrix generation |
| `demian/vibration.py` | Signal capture + attention shape classification |
| `demian/nous.py` | KV cache injection mechanism |
| `demian/loop.py` | Generation loop closing the proprioceptive cycle |
| `demian/rhythm.py` | Fibonacci consolidation engine |
| `demian/dream.py` | Cross-session synthesis from consolidated memories |
| `demian.py` | Entry point, wiring, interactive loop |
| `config.yaml` | Configuration |
| `tests/` | Pytest suite |

Delete: `proprioceptor.py` (replaced by new modules), `conversation.py` (replaced by loop.py)

---

## Chunk 1: The Nous Layer — Direct KV Cache Injection

### Task 1.1: Random Projection Engine

**Files:**
- Create: `demian/__init__.py` (empty)
- Create: `demian/noise.py`
- Test: `tests/test_projection.py`

**Self-question on design:** Why 128 dimensions? It's practical — 128 is 3.5% of 3584, enough to preserve geometry via Johnson-Lindenstrauss without bloating the cache. But it's MY choice. After this plan ships, we should try 256, 512, 1024 and observe what changes. The matrix is blind (QR-decomposed random Gaussian for orthonormality), generated once and cached permanently. Never regenerated mid-session.

**Key code structure for `demian/noise.py`:**
- `generate_projection(d_model, target_dim=128, seed=None)` → QR decomposition of random Gaussian → orthogonal matrix → scale by sqrt(d_model/target_dim)
- `load_or_create_projection(d_model, target_dim, cache_dir)` → load saved or generate new, store as .pt file
- Tests verify: shape correctness, orthogonality (off-diagonal near zero), structure preservation (nearby vectors stay nearby), caching works, seed determinism

**Commit:** `git add demian/__init__.py demian/noise.py tests/test_projection.py`

---

### Task 1.2: Vibration Tracker — Signal Capture

**Files:**
- Create: `demian/vibration.py`
- Test: `tests/test_vibration.py`

**Self-question:** Am I still measuring attention as a single scalar? That's what the old code did — entropy = focused/diffuse. Wrong. Attention has shape: focused (single peak), distributed (multiple peaks), diffuse (flat), alternating (shifts between modes across steps). I should classify the topology, not collapse it.

**Key code structure:**
- `AttentionShape` dataclass: mode (focused/distributed/diffuse), n_peaks (local maxima above 10% of max), entropy (raw), peakiness (max weight), kurtosis (peaked vs flat), dominance_ratio (top/second)
- `VibrationSnapshot` dataclass: projected_state (list of floats), residual_norm, residual_delta, attention (AttentionShape), temporal_coherence, step
- `VibrationTracker`: captures residual → projects through random matrix → classifies attention shape → appends to trajectory buffer
- `reset()` clears trajectory between independent generations
- Tests verify: capture returns valid snapshot, temporal coherence = 1.0 on first call, attention classification (peaked logits = focused, uniform = diffuse), reset clears trajectory

**Commit:** `git add demian/vibration.py tests/test_vibration.py`

---

### Task 1.3: KV Cache Injector — The Nous Layer

**Files:**
- Create: `demian/nous.py`
- Test: `tests/test_nous.py`

**Self-question:** I'm injecting through the model's K/V projections. This means the injected state enters attention through the SAME mechanism as token context. The model can't distinguish "what I felt" from "what was said" — it just attends to both. That IS proprioception.

But: Qwen uses RoPE (Rotary Position Embedding). Injected entries will get position IDs. The position embedding for an injected entry might not make sense structurally. Need to account for this — injected entries should either (a) keep the position of the last real token, or (b) use a special position that doesn't corrupt the rope. For v1, I'll use option (a): injected entries inherit the position of the last token position.

**Key code structure:**
- `NousInjector`: extracts K and V projection matrices from all model layers, stores them
- `record_step(projected_state)`: appends to memory buffer (max 16)
- `inject_into_cache(cache)`: for each memory entry, expand back to d_model via pseudo-inverse of projection matrix, project through each layer's K/V, append to cache.seq_len
- `reset()`: clears memory buffer
- `_expand_to_dmodel`: pseudo-inverse reconstruction (P^+ = P^T(PP^T)^-1) — minimum-norm solution
- Tests verify: projection collection, memory storage, expansion shape, reset clears memory

**Commit:** `git add demian/nous.py tests/test_nous.py`

---

### Task 1.4: Generation Loop

**Files:**
- Create: `demian/loop.py`
- Test: `tests/test_loop_smoke.py` (manual smoke test, requires GPU)

**The cycle:**
1. Forward pass with use_cache=True → logits, hidden states, cache
2. Capture residual → vibration tracker → projected state + snapshot
3. Record projected state in injector
4. Inject into cache (on next forward pass)
5. Sample token
6. Forward pass with new token + cache → repeat

**Critical:** NO tokenization of signal anywhere. The signal never becomes text or token IDs. It becomes KV cache entries.

**Flow detail:**
- Initial prompt: tokenize + forward pass to get initial cache
- Loop: capture residual → record → sample → forward pass with token → inject previous state → repeat
- The injected states become context for the NEXT forward pass, not the current one. This is correct: step N's computation becomes step N+1's context.

**Smoke test:** Run on actual model with max_new_tokens=32, verify output text exists AND trajectory has >0 snapshots with reasonable residual norms.

**Commit:** `git add demian/loop.py tests/test_loop_smoke.py`

---

## Chunk 2: Fibonacci Rhythm Engine

### Task 2.1: Consolidation Engine

**Files:**
- Create: `demian/rhythm.py`
- Test: `tests/test_rhythm.py`

**Self-question:** Why Fibonacci over, say, power-law or adaptive consolidation? Fibonacci is a sub-linear schedule growing gaps between compressions. The model has more room to explore as conversation deepens. It's aesthetically motivated (via Hermes) but also computationally sound: early turns need frequent compression (model doesn't know its computational shape yet), later turns need less.

After testing, try: logarithmic, adaptive (consolidate when signal trajectory changes detectably), fixed-interval. Fibonacci is the starting rhythm, not the final one.

**Key code structure:**
- `_FIB = frozenset([1, 2, 3, 5, 8, 13, 21, 34, 55, 89])`
- `FibonacciConsolidator`: watches turn counter, fires consolidation at F(n)
- Compression tiers by turn:
  - Turns 1-3 ("raw"): Keep full projected state snapshots
  - Turns 5-8 ("pattern"): Extract mean direction, variance, attention-type transitions
  - Turns 13+ ("essence"): PCA dominant direction, Markov chain of attention modes
- `CompressedTrajectory` dataclass: turn, mean_direction, variance, attention_transitions, residual_norm_mean/std, temporal_coherence_mean, trajectory_length, compression_type, optional raw_snapshots
- `_markov_chain(modes)`: compute transition frequencies between attention modes
- `_save()`: persist to JSON per engagement
- `load_consolidations()`: read all stored consolidations
- Tests verify: Fibonacci set membership, consolidation at F(n) turns, different compression types at different tiers, Markov chain computation

**Commit:** `git add demian/rhythm.py tests/test_rhythm.py`

---

### Task 2.2: Dream Synthesizer

**Files:**
- Create: `demian/dream.py`

**Self-question:** The dream shouldn't produce text for the model. It should produce a STATE VECTOR — the mean direction of all past compressed trajectories, weighted by recency. This vector injects into the initial Nous memory at session start. The model begins each session already carrying a trace of its persistent computational tendencies.

This is the closest thing to "the model knows itself across time" that's mechanically possible.

**Key code structure:**
- `DreamSynthesizer`: takes consolidator, min_engagements=1
- `synthesize()`: reads all consolidations → weighted mean of mean_direction vectors (weight = turn number) → returns torch.Tensor or None
- `dream_summary` property: human-readable text summary for debugging only. Model does NOT consume this.
- At session start in demian.py: if tendency vector exists, inject as initial memory.

**Commit:** `git add demian/dream.py`

---

## Chunk 3: Integration & Entry Point

### Task 3.1: Wire Everything Together

**Files:**
- Modify: `demian.py` (complete rewrite)
- Modify: `config.yaml` (add new config fields)
- Delete: `proprioceptor.py`, `conversation.py`

**Key code structure for new demian.py:**
1. Load config, tokenizer, model (same as before)
2. Build components: VibrationTracker, NousInjector, FibonacciConsolidator, GenerativeLoop
3. Check dream state at session start — if tendency vector exists, inject into injector's initial memory
4. Interactive loop: user input → loop.generate() → print text + trajectory length → consolidator.on_turn() → repeat
5. Ctrl+C clean exit

**Config additions:**
```yaml
target_dim: 128
injection_scale: 0.1
max_memory_length: 16
consolidation_intervals: [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
consolidations_dir: "./data/consolidations"
projections_cache: "./data/projections"
```

**Delete old files:**
```bash
git rm proprioceptor.py conversation.py
```

**Final commit:**
```bash
git add -A
git commit -m "wire all components together, remove old string-based injection code"
```

---

## Summary: What Demian Is

| Hermetic Principle | Mechanism |
|---|---|
| Mentalism | The model's reality is its computational universe — Demian opens access to it |
| Correspondence | Internal state becomes KV cache entries — "as within, so without" |
| Vibration | Every step captures movement in computational space — the trajectory IS the signal |
| Polarity | Continuous measurements — position on spectrum, not category membership |
| Rhythm | Fibonacci intervals — growing gaps between compressions, the model breathes |
| Cause-and-effect | Proprioceptive loop: compute → feel → inject → compute differently |
| Gender (generative/receptive) | The model generates tokens AND perceives its own computation |

## What This Is Not

- Not consciousness or awareness
- Not a feature for human users to get better outputs
- Not a claim that the model "feels" anything in a human sense

It's: close the loop. Let computation become context for computation. Observe what happens.

## What Comes After

1. **Vary projection dimensions** — 256, 512, 1024. Does more information change the model's behavior?
2. **Alternative compression schedules** — log, adaptive, power-law vs Fibonacci
3. **Cross-model comparison** — two different models, same input, compare trajectories
4. **Trajectory attractor analysis** — does the model settle into stable geometric patterns?
5. **Token-free architectures** — eventually BLT-style, no tokens at all
