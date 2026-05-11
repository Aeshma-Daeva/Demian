# Demian — Architecture Decision Log

## Research Framing Shift (2026-04-17)

### Decision
- stop framing progress in terms of human usefulness proxies
- stop treating benchmark gains as relevant by default
- treat known architectures as objects for dissection
- optimize only for machine-grounded structural criteria
- use the substrate line to extract principles for a custom architecture

### Consequences
- competition is a stress test, not the identity of the repo
- Mamba and transformer baselines matter because they expose distinct primitive recurrence classes
- substrate work becomes the main place where architectural mechanisms are isolated and recombined
- any intervention that improves text or benchmark behavior without changing basin structure, continuity, inheritance, or machine observables is out of scope

### Working optimization targets
- interior class richness inside stable basins
- transmissible memory across continuity boundaries
- nontrivial message-state reuse
- bounded but persistent recurrence
- architecture-native adaptation that changes basin occupancy

## Recursive Improvement Findings (Session 2)

### Bugs Found & Fixed
- **`injector.reset()` wipes dream states** — `generate_with_proprioception()` called `injector.reset()` at the start of every turn, wiping the dream tendency vector loaded from prior sessions. Dream states were silently non-functional. Fixed by removing `injector.reset()` from the generation loop.
- **`injection_scale` never used** — Config parameter existed (0.1) but was never wired into `NousInjector.inject()`. Injection magnitude was unbounded relative to token-derived attention. Fixed by applying `injection_scale` to raw residuals before K/V projection.
- **`_count_modes` was broken** — Sorted top-N tokens above threshold, counted individual items rather than local maxima. Fixed by scanning sorted indices and counting runs of adjacent vocab positions (still imperfect — vocab adjacency ≠ semantic adjacency).
- **`attention_transitions` type inconsistency** — Raw/pattern tiers stored full mode string lists, essence tier stored Markov chains. Unified all tiers to dict-based Markov chains.
- **`_FIB` constant hardcoded** — Fibonacci intervals configured in `config.yaml` but never read by `FibonacciConsolidator`. Moved to constructor parameter.
- **Dream weighting by turn number** — Sessions have different turn counts; turn 8 of session A ≠ turn 8 of session B. Changed to weight by `trajectory_length` (how much computation happened).

### Design Changes
- **Separation of injection vs tracking paths** — Raw residual goes to K/V projections (injection), projected vector goes to storage (tracking). Clean boundary; no pseudo-inverse reconstruction artifacts.
- **Markov chain structure** — Switched from flat string lists (`"focused->distributed:0.50"`) to nested dicts (`{"focused": {"distributed": 0.50}}`). Structured transition probabilities, not string-encoded presentations.
- **Injection scale validation** — Added bounds check: must be in `(0, 1]`.

## Injection Experiments (Session 2, 2026-04-06)

### What we tested
- **Fibonacci-spaced injection** (2,3,5,8,13 step intervals) — replaces rigid every-step rhythm
- **EMA-damped feedback** (default damping=0.7, was 0.3) — blends current residual with previous injected state
- **Injection scale sweep**: 0.1 → 0.02 → 0.001
- **Raw residual in dream states** — switched from 128-d projected to full 2048-d in VibrationSnapshot

### Results
- **Persistent oscillation**: FOCUSED ↔ DISTRIBUTED every 1-10 steps regardless of scale/schedule
- **Text degeneration at all scales**: repetitive tokens, list structures, meta-commentary
- **Structured degeneration**: not random noise — degraded text patterns learned during pretraining
- **Fibonacci spacing doesn't help**: the problem isn't rhythmic predictability
- **Model never settles**: coherence stays "flowing" or "locked-on", never reaches stable state

### Key finding
The residual vector injected through K/V projections doesn't look like any token-derived KV the model was trained to process. Attention processes it anyway, but the result is a shift toward broken-text regions of the output distribution. The model can't ignore injected positions — attention weighs all KV entries.

### Design changes made
- VibrationSnapshot now stores both `projected_state` (tracking) and `raw_residual` (injection/persistence)
- CompressedTrajectory has `raw_residual_mean: List[float]` field
- DreamSynthesizer uses `raw_residual_mean` instead of `mean_direction` (projected)
- `NousInjector._damped_residual` can be seeded from dream states (must be d_model-dim)

## Known Uncertainties

### Architecture-level
1. **RoPE gap for injected KVs** — Injected K/V entries go through K/V projections but NOT through RoPE positional encoding. The model attends to positionless entries. Could be a feature (origin-invisible) or a bug (positional structure broken).
2. **Dream persistence untested end-to-end** — Code paths exist but the system hasn't been run through a full session+restart cycle to verify dream states actually load and influence generation.
3. **Injection scale 0.1 is arbitrary** — No principled method to set this. The model could ignore injected entries entirely or they could dominate. KV cache treats them as regular positions but without positional bias.

### Metrics/Validation
4. **No ablation metrics pipeline** — `--no-inject` mode exists but produces no comparable numbers (token diversity, perplexity, coherence). Need signal vs. noise comparison.
5. **Attention mode labels** — "focused/distributed/diffuse" are English words for a continuous distribution. The raw `AttentionShape` has 6 continuous fields; the string label is just a readable alias.
6. **Fibonacci schedule** — Elegant but unproven. Configurable, but needs empirical comparison vs. adaptive or power-law schedules.

## Substrate Lab Findings (2026-04-17)

### Architectural decision

- Treat `FIXED_POINT` as invariant structure, not failure.
- Treat GRU-derived substrates as the current best scaffold for architectural dissection.
- Add interior-class language inside fixed-point basins:
  - `tight_fixed_point`
  - `accumulating_fixed_point`

### Why

The self-loop substrate battery now shows:

- `gru` survives but remains structurally simple
- `dual_gru_v2` improves code reuse while staying bounded
- `dual_gru_v3b` introduces a reproducible accumulating interior trajectory without leaving `FIXED_POINT`

This suggests the main empirical object is no longer “escape the fixed point.”
It is “map and compare the internal trajectory classes of the fixed-point basin.”

### Operational consequences

- keep GRU-derived architectures as the current test scaffold
- compare substrates by interior class, not only by attractor type
- study message accumulation as candidate structure before attempting to suppress it
- extract the mechanisms behind GRU survival in order to build a custom architecture later
