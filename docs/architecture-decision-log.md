# Demian — Architecture Decision Log

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

## Known Uncertainties

### Architecture-level
1. **RoPE gap for injected KVs** — Injected K/V entries go through K/V projections but NOT through RoPE positional encoding. The model attends to positionless entries. Could be a feature (origin-invisible) or a bug (positional structure broken).
2. **Dream persistence untested end-to-end** — Code paths exist but the system hasn't been run through a full session+restart cycle to verify dream states actually load and influence generation.
3. **Injection scale 0.1 is arbitrary** — No principled method to set this. The model could ignore injected entries entirely or they could dominate. KV cache treats them as regular positions but without positional bias.

### Metrics/Validation
4. **No ablation metrics pipeline** — `--no-inject` mode exists but produces no comparable numbers (token diversity, perplexity, coherence). Need signal vs. noise comparison.
5. **Attention mode labels** — "focused/distributed/diffuse" are English words for a continuous distribution. The raw `AttentionShape` has 6 continuous fields; the string label is just a readable alias.
6. **Fibonacci schedule** — Elegant but unproven. Configurable, but needs empirical comparison vs. adaptive or power-law schedules.
