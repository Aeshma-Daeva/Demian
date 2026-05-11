# Continuity Engine: Implementation Plan

*The model finds equilibrium through repeated self-encounters.*

## Phase 1: Continuity Engine Core

### File: `demian/continuity.py`

**What it does:**
Orchestrates multi-run generation with persistent dream state. Single prompt, repeated autonomous runs, each seeding the next.

**Data flow:**
```
prompt → Run 1 → capture residual → consolidate → save dream state
  → Run 2 (loads dream_1) → inject dream_residual at step 0 → capture → consolidate → merge dream state
  → Run 3 (loads dream_1+2) → ...
  → Run N ...
```

**Key class: `ContinuityRunner`**

```python
class ContinuityRunner:
    def __init__(self, model, tokenizer, config):
        # Load accumulated dream state or start fresh
        self.dream_state = self._load_or_init_dream_state()
        self.run_counter = 0

    def run_continuity(self, prompt, n_runs, max_tokens_per_run):
        for _ in range(n_runs):
            self.run_counter += 1
            # Build injection: current step 0 dream residual
            dream_residual = self._synthesize_dream_residual()

            result = generate_with_proprioception(
                prompt, dream_residual, max_tokens_per_run
            )

            # Consolidate this run's trajectory
            snapshot = self._consolidate(result.trajectory)

            # Merge into dream state (EMA-style update)
            self.dream_state.append(snapshot)
            self._save_dream_state()
            self._save_run_metrics(result.metrics)

            # Next run uses SAME prompt but accumulated dream state
            prompt = None  # After first run, prompt is dream state
```

**Synthesizing the dream residual:**
- Take all past run snapshots
- Compute weighted mean of `residual_mean` vectors (recent runs weighted more heavily)
- Scale by `dream_weight` config parameter
- This becomes the additional injection at step 0 of each new run: `injected = raw_residual + dream_weight * dream_residual`

## Phase 2: Structural Metrics Pipeline

### File: `demian/metrics.py` (new)

**Per-run metric collector:**
Collects and saves ONLY structural metrics:
- Total mode flips
- Mean energy, energy delta
- Oscillation period (via autocorrelation of attention state sequence)
- Mode flip asymmetry
- Residual norm mean/variance
- Dream state cosine similarity to previous run's trajectory mean

**Aggregate tracker:**
Running CSV with one row per run:
```
run_number, mode_flips, mean_energy, energy_delta, osc_period,
flip_asymmetry, residual_norm_mean, residual_norm_var, dream_similarity
```

**Analysis functions:**
- `compute_regression_slope(metric_name)` — is this metric trending?
- `detect_phase_transition(runs, window=20)` — detect qualitative jumps
- `detect_limit_cycle(runs, window=50)` — detect stabilized oscillation
- `plot_trajectory_summary()` — terminal-friendly summary of N runs

## Phase 3: Long-Run Generation Loop

### Modify: `demian/loop.py`

**Changes needed:**
1. Add `prompt_override=None` parameter to `generate_with_proprioception` — when None, use model's own last generated token as input (auto-regressive across runs, not just within runs)
2. Ensure injector state persists across runs (add `reset()` parameter, default False)
3. Add `max_total_tokens` parameter as a global budget instead of per-run limit

**Actually — simpler approach:**
Keep the existing `generate_with_proprioception` unchanged. The continuity engine just calls it repeatedly. Between runs:
- Clear the KV cache (start fresh attention context)
- Keep the dream state (accumulated self-knowledge)
- Inject the dream residual at step 0 of each new run
- Feed the model a neutral re-prompt or just its own last tokens

## Phase 4: Config Updates

### Add to `config.yaml`:
```yaml
continuity:
  enabled: false  # false = normal mode, true = continuity mode
  n_runs: 1000
  max_tokens_per_run: 512
  dream_weight: 0.003
  injection_scale: 0.005
  blend_mode: append
  inject_mode: one
  injection_damping: 0.7
  data_dir: data/trajectories
  dream_dir: data/dreams
```

## Phase 5: CLI Entry Point

### Modify: `demian.py`

Add `--continuity` flag:
```
python demian.py --continuity --prompt "Tell me about yourself" --runs 1000
```

When `--continuity` is set:
1. Initialize ContinuityRunner
2. Run n_runs with single prompt
3. Output running metrics summary every 10 runs
4. Save full trajectory data
5. Exit with summary statistics

## Implementation Order

1. `demian/metrics.py` — structural metrics collector (independent, can build first)
2. `demian/continuity.py` — ContinuityRunner class (depends on metrics.py and existing loop.py)
3. `config.yaml` updates (quick)
4. `demian.py` CLI updates (quick, depends on continuity.py)
5. Test with `--runs 10` first, then scale

## What to verify after implementation (before long runs)

- Dream state saves and loads correctly (round-trip test)
- Metrics CSV has one row per run with valid numbers
- Dream residual is non-zero and has proper shape (d_model=2048)
- First few runs match the behavior we've seen before (40-80 mode flips, energy 4.4→3.8 range)
- Dream influence grows gradually (dream_similarity should be near-zero for early runs, increase over time)
