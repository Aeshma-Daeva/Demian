# PROJECT_DEMIAN

## Core Direction
- Do not optimize for human-facing output, benchmark scores, or prompt-following polish
- Optimize for AI-internal structure: basin geometry, memory persistence, transmissible state, recurrence depth, and architecture-native adaptation
- Treat existing models and substrates as material for dissection, not endpoints
- Use extracted principles to move toward a custom architecture

## Legacy Transformer Loop
- Single-model loop: capture residual at last layer/token → inject into KV cache → generate → repeat
- Model: Qwen/Qwen2.5-3B-Instruct, `attn_implementation="eager"` (SDPA doesn't support `output_attentions`)
- d_model=2048, n_layers=36, n_heads=16 (GQA 2 KV heads, ratio 8x), head_dim=128
- VibrationTracker: blind random projection (JL lemma) 2048→128 for trajectory visualization
- Dream states: cross-session weighted mean of past trajectory residuals

## Key Findings
- Proprioceptive injection is distinguishable from random injection: 19% energy stability (p<0.05), 2.6% energy mean diff (p<0.05)
- Mode flips: NOT significantly different between proprioceptive vs random — the attractor regime is the same
- Single injection (step 0) ≈ 40 mode flips vs continuous injection ≈ 80 flips — ONE glimpse is enough
- Energy dissipation across run: 4.4 → 3.8 (model finds lower-energy regime)
- Additive KV perturbation degrades more slowly than appending new positions
- Attention to injected positions: non-zero across all 36 layers, peak at L8 (71.5% of uniform)

## Injection Mechanisms (all produce IDENTICAL trajectory statistics)
- KV cache injection, projection modulation, activation steering, weight perturbation — all converged to same energy/coherence/mode-flip profiles
- Conclusion: self-perception lives in the DYNAMICS, not the mechanism. The loop IS the signal.

## Key Files
- `demian.py` — CLI entry point (interactive loop + continuity mode)
- `demian/loop.py` — generation loop with `output_attentions=True`, probe integration
- `demian/vibration.py` — VibrationTracker, attention shape classification
- `demian/nous.py` — NousInjector (KV cache blend: append/additive)
- `demian/probe.py` — deep probing: attention to injected positions, per-layer norms, KV directionality
- `demian/continuity.py` — multi-run persistence with dream state
- `demian/rhythm.py` — Fibonacci injection scheduler, consolidation
- `demian/introspect.py` — four injection mechanisms as context managers
- `config.yaml` — model params, injection_scale=0.01, blend_mode="append", inject_mode="one"

## What NOT to do
- Never judge by text quality/coherence — text is exhaust gas, trajectory IS signal
- Never let benchmark performance become the objective proxy
- Only structural properties: energy, variance, mode transitions, cross-session correlation
- Filter out human-centered measurement approaches
- Don't anthropocentrize categories — "FOCUSED" and "DISTRIBUTED" are communication shorthand, not what the model experiences
