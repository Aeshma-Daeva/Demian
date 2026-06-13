# Fixed-Point Internal Structure Paper Plan

Working title:

> Fixed Points Are Not Empty: Hidden Internal Structure Behind Apparently
> Static Recurrent Surfaces

Repository:

> <https://github.com/Aeshma-Daeva/Demian-Lab>

## Core Thesis

Apparently static surface behavior is not enough to diagnose a recurrent
system. A trajectory can be classified as `FIXED_POINT` at the exposed surface
while internal state channels continue to accumulate, separate, route, and
preserve continuation-relevant information.

The paper should not claim that every fixed point is rich. It should claim that
fixed-point labels are insufficient unless they are paired with internal-state
measurements, ablations, and resume controls.

## Clean Claim

Strong version:

> Across Demian recurrent-substrate experiments and matched baseline probes,
> surface-level fixed-point behavior repeatedly fails to capture internal-state
> structure. Interior classifications, message/carrier activity, channel
> separation, and capsule-resume controls reveal computational differences that
> are invisible from the surface label alone.

Conservative version for the abstract:

> We show that fixed-point surface classifications can hide nontrivial internal
> dynamics in recurrent systems, and we provide a measurement protocol for
> separating true collapse from internally structured fixed-point basins.

## What This Is Not

Do not frame the paper as:

- Demian beats RNNs, GRUs, LSTMs, Transformers, or Mamba.
- Fixed points are always good.
- Internal richness is consciousness, self-awareness, or intelligence.
- Gate-State Causal Propagation is fully proven.
- Demian v1 is a completed architecture.

Frame it as:

- a measurement paper;
- a warning against over-reading surface attractor labels;
- an artifact-backed methodology for asking whether a static surface hides
  active internal state.

## Evidence Backbone

### 1. Dual-GRU Fixed-Point Interiors

Primary artifact:

- `data/substrate_lab/dual_gru_family_summary.json`

Compact result:

| Substrate | Surface attractor | Interior classes | Key readout |
| --- | --- | ---: | --- |
| `dual_gru_v3b:current` | `FIXED_POINT` in 8/8 | 1 | accumulating fixed point in 8/8; mean message norm `23.1086`; bottleneck entropy `1.7779`. |
| `dual_gru_v3b:tight` | `FIXED_POINT` in 8/8 | 2 | tight and accumulating interiors under the same surface label. |
| `dual_gru_v3b:threshold` | `FIXED_POINT` in 8/8 | 2 | threshold variant splits into tight and accumulating interiors. |
| `dual_gru_v3` | `FIXED_POINT` in 8/8 | 2 | older dual-GRU also separates interior modes. |

Claim supported:

> Same surface attractor class, different internal basin classes.

### 2. Capsule Continuity: Surface-Only Resume Fails

Primary artifact:

- `data/substrate_lab/v9_capsule_continuity_20260511/summary.json`
- `docs/CAPSULE_CONTINUITY.md`

Compact result:

| Substrate | Full capsule mean gap | Surface-only mean gap | Interpretation |
| --- | ---: | ---: | --- |
| `demian_native_v9` | `0.0` | at least `0.2262` across sweep | full state resumes; surface alone does not. |
| `v9_five_channel` | `0.0` | at least `0.2746` across sweep | continuity spreads beyond the exposed surface. |

Claim supported:

> If a surface-only reconstruction cannot resume the trajectory but a full
> internal capsule can, the visible surface is not the full state of the
> computation.

### 3. Plain Recurrent And Native Baselines

Primary artifact:

- `data/diagnostics/gate_state_truth_campaign_20260516_3trackb_multisurgery/baseline_comparison_summary.json`

Compact result:

| Substrate | Final gap mean | Recovery-window gap mean | Runs |
| --- | ---: | ---: | ---: |
| `rnn` | `1.19e-7` | `1.11e-4` | 3 |
| `gru` | `5.18e-8` | `1.44e-4` | 3 |
| `lstm` | `4.46e-8` | `6.13e-5` | 3 |
| `diag_ssm` | `0.0650` | `0.00237` | 3 |
| `demian_native_v8` | `0.2206` | `0.00391` | 3 |
| `demian_native_v9` | `0.00322` | `0.000531` | 3 |

Use carefully:

- Plain RNN/GRU/LSTM baselines recover almost exactly under this pseudo-channel
  perturbation protocol.
- Native substrates show larger perturbation consequences.
- This is not a superiority claim; it is a comparison showing why matched
  baselines matter.

### 4. Transformer And Mamba Self-Reference Baselines

Primary artifacts:

- `data/reservoir_batch/batch_summary.json`
- `data/mamba_batch/mamba_batch_summary.json`
- `docs/CLAIMS.md` claims C1 and C2

Use as broader motivation, not the central proof:

- Transformer reservoir runs supported a deterministic period-2 attractor
  observation.
- Mamba self-reference runs landed in a lower-energy fixed-point-like basin
  rather than matching the transformer 2-cycle.
- These results show that surface recurrence primitives differ by architecture,
  so the paper should not treat all recurrence as one thing.

### 5. Gate-State Truth Campaign As A Negative Control

Primary artifacts:

- `data/diagnostics/gate_state_truth_campaign_20260516_3trackb_multisurgery/truth_campaign_summary.json`
- `data/diagnostics/gate_state_track_b_replication_summary_20260516_30/summary.json`

Use as scientific discipline:

- Track B generated interesting internal structure.
- The stricter truth campaign demoted the strong mechanism name when held-out
  strict-profile checks did not pass.
- This protects the fixed-point paper from overclaiming: the paper is about
  measurement and hidden structure, not settled causal mechanism naming.

## Proposed Contribution List

1. A surface/internal distinction for recurrent-system evaluation.
2. An `interior fixed-point` vocabulary separating static surface labels from
   hidden state organization.
3. A compact measurement protocol:
   - surface attractor class;
   - interior class;
   - channel separation;
   - route/channel norms;
   - ablation response;
   - full-state versus surface-only resume.
4. Evidence that several Demian fixed-point surfaces preserve distinct internal
   structure.
5. Matched baselines showing which effects are ordinary recurrent recovery and
   which require further mechanism tests.

## Draft Abstract

Recurrent systems are often summarized by their exposed trajectory: fixed
point, cycle, transient, or chaotic regime. This paper argues that such surface
labels can be misleading. In Demian, a structured recurrent-substrate research
program, we repeatedly observe apparently static surface behavior coexisting
with differentiated internal state. Dual-GRU and native-substrate experiments
show fixed-point surfaces that split into distinct basin interiors, including
tight and accumulating fixed-point classes. Capsule-continuity probes further
show that full internal-state restore can resume trajectories exactly while
surface-only restore fails, implying that continuation-relevant information is
not contained in the exposed surface alone. We compare these findings against
plain RNN, GRU, LSTM, diagonal SSM, Transformer-reservoir, and Mamba-reservoir
controls, and we use negative gate-state checks to bound mechanism claims. The
result is not a claim of benchmark superiority or a finished architecture, but
a measurement protocol: fixed-point behavior should be treated as a hypothesis
about surface dynamics, not as evidence that the internal computation is empty.

## Section Plan

1. Introduction
   - Problem: fixed-point labels can collapse meaningful internal differences.
   - Thesis: evaluate surface and internal state separately.
   - Contributions: measurement protocol and Demian evidence.

2. Background
   - Recurrent dynamics and attractor labels.
   - Why surface readouts are convenient but incomplete.
   - Relation to RNN mechanistic interpretability and dynamical analysis.

3. Methods
   - Substrates and baselines.
   - Surface attractor classification.
   - Interior class metrics.
   - Channel/route metrics.
   - Ablations and perturbations.
   - Capsule resume protocol.

4. Results A: Fixed-Point Basins With Different Interiors
   - Dual-GRU family table.
   - Tight versus accumulating fixed-point classes.
   - Message norms and bottleneck entropy.

5. Results B: Surface-Only State Is Not Enough
   - v9 and v9 five-channel capsule continuity.
   - Full capsule versus surface-only resume.
   - Component-only readouts.

6. Results C: Matched Baselines
   - RNN/GRU/LSTM/diag-SSM/native comparison.
   - Transformer versus Mamba self-reference as architecture-contrast context.

7. Negative Results And Claim Discipline
   - Gate-state truth-campaign demotion.
   - Why the paper does not claim solved mechanism or general superiority.

8. Discussion
   - Fixed point as surface class, not computational verdict.
   - Implications for recurrent-substrate design.
   - Why Demian v1 keeps explicit state channels.

9. Limitations
   - Custom metrics and vocabulary.
   - Limited seeds in some baselines.
   - Artifact heterogeneity across historical runs.
   - Some evidence is diagnostic rather than task-performance evidence.

10. Reproducibility
   - Link data index, commands, and exact artifacts.

## Figure And Table Plan

- Figure 1: Surface label versus internal-state measurement stack.
- Figure 2: Fixed-point basin interiors in dual-GRU variants.
- Figure 3: Capsule resume schematic: uninterrupted, full capsule,
  surface-only.
- Figure 4: Baseline comparison across RNN/GRU/LSTM/SSM/native substrates.
- Table 1: Definitions: surface fixed point, tight fixed point,
  accumulating fixed point, internal richness, channel separation.
- Table 2: Dual-GRU fixed-point interior summary.
- Table 3: Capsule-continuity resume results.
- Table 4: Baseline perturbation/recovery summary.
- Table 5: Negative checks and claim boundaries.

## Immediate Work Items

1. Generate a compact evidence CSV for the paper tables.
2. Decide whether Transformer/Mamba remain in the main paper or move to an
   appendix.
3. Re-run or validate the most important fixed-point interior artifacts from a
   clean command.
4. Create publication figures from existing SVG/JSON assets.
5. Convert this plan into `papers/fixed_point_internal_structure/main.tex`
   after the evidence tables are frozen.

