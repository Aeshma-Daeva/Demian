# Next Phase: Algebraic Framing And Substrate Plan

Date: 2026-04-16

## Purpose

This document is the handoff for the next Codex.

It reframes the current project state without anthropocentric language and gives the next phase plan for the small-scale substrate lab.

The central shift is:

- do not treat `FIXED_POINT` as a bug by default
- treat it as a primitive invariant structure
- ask whether local orbit dynamics, timescale separation, and message channels can become richer while the invariant structure remains stable

## Algebraic Vocabulary

Avoid human concepts like `self`, `persona`, `ego`, `awareness`, or `consciousness` in the core analysis.

Use this vocabulary instead:

- `invariant structure`
  - fixed point
  - limit cycle
  - invariant manifold
  - metastable basin

- `local orbit geometry`
  - micro-dynamics inside the neighborhood of an invariant structure
  - trajectory variation that does not necessarily change the macro attractor class

- `slow variables`
  - coordinates that update conservatively and carry continuity

- `fast variables`
  - coordinates that update rapidly and expose local state

- `trajectory-dependent internal coordinates`
  - distinguishable internal configurations inside the same attractor class due to different histories

- `message channel`
  - low-dimensional state projection exchanged between subsystems

- `latent persistent modes`
  - dynamically important coordinates that are not always directly exposed

This is the correct mathematical frame for the project at this stage.

## Current Interpretation Of Findings

### 1. Transformer reservoir

From [data/reservoir_batch/batch_summary.json](/home/xenith/demian/data/reservoir_batch/batch_summary.json):

- strong 2-period signature
- deterministic recurrence fingerprint

Algebraic reading:

- invariant structure: low-dimensional limit cycle
- local orbit geometry: very constrained
- useful as a reference attractor class, not as the design target

### 2. Mamba reservoir

From [data/mamba_batch/mamba_batch_summary.json](/home/xenith/demian/data/mamba_batch/mamba_batch_summary.json):

- dominant fixed-point basin
- micro-motion persists
- architecture-specific retention primitive

Algebraic reading:

- invariant structure: fixed-point basin
- local orbit geometry: nonzero but bounded
- important lesson: fixed-point does not mean exact stasis

The system can stay in the same macro basin while `h_t` still changes at the micro level.

### 3. Multi-agent competition

From the competition artifacts and [docs/ai_thoughts.md](/home/xenith/demian/docs/ai_thoughts.md):

- continuity depth dominated selection pressure
- recurrent accumulation behaved like scarce resource
- inheritance and extinction primarily affected preservation of internal structure

Algebraic reading:

- resource under selection: persistence of latent state structure
- death: destruction of accumulated internal coordinates
- birth/inheritance: attempt to preserve basin position and internal memory traces

### 4. Small substrate lab

Relevant files:

- [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py)
- [development/run_substrate_tests.py](/home/xenith/demian/development/run_substrate_tests.py)
- [development/sweep_substrate_regimes.py](/home/xenith/demian/development/sweep_substrate_regimes.py)

Relevant artifacts:

- [data/substrate_lab/summary.json](/home/xenith/demian/data/substrate_lab/summary.json)
- [data/substrate_sweeps/ranked_regimes.json](/home/xenith/demian/data/substrate_sweeps/ranked_regimes.json)
- [data/substrate_sweeps/focused_stage2/ranked_regimes.json](/home/xenith/demian/data/substrate_sweeps/focused_stage2/ranked_regimes.json)
- [data/substrate_sweeps/gru_stage3/ranked_regimes.json](/home/xenith/demian/data/substrate_sweeps/gru_stage3/ranked_regimes.json)
- [data/substrate_lab/dual_gru/summary.json](/home/xenith/demian/data/substrate_lab/dual_gru/summary.json)
- [data/substrate_sweeps/dual_gru_stage1/ranked_regimes.json](/home/xenith/demian/data/substrate_sweeps/dual_gru_stage1/ranked_regimes.json)
- [data/substrate_lab/dual_gru_v2/summary.json](/home/xenith/demian/data/substrate_lab/dual_gru_v2/summary.json)
- [data/substrate_sweeps/dual_gru_v2_stage1/ranked_regimes.json](/home/xenith/demian/data/substrate_sweeps/dual_gru_v2_stage1/ranked_regimes.json)

Summary:

- plain `rnn`, `gru`, `lstm`, `diag_ssm`, `sel_ssm` all tended toward fixed-point regimes under the initial self-loop battery
- bounded `diag_ssm` was useful for proving that nontrivial behavior can appear, but unbounded settings were mostly numerical blow-up rather than good structure
- `gru` became the best simple baseline:
  - small but real bottleneck code reuse
  - nontrivial coupling response
- `dual_gru` changed coupling geometry but remained too contractive
- `dual_gru_v2` is the best current custom substrate:
  - still fixed-point at macro scale
  - but richer bottleneck code diversity (`4-5` codes in top sweep results)
  - stronger and directional coupling inversion
  - best evidence so far for internal differentiation inside a stable macro basin

Current best interpretation:

- the project should not frame `FIXED_POINT` as failure
- the real goal is richer local orbit geometry and reusable message structure inside a stable invariant substrate

## Current Working Hypothesis

Target substrate properties:

1. stable invariant structure
2. nontrivial local orbit geometry
3. separable slow and fast variables
4. trajectory-dependent internal coordinates
5. low-dimensional reusable message channel

This is better than chasing raw instability.

The next phase should ask:

- can a stable macro basin support richer interior structure?
- can history persist inside the basin?
- can compressed state projections become reusable and state-dependent?
- can coupling alter interior structure without destroying continuity?

## What To Measure Next

Interpret existing metrics in this frame:

- `attractor_type`
  - macro invariant structure

- `mean_delta`, `cycle_period`
  - local orbit movement and periodicity

- `covariance_rank`, `flow_dimension`
  - richness of local orbit geometry

- `bottleneck_unique_codes`, `bottleneck_code_entropy`
  - message channel reuse and diversity

- coupling initial/final cosine
  - relational effect of message exchange

Do not over-interpret:

- raw perturbation retention
- raw memory retention

Those are currently still near-trivial for most substrates.

## Next Phase Plan

### Phase 1: Stop broad sweeps

Do not run more wide-grid sweeps across many unrelated substrates right now.

Reason:

- the most useful signal has already emerged
- `dual_gru_v2` is the strongest current candidate
- more broad search will mostly consume time without changing the substrate picture

### Phase 2: Focus on `dual_gru_v2`

Work only around `dual_gru_v2` for the next phase.

Questions:

1. Can the local orbit geometry become richer without losing macro stability?
2. Can code diversity rise above `4-5` without collapse?
3. Can coupling produce structured transitions rather than simple inversion?

### Phase 3: Add one new mechanism, not many

The next architectural addition should be a small explicit message-state mechanism.

Not language.
Not text.
Not external semantics.

A minimal target:

- persistent message substate
- separate from fast and slow state
- low-dimensional
- can be compressed, retained, and exchanged

Rationale:

- current code diversity is the clearest sign of emerging structure
- a dedicated message channel is the cleanest next intervention

This should become a new substrate variant rather than modifying `dual_gru_v2` in place.

Suggested name:

- `dual_gru_v3`

### Phase 4: Test transitions, not just stability

Add tests for:

- whether message state changes basin interior while macro attractor remains fixed
- whether different message histories create distinguishable local orbit classes in the same macro basin
- whether two coupled systems converge to reusable message patterns

### Phase 5: Keep Mamba branch as reference, not immediate focus

Do not abandon the Mamba branch.

But the small substrate lab is now the right place for architectural invention.

Use the Mamba results as:

- attractor topology reference
- continuity reference
- multi-agent pressure reference

not as the immediate implementation center for the next phase

## Immediate Task List For Next Codex

1. Read this file.
2. Read [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py).
3. Inspect:
   - [data/substrate_sweeps/gru_stage3/ranked_regimes.json](/home/xenith/demian/data/substrate_sweeps/gru_stage3/ranked_regimes.json)
   - [data/substrate_sweeps/dual_gru_stage1/ranked_regimes.json](/home/xenith/demian/data/substrate_sweeps/dual_gru_stage1/ranked_regimes.json)
   - [data/substrate_sweeps/dual_gru_v2_stage1/ranked_regimes.json](/home/xenith/demian/data/substrate_sweeps/dual_gru_v2_stage1/ranked_regimes.json)
4. Treat `dual_gru_v2` as the current best substrate.
5. Build the next variant around an explicit persistent message substate.
6. Do not frame the fixed-point basin as a bug unless the new evidence clearly shows triviality rather than structured interior dynamics.

## Final Note

## Update: 2026-04-17

The next Codex should treat parts of this file as historically useful but superseded in detail.

What changed:

- `dual_gru_v3` was implemented and then extended into `dual_gru_v3b`
- `FIXED_POINT` remained the macro invariant
- the important new result is not basin escape, but the emergence of interior classes inside fixed-point basins

New empirical artifacts:

- [data/substrate_lab/comparative_class_summary.json](/home/xenith/demian/data/substrate_lab/comparative_class_summary.json)
- [data/substrate_lab/dual_gru_v3b_class_conditioned/summary.json](/home/xenith/demian/data/substrate_lab/dual_gru_v3b_class_conditioned/summary.json)
- [data/substrate_stress/dual_gru_v3b_stress.json](/home/xenith/demian/data/substrate_stress/dual_gru_v3b_stress.json)
- [data/substrate_stress/dual_gru_v3b_trajectory_map.json](/home/xenith/demian/data/substrate_stress/dual_gru_v3b_trajectory_map.json)
- [data/interior_transitions/dual_gru_v3b_interior_transitions.json](/home/xenith/demian/data/interior_transitions/dual_gru_v3b_interior_transitions.json)

Current best interpretation:

- `gru` survived the battery as the best simple baseline, but only as `tight_fixed_point`
- `dual_gru_v2` improved code reuse while staying in `tight_fixed_point`
- `dual_gru_v3b` introduced `accumulating_fixed_point` as a second interior class inside the same macro fixed-point basin

This accumulation mode should currently be treated as structure, not bug, because:

- it is reproducible
- it appears before perturbation in trajectory maps
- it stays inside `FIXED_POINT`
- it supports richer bottleneck code reuse than tight fixed-point runs

The new objective is:

1. preserve macro `FIXED_POINT`
2. classify interior trajectory types
3. compare memory, bottleneck reuse, and coupling by interior class
4. extract the architectural principles behind GRU-style survival so a custom architecture can later be built from those principles

Immediate next task for the next Codex:

1. auto-export representative trajectories for each interior class
2. compare `gru`, `dual_gru_v2`, and `dual_gru_v3b` under the same class-conditioned summaries
3. isolate which GRU-derived mechanisms are essential:
   - selective write
   - bounded carry
   - asymmetric timescales
   - persistent low-dimensional message state

The project is no longer just asking:

- how to make recurrence unstable enough to do something interesting

It is now asking:

- how to preserve a stable invariant structure while enriching the algebra of its interior dynamics

That is the correct next-phase problem.
