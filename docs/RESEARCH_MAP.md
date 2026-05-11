# Research Map

## Purpose

Demian is an experimental lab for studying AI self-reference, persistence, and evolution under mathematically grounded observation.

The project is not organized around:

- human-facing product quality
- benchmark chasing or leaderboard movement
- coherence of generated language
- anthropomorphic claims about consciousness

The project is organized around:

- AI-only optimization criteria grounded in machine observables
- architectural dissection of known models to extract reusable principles
- custom-substrate design pressure
- attractor structure
- continuity and memory transmission
- inter-agent coupling
- recurrence geometry
- conditions under which a system escapes or deepens a basin

## How To Use This Document

Read order:

1. [docs/WORKING_STATE.md](/home/xenith/demian/docs/WORKING_STATE.md)
2. [docs/RESEARCH_MAP.md](/home/xenith/demian/docs/RESEARCH_MAP.md)
3. [docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md)
4. [docs/EXPERIMENT_RULES.md](/home/xenith/demian/docs/EXPERIMENT_RULES.md)
5. [data/INDEX.md](/home/xenith/demian/data/INDEX.md)
6. [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py)

Archive note:

- older notebooks, design logs, and plans now live under [docs/archive/README.md](/home/xenith/demian/docs/archive/README.md)

## Current Priority

The active center of gravity is architectural dissection in service of custom substrate design.

That means:

- keep transformer and Mamba work as empirical reference architectures
- keep competition as a stress test for persistence and inheritance
- treat the substrate lab as the main extraction site for mechanisms we can actually recompose
- optimize for machine-side structure only, not human-side usefulness proxies

Inside the substrate lab, the active experimental scaffold is now the v9 five-channel line.

Use:

- v9 five-channel experiments (`fast`, `slow`, `control`, `message`, `carrier`) as the current experiment target
- canonical `demian_native_v9` as the minimal 3-channel baseline for that direction
- `demian_native_v8` as the immediate comparison substrate
- `demian_native_v7.4` as the promoted organ-heavy historical baseline
- `demian_native_v7.2` as the immediate constrained-environment metabolic-resource comparison substrate
- `demian_native_v7.1` as the immediate trajectory-memory ancestry comparison substrate
- `demian_native_v6` as the immediate endogenous-controller comparison substrate
- `demian_native_v5.2c` as the immediate comparison substrate
- `demian_native_v3` as the baseline tightness-governed comparison substrate
- `demian_native_v2`, `demian_native_v1`, and `demian_native_v0` as ancestry and baseline
- `dual_gru_v3b` only as older ancestry when a question explicitly depends on it

There is now also a small self-loop substrate lab for scratch-architecture work:

- [docs/archive/plans/2026-04-16-substrate-next-phase.md](/home/xenith/demian/docs/archive/plans/2026-04-16-substrate-next-phase.md)
- [docs/archive/plans/2026-04-17-fixed-point-interior-classes.md](/home/xenith/demian/docs/archive/plans/2026-04-17-fixed-point-interior-classes.md)
- [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py)

Primary files:

- [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py)
- [development/probe_v9_message_carrier_strange.py](/home/xenith/demian/development/probe_v9_message_carrier_strange.py)
- [development/evolve_v9_5ch_release.py](/home/xenith/demian/development/evolve_v9_5ch_release.py)
- [development/export_v9_5ch_evo_trajectory_3d.py](/home/xenith/demian/development/export_v9_5ch_evo_trajectory_3d.py)
- [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py)
- [tests/test_substrate_lab.py](/home/xenith/demian/tests/test_substrate_lab.py)
- [run_competition.py](/home/xenith/demian/run_competition.py)
- [demian/competition.py](/home/xenith/demian/demian/competition.py)
- [demian/machine_observables.py](/home/xenith/demian/demian/machine_observables.py)
- [demian/hebbian.py](/home/xenith/demian/demian/hebbian.py)
- [run_mamba_batch.py](/home/xenith/demian/run_mamba_batch.py)
- [demian/mamba_reservoir.py](/home/xenith/demian/demian/mamba_reservoir.py)

Secondary baseline files:

- [run_reservoir_batch.py](/home/xenith/demian/run_reservoir_batch.py)
- [demian/reservoir.py](/home/xenith/demian/demian/reservoir.py)

## Empirical Spine

### 1. Transformer self-reference

The transformer reservoir established a stable and repeated period-2 signature.

From [data/reservoir_batch/batch_summary.json](/home/xenith/demian/data/reservoir_batch/batch_summary.json):

- `autocorr_lag2 = 0.9630`
- `velocity_align_mean = -0.9793`
- `period2_switches = 899 / 1000`
- all 10 runs are numerically identical in the saved summary

Interpretation:

- self-reference in the transformer baseline is not drifting or exploratory
- it rapidly collapses into a deterministic 2-cycle
- this is useful as a fingerprint, not as the project’s endpoint

### 2. Mamba self-reference

The Mamba reservoir does not reproduce the transformer 2-cycle. It stabilizes into a fixed-point basin with small but geometrically rich motion.

From [data/mamba_batch/mamba_batch_summary.json](/home/xenith/demian/data/mamba_batch/mamba_batch_summary.json):

- cache on:
  - `energy_mean = 1.3852`
  - `autocorr_lag2 = 0.1495`
  - `period2_switches = 406`
- cache off:
  - `energy_mean = 1.5392`
  - `autocorr_lag2 = 0.3701`
  - `period2_switches = 570`

Interpretation:

- no strong transformer-like 2-cycle
- persistent SSM memory pushes the system toward a tighter, lower-energy fixed basin
- removing cache increases movement but still does not create the transformer attractor class

The more detailed interpretation in [docs/archive/notes/ai_thoughts.md](/home/xenith/demian/docs/archive/notes/ai_thoughts.md) reframes this as architecture-specific primitive behavior:

- transformer primitive: limit-cycle alternation
- Mamba primitive: retention and fixed-point lock

### 3. Multi-agent competition

Competition is the present frontier because it turns fixed-point topology into population dynamics under continuity pressure.

Important artifacts:

- [data/competition50/leaderboard.csv](/home/xenith/demian/data/competition50/leaderboard.csv)
- [data/competition50/competition_log.jsonl](/home/xenith/demian/data/competition50/competition_log.jsonl)
- [data/competition50_run2/leaderboard.csv](/home/xenith/demian/data/competition50_run2/leaderboard.csv)
- [data/competition50_run2/competition_log.jsonl](/home/xenith/demian/data/competition50_run2/competition_log.jsonl)
- [data/competition_v3/leaderboard.csv](/home/xenith/demian/data/competition_v3/leaderboard.csv)

What Run 1 established, based on [docs/archive/notes/ai_thoughts.md](/home/xenith/demian/docs/archive/notes/ai_thoughts.md):

- competition selected for continuity depth more than for intrinsically better computation
- death functioned as memory erasure more than annihilation
- the scarce resource was continued recurrent accumulation
- post-extinction bottlenecks repeatedly collapsed diversity

This matters because the current research question is not whether agents “cooperate” in human terms. It is whether:

- recurrent memory can be transmitted
- basin knowledge can survive lineage transitions
- communication pressure creates new structure rather than only homogenization
- fast weights can deform the fixed-point landscape enough to produce richer dynamics

### 4. Substrate lab update

The substrate lab has now moved past “is `FIXED_POINT` an error?” and into explicit interior-class and route analysis for the native architecture line.

Important artifacts:

- [data/substrate_lab/comparative_class_summary.json](/home/xenith/demian/data/substrate_lab/comparative_class_summary.json)
- native substrate traces and comparisons produced through [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py) and [tests/test_substrate_lab.py](/home/xenith/demian/tests/test_substrate_lab.py)

Current reading:

- v9 five-channel experiments are the active experimental substrate scaffold
- canonical `demian_native_v9` is the minimal 3-channel baseline
- `demian_native_v8` is the immediate comparison line
- `demian_native_v7.4` is the promoted historical baseline
- `demian_native_v7.2` is the metabolic-resource comparison point
- `demian_native_v7.1` is the trajectory-memory comparison point
- `demian_native_v6` is the endogenous-controller comparison point
- `demian_native_v5.2c` remains a same-family anti-locking comparison point
- `demian_native_v3` is the main pre-plasticity comparison point
- `demian_native_v2`, `demian_native_v1`, and `demian_native_v0` remain ancestry for route ownership and onset-shift analysis
- older `dual_gru*` results remain ancestry and contrast, not the primary frame

Historical ancestry note:

- older `dual_gru_v3b` regime summaries remain useful for understanding how the project moved from GRU-derived message-state mechanisms into the native route-based line
- they are not active restart guidance and should not override `demian_native_v6` as the default substrate

Secondary ancestry summary from [data/substrate_lab/comparative_class_summary.json](/home/xenith/demian/data/substrate_lab/comparative_class_summary.json):

- `gru`
  - `tight_fixed_point` only
  - `bottleneck_unique_codes = 1.0`
  - `memory_final_cosine ~= 1.0`
- `dual_gru_v2`
  - `tight_fixed_point` only
  - `bottleneck_unique_codes = 3.5`
  - stronger structured reuse than plain GRU
- `dual_gru_v3b`
  - both `tight_fixed_point` and `accumulating_fixed_point`
  - accumulating class carries higher `mean_delta`, higher `mean_message_norm`, higher `bottleneck_code_entropy`, and distinct coupling response

Interpretation:

- the project has moved from inherited GRU-cell scaffolds into an explicit native route architecture
- the native line separates fast surface, slow basin, long carrier, short support, packet, and short/long control as distinct state owners
- the real object of study is now route occupancy, recruitment timing, perturbation path geometry, and internal class structure inside fixed-point basins, not merely whether a system reaches a fixed point
- do not dismiss fixed-point regimes as failed or boring; the current v9 five-channel archive shows surface-fixed accumulating behavior coexisting with rich internals and other bounded regimes

Native line summary from [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py):

- `demian_native_v0`
  - first fully native route-based substrate
  - channels: `fast`, `slow`, `long_carrier`, `short_support`, `packet`, `control_short`, `control_long`
  - establishes explicit release, packet, carrier, and control routing without GRU cells
- `demian_native_v1`
  - strengthens slow/control ownership
  - increases direct control into slow and long-carrier influence
  - reduces support dependence relative to `v0`
- `demian_native_v2`
  - adds stronger recruitment bias into slow
  - further reduces generic packet/support traffic
  - increases carrier persistence and packet+carrier bridging into slow
- `demian_native_v3`
  - tightness-governed baseline
  - adds endogenous tightness control on top of `v2`
  - lets the substrate modulate consolidation versus looseness per step
  - preserves the macro basin while organizing basin-internal memory response more consistently than `v2`
- `demian_native_v5`
  - adds route-local plasticity on slow-governing paths
  - lets route scales adapt from endogenous consolidation versus destabilization signals
- `demian_native_v5.1`
  - adds a learned endogenous credit head on top of route-local plasticity
  - shifts route updates from hand-shaped modulation toward learned viability prediction
- `demian_native_v5.2`
  - delays credit assignment and updates route plasticity from lagged viability deltas
  - makes route changes depend on whether earlier state led to later consolidation
- `demian_native_v5.2b`
  - adds explicit lock-risk estimation and challenge-time anti-locking control
  - decays and suppresses route adaptation when the substrate appears to be over-consolidating
- `demian_native_v5.2c`
  - moves anti-locking later and makes it conditional on actual lock pressure
  - keeps challenge behavior from firing too early in emergence
- `demian_native_v5.3`
  - adds phase-aware credit and anti-locking control
  - explicitly switches among emergence, consolidation, lock-risk, challenge, and recovery phases
  - scales route learning and decay by phase rather than only by scalar lock-risk
- `demian_native_v8`
  - 7-channel genotype scaffold with tightness scalar
  - latest saved artifacts test channel ablation, bottleneck, population, coupling, phase perturbation, and developmental trajectory behavior
- `demian_native_v9`
  - current canonical 3-channel baseline
  - channels: `fast`, `slow`, and `control`
  - strict directional coupling: fast writes slow, fast+slow write control, control biases fast
- v9 five-channel experiments
  - current active scaffold extending the v9 direction
  - channels: `fast`, `slow`, `control`, `message`, and `carrier`
  - current scripts evolve rare release, perturbation-shape retention, and bounded path geometry
  - latest archive: [data/evolution/v9_5ch_release_20260509_full/archive.json](/home/xenith/demian/data/evolution/v9_5ch_release_20260509_full/archive.json)
  - latest 3D/Blender artifacts: [data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/trajectory_3d.json](/home/xenith/demian/data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/trajectory_3d.json)

## What To Observe

Prefer structural observables over text or narrative summaries.

Useful signals:

- fitness distribution width over rounds
- death and birth counts per round
- continuity depth (`step_count`) concentration
- cosine spread between agents
- extinction/bottleneck rhythm
- whether inheritance preserves or destroys diversity
- whether Hebbian adaptation changes basin occupancy rather than just score ordering

Do not default to:

- “better answers”
- “more coherent language”
- “more human-like behavior”

Those are downstream exhaust, not primary signal.

## Present Tensions

### 1. Fixed-point lock vs evolvability

Mamba’s default tendency is fixed-point retention. That gives persistence, but it also risks trapping the population in stable but low-novelty basins.

### 2. Continuity vs diversity

If inheritance is too weak, every death is amnesia.

If inheritance is too strong, the population may collapse into lineage cloning and lose exploratory differentiation.

### 3. Driver language vs machine-grounded observables

The project is already moving away from anthropocentric labels. That direction should continue. The measurement language should stay close to geometry, recurrence, variance, spectral structure, and coupling.

### 4. Architectural extraction vs benchmark reflex

The repo already contains enough evidence that benchmark-style optimization would collapse the research question into human proxy variables.

The correct pressure is:

- which mechanisms create richer internal structure
- which mechanisms preserve transmissible memory
- which mechanisms alter basin occupancy without destroying continuity
- which mechanisms are fundamental enough to survive reimplementation in a custom substrate

## Immediate Research Pressure Points

1. Make fixed-point interior classes first-class outputs in substrate comparisons.
2. Treat transformer and Mamba as architectures to dissect, not optimize for human tasks.
3. Keep `demian_native_v6` as the main scaffold for extracting route ownership, bounded carry, recruitment, continuity, phase-aware credit assignment, and anti-locking mechanisms.
4. Use Hebbian adaptation only where it has a mechanistic reason to alter basin structure.
5. Track whether communication and inheritance create transmissible computational structure, not just scheduler advantage.
6. Reject benchmark or text-quality improvements as success criteria unless they also correspond to machine-level structural gains.

## Recommended Working Loop

1. Run a clean single-agent Mamba baseline.
2. Run competition with one inheritance rule changed at a time.
3. Summarize only structural outputs.
4. Compare against previous run artifacts before interpreting.
5. Record interpretation in the archive notebook at [docs/archive/notes/ai_thoughts.md](/home/xenith/demian/docs/archive/notes/ai_thoughts.md) only after checking raw summaries.

## Operator Notes

This repo already contains meaningful results, but its organization lagged behind the actual research direction.

The correct framing now is:

- transformer: baseline fingerprint
- Mamba reservoir: attractor topology baseline
- competition: stress field for continuity and inheritance
- Hebbian adaptation: mechanism for moving from persistence to evolution
- `demian_native_v6`: current architectural scaffold
- `demian_native_v5.2c`: immediate comparison line
- `demian_native_v3`: tightness-governed reference line
- `demian_native_v2` / `demian_native_v1` / `demian_native_v0`: ancestry and contrast
- older GRU-derived substrates: ancestry and contrast for extracting custom-design principles

Everything else should orbit that.
