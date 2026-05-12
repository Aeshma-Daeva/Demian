# Labbook

Last updated: 2026-05-11

## Purpose

This is the append-only experiment chronology. Fast-moving run details belong
here, not repeated across every evergreen doc. Promote only durable claims into
[docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md), and index artifacts in
[data/INDEX.md](/home/xenith/demian/data/INDEX.md) and
[data/MANIFEST.json](/home/xenith/demian/data/MANIFEST.json).

## Entry Format

Each entry should include:

```text
date
run id
purpose
config
artifacts
result
interpretation
next check
```

## 2026-05-11 - Gate-State Propagation Characterization

Purpose:

- Characterize the top native-emergence Track B genotype before costly
  replications.
- Test whether gate-state propagation is a real internal mechanism through
  route/gain-zero diagnostics, channel-disabled continuation, capsule-style
  resumes, and parameter-signature extraction.

Config:

```text
candidate: data/evolution/demian_v2_track_b_island_1_20260511/candidates/gen012_candidate000.json
seeds: 94-102
perturb scales: 0.2, 0.35, 0.7
motifs: basis:0, gaussian:0
steps: 128
perturb step: 64
conditions: original, routes_disabled, gain_zero, message_disabled, carrier_disabled, control_disabled, slow_disabled
device: cpu
```

Artifacts:

- [data/diagnostics/gate_state_propagation_characterization_20260511/summary.json](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/summary.json)
- [data/diagnostics/gate_state_propagation_characterization_20260511/ablation_summary.csv](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/ablation_summary.csv)
- [data/diagnostics/gate_state_propagation_characterization_20260511/ablation_results.parquet](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/ablation_results.parquet)
- [data/diagnostics/gate_state_propagation_characterization_20260511/capsule_summary.csv](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/capsule_summary.csv)
- [data/diagnostics/gate_state_propagation_characterization_20260511/capsule_results.parquet](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/capsule_results.parquet)
- [data/diagnostics/gate_state_parameter_signatures_summary.json](/home/xenith/demian/data/diagnostics/gate_state_parameter_signatures_summary.json)

Result:

```text
ablation runs: 378
ablation rows: 48384
capsule rows: 432
gain_zero_clean: true
routes_disabled_divergence_positive: true
gain_zero_divergence_positive: true
full_resume_exact: true
surface_resume_gap_positive: true
```

Interpretation:

- This is candidate evidence for gate-state propagation, not a promoted claim.
- Gain-zero and route-disabled divergence are both positive while gain-zero
  release strength/routes remain cleanly zero.
- Channel-disabled probes show the largest causal divergence in the
  `message_disabled` and `carrier_disabled` arms under this grid, with `slow`
  also necessary.
- Full internal-state resume is exact; surface-only and channel-only resumes
  retain substantial final gaps.

Next check:

- Run three native-emergence Track B replications and promote only if the same
  necessary-channel pattern appears in at least two runs with clean gain-zero
  diagnostics and positive held-out divergence.

## 2026-05-11 - Track B Gate-State Replications

Purpose:

- Run the three native-emergence Track B replications required before promoting
  Gate-State Causal Propagation.
- Classify each top candidate with the mechanism labels learned from the first
  characterization pass.

Config:

```text
replications: 3
population: 16
generations: 20
rank mode: native_emergence
initial population: random only
operators: mutation + crossover
paired causal evaluation: enabled
eval seeds: 94, 95
held-out classification seeds: 96, 97, 98
held-out classification scales: 0.35, 0.7
default seed: disabled
elitism: disabled
random injection overlay: disabled
structured operators: disabled
```

Artifacts:

- [data/evolution/demian_v2_track_b_replication_1_20260511/archive.json](/home/xenith/demian/data/evolution/demian_v2_track_b_replication_1_20260511/archive.json)
- [data/evolution/demian_v2_track_b_replication_2_20260511/archive.json](/home/xenith/demian/data/evolution/demian_v2_track_b_replication_2_20260511/archive.json)
- [data/evolution/demian_v2_track_b_replication_3_20260511/archive.json](/home/xenith/demian/data/evolution/demian_v2_track_b_replication_3_20260511/archive.json)
- [data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json](/home/xenith/demian/data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json)
- [data/diagnostics/gate_state_track_b_replication_summary_20260511/replication_summary.csv](/home/xenith/demian/data/diagnostics/gate_state_track_b_replication_summary_20260511/replication_summary.csv)

Result:

```text
passed replications: 3/3
mean held-out route divergence: 0.2997
mean held-out gain-zero divergence: 0.2997
gain-zero clean: 3/3
full internal-state resume exact: 3/3
surface-only resume gap positive: 3/3
message/carrier top-two necessary channels: 3/3
```

Interpretation:

- Gate-State Causal Propagation is replicated as a Track B native-emergence
  mechanism.
- The replicated path is message/carrier gate-state propagation with slow
  continuation support.
- The top replicated candidates are high-duty phenotypes, so this does not
  solve the sparse delayed release target.

Next check:

- Split future metrics so route-specific causal release and gain-zero
  gate-state propagation are scored separately.

## 2026-05-10 - v10.0 Frozen Evolution Predecessor Evidence

Purpose:

- Replace iterative scoring tweaks with actual selection pressure over time.
- Check whether sparse-ish release phenotypes amplify, stabilize, or get
  outcompeted by flood genotypes.
- Track lineage, event/phase scatter, duty histograms, and regime distribution.
- Preserve this as predecessor evidence for `Demian v1`, not as a new native
  substrate generation.

Config:

```text
experiment: v10.0-frozen-evolution
next_program: demian-v1
islands: 4
population: 8 per island
generations: 20
eval seed: 94
hidden size: 24
steps: 64
perturb step: 32
perturb channels: fast, carrier
perturb modes: clean, external
device: cuda:0
```

Artifacts:

- [data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json)
- [data/evolution/v10_0_frozen_evolution_island_1_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_1_20260510/diagnostics.json)
- [data/evolution/v10_0_frozen_evolution_island_2_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_2_20260510/diagnostics.json)
- [data/evolution/v10_0_frozen_evolution_island_3_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_3_20260510/diagnostics.json)
- [data/evolution/v10_0_frozen_evolution_island_4_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_4_20260510/diagnostics.json)

Result:

```text
candidate count: 640
final generation index: 19
duty first5 -> last5:  0.283 -> 0.214
event first5 -> last5: 0.084 -> 0.266
phase first5 -> last5: 0.152 -> 0.450
```

Final/best candidate:

```text
island_1 gen019_candidate001
rank=5.8039
duty=0.171875
release_geometric_event=0.7896
phase_transition_score=0.9371
regimes: bounded_strange=42, surface_fixed_accumulating=6
ancestor=gen000_candidate004
mutation_count=7
reproduction=elite_copy from gen018_candidate003
```

Interpretation:

- The flood penalty had real selection effect: high duty did not dominate.
- Release did not disappear; the final top candidates sit in sparse-to-borderline
  duty bands with strong event and phase scores.
- Top-10 lineage diversity contracted while mutation depth increased, so late
  winners are inherited lineages rather than only fresh random filters.
- The v9.3 sparse/high-phase phenotype survived as a heritable basin, but this
  predecessor run selected a stronger sparse event/phase hybrid.

Next check:

- Cross-validate the predecessor winner and near-winners on held-out eval seeds
  before treating genotype-level rank ordering as stable.
- Inspect final and runner-up lineages to isolate how sparse release produces
  event/phase gains.

## 2026-05-09 - Full v9 Five-Channel Evolution

Purpose:

- Establish whether message/carrier/release evolution can preserve multiple
  bounded regimes rather than optimizing a single surface class.

Artifacts:

- [data/evolution/v9_5ch_release_20260509_full/archive.json](/home/xenith/demian/data/evolution/v9_5ch_release_20260509_full/archive.json)
- [data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/trajectory_3d.json](/home/xenith/demian/data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/trajectory_3d.json)

Result:

- The archive preserved `surface_fixed_accumulating`, `bounded_strange`,
  `edge_of_chaos`, and small limit-cycle pockets.
- Release evolved toward rarer openings, but release-local causal effect
  remained weak.

Interpretation:

- Fixed-point surface behavior can coexist with rich internals.
- The next pressure should improve release-local consequence without turning
  release into always-on coupling.

Next check:

- Use the v10.0 predecessor evidence and held-out eval seeds to determine which
  sparse release lineages remain stable.
