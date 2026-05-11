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
