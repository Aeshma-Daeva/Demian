# Demian Technical Report

Last updated: 2026-05-12

Working title: **Demian: Discovering Native Mechanisms in Structured Recurrent
Substrates**

Public repository: <https://github.com/Aeshma-Daeva/Demian-Lab>

## Abstract

Demian began as an attempt to build a native recurrent substrate: a system whose
state owners, routes, and memory dynamics are designed directly rather than
inherited from a standard recurrent cell. The project became a lab methodology
because building that substrate required reliable ways to observe internal
dynamics, falsify candidate mechanisms, and preserve both failed and successful
design paths.

This report presents the current publication split. GitHub is the living lab:
artifact-indexed, historically complete, and explicit about failures. The first
arXiv paper should be narrower: a methods-first statement centered on one
replicated finding, **Gate-State Causal Propagation**, in a structured
five-channel recurrent substrate.

## Introduction

The initial goal was not to optimize chatbot outputs or claim benchmark
superiority. The goal was to build a custom substrate capable of sustaining
machine-internal recurrence, memory, and routing dynamics. Direct optimization
and surface behavior were repeatedly misleading: stable or fixed exposed vectors
could hide internal channel structure, while attractive target metrics could be
met by fragile or non-causal shortcuts.

The response was methodological. Demian now treats substrate-building as an
experimental program:

- define factorized state and routes explicitly;
- perturb, ablate, and resume trajectories rather than judging only final
  outputs;
- classify regimes and preserve negative evidence;
- promote claims only after artifact-backed replication or focused
  falsification.

The strongest current result is [Gate-State Causal Propagation](NATIVE_MECHANISMS.md#gate-state-causal-propagation),
recorded in [CLAIMS.md C20](CLAIMS.md#c20-gate-state-causal-propagation-replicates-in-track-b-native-emergence-runs).
It replicated in 3/3 Track B native-emergence runs under the current protocol.

## Background And Framing

Demian uses custom vocabulary because much of the work concerns internal
mechanism discovery rather than task accuracy. The corresponding standard
terms are listed in [GLOSSARY.md](GLOSSARY.md). In brief:

- a substrate is a recurrent dynamical system;
- channels are factorized recurrent state components;
- routes are learned or fixed transformations between state components;
- a surface is the observable readout vector;
- regimes are attractor or trajectory classes;
- capsule continuity compares full internal-state resume against surface-only
  replay.

The central discipline is to avoid over-reading surface labels. A trajectory
with fixed readout can still contain structured internal dynamics. Conversely,
a promising target score can be rejected if ablations show weak causal support.

## Substrate Under Study

The current study substrate is the v9 five-channel scaffold, with state
components:

- `fast`: exposed/readout-adjacent state;
- `slow`: persistent accumulator;
- `control`: lower-dimensional route-control state;
- `message`: intermediate routed signal;
- `carrier`: slower routed signal that supports delayed propagation.

The scaffold extends canonical three-channel `demian_native_v9`; it is not the
final Demian v1 architecture. Its purpose is to expose route, channel, release,
and capsule behavior clearly enough to discover and falsify native mechanisms.

Main code and artifacts:

- v9 five-channel probe and release workbench:
  [development/probe_v9_message_carrier_strange.py](../development/probe_v9_message_carrier_strange.py)
  and [development/evolve_v9_5ch_release.py](../development/evolve_v9_5ch_release.py)
- artifact map: [data/INDEX.md](../data/INDEX.md)
- anatomy and visual readouts: [SUBSTRATE_ANATOMY.md](SUBSTRATE_ANATOMY.md)

## Methodology

Demian uses a dual-track search and validation pattern.

**Track A: engineered targets.** These runs target explicit properties such as
sparse or delayed release. They are useful for probing the search space but can
select brittle shortcuts.

**Track B: native emergence.** These runs favor internally useful mechanisms
even when they do not satisfy the engineered sparse/timed target. The current
main finding came from this track.

The core methodological tools are:

- multi-objective evolutionary search over scaffold parameters and route
  settings;
- causal ablations including `routes_disabled`, `gain_zero`, and per-channel
  disabled interventions;
- capsule continuity, comparing uninterrupted continuation, full internal-state
  resume, surface-only resume, and channel-only resumes;
- attractor/regime classification from machine observables;
- evidence gates that separate observation, inference, and speculation.

The live claim ledger is [CLAIMS.md](CLAIMS.md). The mechanism catalog is
[NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md).

## Results

### Gate-State Causal Propagation

Gate-State Causal Propagation is the current publishable scientific finding. In
the focused Track B characterization and replication artifacts, top candidates
preserved positive divergence under route-disabled and gain-zero comparisons,
while gain-zero diagnostics remained clean. In the three Track B replications:

- the mechanism gate passed in 3/3 runs;
- held-out route-disabled and gain-zero divergence were positive in 3/3;
- mean held-out divergence was recorded as `0.2997`;
- `message` and `carrier` were the top two necessary channels in 3/3;
- full internal-state resume was exact in 3/3;
- surface-only resume left positive gaps in 3/3.

Primary artifacts:

- [data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json](../data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json)
- [data/diagnostics/gate_state_propagation_characterization_20260511/summary.json](../data/diagnostics/gate_state_propagation_characterization_20260511/summary.json)
- [NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md#gate-state-causal-propagation)

This is not a sparse-release claim. The replicated Track B candidates are
high-duty native-emergence phenotypes.

### Capsule Continuity

The capsule-continuity probe shows that canonical v9 and v9 five-channel resume
from full internal state while surface-only replay fails under the deterministic
probe. This supports the methodological claim that internal state, not only
surface output, is the computation object.

Primary artifact:

- [data/substrate_lab/v9_capsule_continuity_20260511/summary.json](../data/substrate_lab/v9_capsule_continuity_20260511/summary.json)

This is a full-state resume result, not a compressed-memory result.

### Fixed Surface, Structured Internals

The v9 five-channel archive preserved multiple bounded regimes, including
`surface_fixed_accumulating`, `bounded_strange`, `edge_of_chaos`, and small
limit-cycle pockets. The conservative claim is that fixed-point surface
behavior is not sufficient to classify a substrate as dynamically trivial.

Primary artifact:

- [data/evolution/v9_5ch_release_20260509_full/archive.json](../data/evolution/v9_5ch_release_20260509_full/archive.json)

## Negative Results

Several important paths should be preserved as negative evidence.

- Sparse release did not yet generalize into a stable cross-seed result.
- The v10.0 predecessor was not backend/held-out stable. CPU reruns did not
  exactly reproduce archived CUDA metrics, held-out release duty flooded, and
  release-route-zero ablation barely changed held-out behavior.
- Dynamic selection preserved causality in some checks but failed the sparse
  and timed criteria that originally motivated it.
- `bounded_strange` appeared as an inspection regime, but it should not be
  overclaimed as stable adaptation or formal chaos.

These negatives are part of the result, because they shaped the move away from
release-vector-centric design.

## Synthesis: Demian v1

Demian v1 is the next design consequence, not a validated finished result. The
prototype makes gate state explicit:

- `gate` becomes a first-class internal channel;
- `message` and `carrier` remain preserved because the replicated mechanism
  repeatedly depended on them;
- surface readout is treated as output, while internal state is treated as the
  computation substrate;
- release-open duty is retired as the primary causal mechanism.

Implementation and tests:

- [development/demian_v1_gate_state.py](../development/demian_v1_gate_state.py)
- [tests/test_demian_v1_gate_state.py](../tests/test_demian_v1_gate_state.py)

## Limitations

- The architecture family is small.
- The vocabulary is custom and must be translated carefully.
- The work is CPU-scale plus archived evolutionary searches, not a benchmark
  superiority campaign.
- The current results do not establish general intelligence, consciousness, or
  a finished architecture.
- Replication is within the current Track B protocol; broader families and
  independent implementations remain future falsification work.

## Reproducibility And Artifacts

Start with [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for compact CPU commands.
Then use:

- [data/INDEX.md](../data/INDEX.md) for the artifact map;
- [CLAIMS.md](CLAIMS.md) for evidence level and falsification conditions;
- [LABBOOK.md](LABBOOK.md) for chronology;
- [RESEARCH_LINEAGE.md](RESEARCH_LINEAGE.md) for the complete historical trail.

## Publication Split

GitHub should remain the canonical living lab: broad, artifact-linked,
historically honest, and explicit about failures. The first arXiv paper should
be the frozen citable statement: methods-first, conservative, and centered on
Gate-State Causal Propagation.
