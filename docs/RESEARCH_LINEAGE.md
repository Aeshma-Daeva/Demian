# Research Lineage

## Purpose

This document preserves the full trail of Demian's development. It is the
legacy-mode entry point: not the shortest public explanation, but the complete
path from the first tentative KV-cache probes to the current Demian v1
gate-state synthesis.

The front page should stay selective. This file keeps the line of thought
visible: what was tried, what failed, what survived, and why the project moved
to the next substrate.

Use with:

- [README.md](/home/xenith/demian/README.md)
- [docs/RESEARCH_MAP.md](/home/xenith/demian/docs/RESEARCH_MAP.md)
- [docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md)
- [docs/NATIVE_MECHANISMS.md](/home/xenith/demian/docs/NATIVE_MECHANISMS.md)
- [data/INDEX.md](/home/xenith/demian/data/INDEX.md)
- [docs/archive/README.md](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/docs/archive/README.md)

## Reading Rule

Legacy does not mean obsolete. It means historical context with status.

Each stage below is marked by its durable contribution:

- `observation`: artifact-backed result
- `inference`: interpretation from one or more observations
- `abandoned target`: useful negative path
- `synthesis`: design consequence carried forward

## 1. KV-Cache Transformer Loop

Question:

- Can a frozen transformer be made to interact with its own internal state, and
  does that interaction have measurable structure beyond text output?

What was tried:

- Capture residuals at the final layer/token.
- Inject or blend them into the KV cache.
- Compare proprioceptive injection, random injection, additive KV perturbation,
  appending positions, projection modulation, activation steering, and weight
  perturbation.
- Track trajectory energy, variance, mode transitions, and attention to injected
  positions instead of judging text quality.

What mattered:

- `observation`: single injection was materially different from continuous
  injection.
- `observation`: additive KV perturbation degraded more slowly than appending
  new positions.
- `inference`: the important object was not the prose produced by the model but
  the trajectory induced by recurrent self-contact.
- `synthesis`: internal trajectory geometry became the project's measurement
  target.

Primary trail:

- [docs/archive/notes/PROJECT_DEMIAN.md](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/docs/archive/notes/PROJECT_DEMIAN.md)
- [docs/archive/notes/ai_thoughts.md](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/docs/archive/notes/ai_thoughts.md)
- [data/trajectory_kv_comparison.json](/home/xenith/demian/data/trajectory_kv_comparison.json)

Status:

- Historical origin. Do not use this as the current architecture target, but do
  preserve it as the first point where trajectory-over-text became explicit.

## 2. Transformer Reservoir Baseline

Question:

- If self-reference is measured as trajectory rather than language, what
  attractor class does a transformer-style loop naturally produce?

What was tried:

- Batch reservoir runs over transformer self-reference.
- Measure autocorrelation, velocity alignment, period-2 switches, and energy.

What mattered:

- `observation`: the transformer reservoir produced a stable repeated period-2
  signature in saved summaries.
- `inference`: transformer self-reference behaved more like deterministic
  alternation than open-ended exploration under that setup.
- `synthesis`: architecture-specific attractor fingerprints became a baseline
  concept.

Artifacts:

- [legacy/root_cli/run_reservoir_batch.py](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/legacy/root_cli/run_reservoir_batch.py)
- [legacy/demian_runtime/reservoir.py](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/legacy/demian_runtime/reservoir.py)
- [data/reservoir_batch/batch_summary.json](/home/xenith/demian/data/reservoir_batch/batch_summary.json)
- [data/reservoir/](/home/xenith/demian/data/reservoir)

Status:

- Baseline evidence. Useful as a contrast class, not the destination.

## 3. Mamba Recurrence Contrast

Question:

- Does a state-space model produce the same self-reference attractor as the
  transformer loop?

What was tried:

- Matched Mamba reservoir runs.
- Compare cache-on and cache-off dynamics.
- Track energy, lag autocorrelation, and period-2 switches.

What mattered:

- `observation`: Mamba did not reproduce the transformer period-2 fingerprint.
- `observation`: persistent SSM memory pushed toward a tighter, lower-energy
  fixed basin with smaller motion.
- `inference`: different inherited architectures have different recurrence
  primitives.
- `synthesis`: Demian should extract mechanisms from architectures rather than
  assume one universal self-loop behavior.

Artifacts:

- [legacy/root_cli/run_mamba_batch.py](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/legacy/root_cli/run_mamba_batch.py)
- [legacy/demian_runtime/mamba_reservoir.py](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/legacy/demian_runtime/mamba_reservoir.py)
- [data/mamba_batch/mamba_batch_summary.json](/home/xenith/demian/data/mamba_batch/mamba_batch_summary.json)
- [data/mamba_reservoir/](/home/xenith/demian/data/mamba_reservoir)

Status:

- Baseline contrast. The transformer primitive and Mamba primitive differ, and
  that difference motivates substrate-specific analysis.

## 4. Competition And Continuity Pressure

Question:

- What happens when recurrent state is exposed to population pressure, memory
  bottlenecks, and inheritance-like continuation?

What was tried:

- Multi-agent competition runs.
- Hebbian/fast-weight variants.
- Log survival, scoring, extinction, and continuity through population events.

What mattered:

- `inference`: competition selected for continuity depth more than human-facing
  usefulness.
- `inference`: death behaved like memory erasure; survival preserved recurrent
  accumulation.
- `synthesis`: continuity and transmissible state became first-class concerns.

Artifacts:

- [legacy/root_cli/run_competition.py](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/legacy/root_cli/run_competition.py)
- [legacy/demian_runtime/competition.py](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/legacy/demian_runtime/competition.py)
- [data/competition50/](/home/xenith/demian/data/competition50)
- [data/competition50_run2/](/home/xenith/demian/data/competition50_run2)
- [data/competition_v3/](/home/xenith/demian/data/competition_v3)
- [data/competition_hebbian/](/home/xenith/demian/data/competition_hebbian)

Status:

- Population-pressure ancestry. It explains why later substrate work cared
  about persistence and resume fidelity.

## 5. GRU And Dual-GRU Message-State Ancestry

Question:

- Can inherited recurrent cells expose useful internal classes, message states,
  or transition structure inside apparently simple basins?

What was tried:

- GRU and dual-GRU families.
- Message-state ablations.
- Interior-class summaries and transition maps.

What mattered:

- `observation`: fixed-point behavior was not automatically trivial.
- `observation`: accumulating fixed-point classes carried distinct internal
  message behavior.
- `synthesis`: the project needed explicit state owners and route analysis
  rather than opaque inherited cell mechanics.

Artifacts:

- [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py)
- [tests/test_substrate_lab.py](/home/xenith/demian/tests/test_substrate_lab.py)
- [data/substrate_lab/comparative_class_summary.json](/home/xenith/demian/data/substrate_lab/comparative_class_summary.json)
- [data/substrate_lab/dual_gru_family_report.json](/home/xenith/demian/data/substrate_lab/dual_gru_family_report.json)
- [development/report_dual_gru_family.py](/home/xenith/demian/development/report_dual_gru_family.py)

Status:

- Method ancestry. Keep it as the bridge from inherited cells to explicit
  native substrates.

## 6. Native Route Substrates v0-v7.4

Question:

- What if persistence, support, recruitment, control, packets, carriers,
  boundaries, and ownership are explicit state owners instead of implicit cell
  behavior?

What was tried:

- Native substrate versions v0 through v7.4.
- Explicit route ownership.
- Slow recruitment, carrier persistence, endogenous control, boundary organs,
  trajectory memory, metabolic constraints, self-policy, quarantine, and
  topology shadow.

What mattered:

- `observation`: the native line separated persistence, support, recruitment,
  and control into explicit state owners.
- `inference`: the organ-heavy line was powerful as a design laboratory but
  increasingly difficult to isolate mechanistically.
- `synthesis`: v8/v9 simplification became necessary to test which mechanisms
  actually mattered.

Artifacts:

- [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py)
- [tests/test_substrate_lab.py](/home/xenith/demian/tests/test_substrate_lab.py)
- [docs/archive/plans/native-v2-memory-plan.md](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/docs/archive/plans/native-v2-memory-plan.md)
- [docs/archive/plans/2026-04-24-native-controller-revision-blueprint.md](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/docs/archive/plans/2026-04-24-native-controller-revision-blueprint.md)
- [docs/archive/plans/2026-04-29-v74-falsification-plan.md](https://github.com/Aeshma-Daeva/Demian-Archive/blob/main/docs/archive/plans/2026-04-29-v74-falsification-plan.md)

Status:

- Historical baseline and design ancestry. v7.4 remains the promoted
  organ-heavy reference line, but not the current active substrate.

## 7. v8/v9 Simplification

Question:

- Can the organ-heavy native line be reduced to a smaller state scaffold without
  losing the ability to study recurrence geometry?

What was tried:

- v8 as immediate seven-channel comparison.
- canonical v9 as minimal `fast`, `slow`, `control` substrate.
- v9-v8 comparison and fixed-point/interior-class reading.

What mattered:

- `observation`: canonical v9 exists and is tested, but saved v9-v8 comparison
  does not support v9 superiority.
- `observation`: fixed-point surface behavior remained common.
- `synthesis`: surface-fixed did not mean internally inactive, so message and
  carrier were reintroduced as explicit accumulation channels.

Artifacts:

- [docs/SUBSTRATE_ANATOMY.md](/home/xenith/demian/docs/SUBSTRATE_ANATOMY.md)
- [data/substrate_lab/v9_v8_compare_20260508/summary.json](/home/xenith/demian/data/substrate_lab/v9_v8_compare_20260508/summary.json)
- [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py)

Status:

- Active baseline context. It is the narrowed scaffold from which v9
  five-channel and Demian v1 emerge.

## 8. v9 Five-Channel Message/Carrier Scaffold

Question:

- Can explicit `message` and `carrier` channels preserve internal structure,
  accumulate perturbation information, and produce sparse consequential release
  events?

What was tried:

- Extend v9 with `message` and `carrier`.
- Add release gates and route-specific release vectors.
- Run v9 five-channel evolution and trajectory exports.
- Render 2D and machine-native diagnostic surfaces.

What mattered:

- `observation`: the archive preserved multiple bounded regimes, including
  `surface_fixed_accumulating` and `bounded_strange`.
- `observation`: release could become rarer without simply becoming always-on
  coupling.
- `abandoned target`: route-specific sparse causal release remained weak and
  did not become the durable mechanism.
- `synthesis`: message/carrier were worth keeping; release-vector injection was
  not the right primitive.

Artifacts:

- [development/probe_v9_message_carrier_strange.py](/home/xenith/demian/development/probe_v9_message_carrier_strange.py)
- [development/evolve_v9_5ch_release.py](/home/xenith/demian/development/evolve_v9_5ch_release.py)
- [data/evolution/v9_5ch_release_20260509_full/archive.json](/home/xenith/demian/data/evolution/v9_5ch_release_20260509_full/archive.json)
- [docs/assets/v9_5ch_neuron_activity_overview.svg](/home/xenith/demian/docs/assets/v9_5ch_neuron_activity_overview.svg)
- [docs/assets/v9_5ch_neuron_activations.svg](/home/xenith/demian/docs/assets/v9_5ch_neuron_activations.svg)
- [docs/assets/v9_5ch_gating_activations.svg](/home/xenith/demian/docs/assets/v9_5ch_gating_activations.svg)

Status:

- Active scaffold and predecessor evidence, not a finished architecture claim.

## 9. v10.0 Frozen Evolution As Predecessor Evidence

Question:

- If selection pressure runs over time instead of manual score tweaking, does
  sparse release stabilize?

What was tried:

- Four-island v10.0 frozen evolution on the v9 five-channel scaffold.
- CPU reproduction and held-out cross-evaluation.
- Release-gain and release-route ablations.

What mattered:

- `observation`: archived eval-seed winner had sparse-to-borderline release and
  strong event/phase scores.
- `observation`: held-out CPU checks did not preserve that result cleanly.
- `abandoned target`: do not publish v10.0 as stable sparse release.
- `synthesis`: keep it as predecessor evidence and falsification pressure for
  Demian v1.

Artifacts:

- [data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json)
- [data/substrate_lab/v10_frozen_repro_seed94_20260511/summary.csv](/home/xenith/demian/data/substrate_lab/v10_frozen_repro_seed94_20260511/summary.csv)
- [data/substrate_lab/v10_frozen_cross_eval_20260511/summary.csv](/home/xenith/demian/data/substrate_lab/v10_frozen_cross_eval_20260511/summary.csv)
- [data/substrate_lab/v10_frozen_cross_eval_routes_20260511/summary.csv](/home/xenith/demian/data/substrate_lab/v10_frozen_cross_eval_routes_20260511/summary.csv)

Status:

- Predecessor evidence. Important because it showed selection could find
  interesting candidates, and because held-out failure clarified what not to
  claim.

## 10. Track A / Track B Split

Question:

- Is the target mechanism engineered sparse release, or is there a native
  mechanism the substrate prefers?

What was tried:

- Track A: engineered sparse causal release objectives.
- Track B: native emergence and structural discovery.
- Ablation, capsule, channel-disabled, held-out, and replication checks.

What mattered:

- `abandoned target`: Track A sparse release repeatedly failed to generalize.
- `observation`: Track B discovered Gate-State Causal Propagation.
- `synthesis`: discovery pressure was more revealing than forcing the original
  designed mechanism.

Artifacts:

- [development/run_track_b_discovery.py](/home/xenith/demian/development/run_track_b_discovery.py)
- [development/gate_state_propagation_characterization.py](/home/xenith/demian/development/gate_state_propagation_characterization.py)
- [data/diagnostics/gate_state_propagation_characterization_20260511/summary.json](/home/xenith/demian/data/diagnostics/gate_state_propagation_characterization_20260511/summary.json)
- [data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json](/home/xenith/demian/data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json)

Status:

- Current strongest empirical line.

## 11. Gate-State Causal Propagation

Question:

- What mechanism survived when explicit release-vector gain was removed?

What was found:

- Gate-state propagation emerged natively from Track B.
- Original trajectories diverged from gain-zero and route-disabled variants even
  when gain-zero diagnostics were clean.
- Full internal-state resume was exact.
- Surface-only resume failed.
- Message and carrier were repeatedly necessary.

What mattered:

- `observation`: 3/3 Track B replications reproduced the mechanism.
- `observation`: held-out route/gain-zero divergence remained positive.
- `observation`: message/carrier were top-two necessary channels in 3/3
  replications.
- `synthesis`: gate state should become a designed primitive in Demian v1.

Artifacts:

- [docs/NATIVE_MECHANISMS.md](/home/xenith/demian/docs/NATIVE_MECHANISMS.md)
- [data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json](/home/xenith/demian/data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json)
- [data/diagnostics/gate_state_track_b_replication_summary_20260511/replication_summary.csv](/home/xenith/demian/data/diagnostics/gate_state_track_b_replication_summary_20260511/replication_summary.csv)

Status:

- Replicated core mechanism. This is the strongest current publishable finding.

## 12. Same-Island And Dynamic Selection Probes

Question:

- Were observed variants dynamic adaptations of one mechanism, lineage
  adaptations, or stable convergent morphology?
- Could temporal objective scaffolding combine causal structure, sparsity, and
  delayed timing?

What was tried:

- Same-island archive analysis around combined-discovery candidates.
- Dynamic selection probe: foundation, sparsity, then timing phases.
- Checkpoint confirmation and held-out evaluation.

What mattered:

- `observation`: causal divergence remained robust.
- `observation`: sparse/timed held-out phenotype did not survive.
- `abandoned target`: temporal scaffolding did not solve sparse causal timing in
  this substrate configuration.
- `synthesis`: objective tuning was not enough; the architecture grain needed
  to change.

Artifacts:

- [development/analyze_same_island_adaptation.py](/home/xenith/demian/development/analyze_same_island_adaptation.py)
- [development/run_dynamic_selection_probe.py](/home/xenith/demian/development/run_dynamic_selection_probe.py)
- [data/diagnostics/dynamic_selection_probe_20260513/checkpoint_report.json](/home/xenith/demian/data/diagnostics/dynamic_selection_probe_20260513/checkpoint_report.json)

Status:

- Negative-result and interpretation layer. It protects the project from
  overclaiming sparse timed causality.

## 13. Demian v1 Explicit Gate-State Synthesis

Question:

- Given the evidence, what should the substrate treat as native rather than
  emergent accident?

What was built:

- A prototype with explicit `gate` state alongside `fast`, `slow`, `control`,
  `message`, and `carrier`.
- Gate state modulates routes directly.
- Sparsity is reframed as gate-state change or pressure variation rather than
  release-open duty.
- Capsule boundary includes internal state, not only surface output.

What mattered:

- `synthesis`: Demian v1 starts from Gate-State Causal Propagation as the
  architectural primitive.
- `abandoned target`: release-vector injection is no longer the primary causal
  mechanism.
- `caveat`: the prototype is contract-tested and smoke-run, not yet an
  empirically promoted mechanism.

Artifacts:

- [development/demian_v1_gate_state.py](/home/xenith/demian/development/demian_v1_gate_state.py)
- [tests/test_demian_v1_gate_state.py](/home/xenith/demian/tests/test_demian_v1_gate_state.py)
- [docs/SUBSTRATE_ANATOMY.md](/home/xenith/demian/docs/SUBSTRATE_ANATOMY.md)

Status:

- Current synthesis point. This is where the trail arrives, not where the trail
  should be erased.

## Public Framing

Front-page story:

- Demian studies internal recurrence, attractor geometry, and native mechanisms
  in structured recurrent substrates.
- The strongest result is replicated Gate-State Causal Propagation.
- Demian v1 is the architectural response.

Legacy-mode story:

- The work began with transformer/KV-cache self-contact.
- Mamba showed architecture-specific recurrence primitives.
- Competition made continuity and memory survival central.
- GRU/dual-GRU work made fixed-point interiors visible.
- Native substrates made route ownership explicit.
- v9 five-channel made message/carrier accumulation measurable.
- Track B revealed the mechanism the substrate wanted to use.
- Demian v1 is the result of following that trail rather than forcing the
  original release-vector idea.
