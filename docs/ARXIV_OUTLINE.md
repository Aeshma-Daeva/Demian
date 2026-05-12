# ArXiv Outline

Working title: **Demian: Discovering Native Mechanisms in Structured Recurrent
Substrates**

Public repository for artifacts and reproducibility:
<https://github.com/Aeshma-Daeva/Demian>

## Thesis

We present Demian, a substrate-building research program that matured into a
methodology for discovering architecture-native computational mechanisms.
Applied to a structured five-compartment recurrent substrate, the methodology
identifies a replicated internal mechanism, Gate-State Causal Propagation,
motivating the next Demian v1 design.

## Abstract Draft

Building custom recurrent substrates requires more than optimizing exposed
behavior: surface stability can hide internal dynamics, and target metrics can
select non-causal shortcuts. We introduce Demian, a methods-first research
program for discovering native mechanisms in structured recurrent substrates.
The method combines factorized state design, evolutionary search, causal
ablations, capsule-continuity probes, and attractor/regime classification.
Applied to a five-channel substrate with `fast`, `slow`, `control`, `message`,
and `carrier` state, the method identifies Gate-State Causal Propagation, a
replicated Track B mechanism in which gate-state history changes downstream
dynamics under route-disabled and gain-zero diagnostics. Across three
native-emergence replications, message and carrier are repeatedly necessary,
full internal-state resume is exact, and surface-only resume fails. These
findings motivate a Demian v1 prototype with explicit gate state. We do not
claim benchmark superiority, general intelligence, consciousness, or completion
of the final architecture.

## Contributions

- A methodology for mechanism discovery in structured recurrent substrates.
- A dual-track search framing separating engineered target pursuit from native
  mechanism emergence.
- Causal ablation and capsule-continuity probes for separating surface behavior
  from internal computation.
- A replicated finding: Gate-State Causal Propagation in 3/3 Track B runs.
- A conservative design synthesis: Demian v1 makes gate state explicit, while
  treating v1 as a hypothesis rather than a validated result.

## Section Plan

1. **Introduction**
   - Original goal: build a native substrate.
   - Direct optimization and surface behavior were misleading.
   - Demian's response: internal dynamics, falsification, and artifact-backed
     claim promotion.
   - Contribution: methodology plus one replicated gate-state mechanism.

2. **Background and Framing**
   - Structured recurrent substrates.
   - Factorized state, routing, recurrent dynamics, attractor regimes.
   - Translate Demian vocabulary into standard ML and dynamical-systems terms.

3. **Substrate Under Study**
   - v9 five-channel scaffold.
   - `fast`, `slow`, `control`, `message`, `carrier`.
   - Release gate and genotype/search space.
   - Clarify this is the study substrate, not the final v1 architecture.

4. **Methodology**
   - Dual-track evolution: engineered targets vs native emergence.
   - Multi-objective evolutionary search.
   - Causal ablations: routes-disabled, gain-zero, per-channel disabled.
   - Capsule continuity: full internal-state resume vs surface-only resume.
   - Attractor/regime classification.
   - Evidence gates and claim promotion.

5. **Results**
   - Gate-State Causal Propagation.
   - Track B discovery context.
   - 3/3 replication summary.
   - Message/carrier necessity.
   - Gain-zero clean and route-disabled positive divergence.
   - Full capsule exact, surface-only gapped.

6. **Negative Results**
   - Sparse release did not generalize.
   - v10 predecessor was not backend/held-out stable.
   - Dynamic selection preserved causality but failed sparse/timed criteria.
   - Bounded-strange appeared but should not be overclaimed as stable
     adaptation.

7. **Synthesis: Demian v1**
   - Gate state becomes explicit.
   - Message/carrier preserved.
   - Surface treated as output, internal state as computation.
   - Release vector retired as primary causal mechanism.
   - v1 is a design consequence, not yet a validated result.

8. **Limitations**
   - Small architecture family.
   - Custom vocabulary.
   - CPU-scale and evolutionary-search limits.
   - No benchmark-superiority claim.
   - No claim of general intelligence, consciousness, or finished architecture.

9. **Reproducibility and Artifacts**
   - Link GitHub repository.
   - Link artifact index.
   - Include compact commands from [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Figure And Table List

- Figure 1: Demian methodology loop: substrate, perturbation, ablation,
  capsule resume, evidence gate.
- Figure 2: v9 five-channel scaffold with `fast`, `slow`, `control`,
  `message`, and `carrier`.
- Figure 3: Gate-State Causal Propagation diagnostic schematic.
- Figure 4: Capsule continuity comparison: full internal-state resume vs
  surface-only replay.
- Table 1: Vocabulary translation from Demian terms to standard terminology.
- Table 2: Track B 3/3 replication summary.
- Table 3: Negative results and resulting design decisions.
