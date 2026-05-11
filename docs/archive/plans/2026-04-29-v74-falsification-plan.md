# v7.4 Falsification Plan

Date: 2026-04-29

Status:
- working plan
- code-grounded on `demian_native_v7.4`
- not yet promoted to live truth

Scope:
- native substrate line only
- active target: `demian_native_v7.4`
- ancestry/comparison: `demian_native_v7.2`, `demian_native_v7.1`, `demian_native_v6`
- objective is falsification, not feature addition

## Current Read

`demian_native_v7.4` is the current active code line.

The live orientation docs still describe `demian_native_v7.1` as active. Treat those docs as stale until updated. Use `development/substrate_lab.py` and v7.4 artifacts as ground truth.

v7.4 adds a self-policy layer over the v7.2 metabolic-resource scaffold:

- `self_potential`
- `quarantine_state`
- `topology_shadow`
- ownership/viability tension
- self-policy gates:
  - preserve
  - adapt
  - recover
  - refuse
  - quarantine
  - integrate
  - hold
- dynamic step length
- dynamic topology injection

Important existing evidence:

- resume-continuity probes show capsule resume nearly matches uninterrupted continuation
- notebook/surface-only resume diverges from uninterrupted continuation
- dynamic step sweep is mixed:
  - broad profiles often lose to fixed step
  - `nano` profile is the current dynamic candidate
- dynamic topology is not yet proven causal:
  - some sweeps select `0.0`
  - selected nonzero coupling produces measurable injection but only marginal score support

## Main Hypothesis

v7.4 is useful if ownership/viability tension can act as a real internal control signal that improves continuity, recovery, route separation, and topology handling without becoming decorative feedback.

The claim is not:
- v7.4 has more metrics
- v7.4 has richer names
- v7.4 produces prettier traces

The claim is:
- the new organs do causal work
- removing or corrupting them should produce measurable behavioral loss
- their route metrics should separate across stress conditions in interpretable ways

## Weakest Assumption

The weakest assumption is that ownership/viability tension is meaningful enough to drive policy.

In code, this signal influences:

- policy gate distribution
- `self_potential`
- quarantine write
- dynamic step length
- active/projection/boundary updates
- dynamic topology authority

If ownership/viability is poorly calibrated, v7.4 is not an architecture yet. It is a feedback maze with good instrumentation.

## Phase 0: Repair Restart Context

Goal:
- prevent future sessions and agents from restarting from stale v7.1 framing

Edits:
- update `README.md`
- update `docs/WORKING_STATE.md`
- update `docs/RESEARCH_MAP.md`
- update `docs/CLAIMS.md`
- update `docs/AUTO_STATUS.md` only through the existing generator if appropriate

Required content:
- active substrate is `demian_native_v7.4`
- `demian_native_v7.1` and `demian_native_v7.2` are ancestry/comparison lines
- `nano` dynamic step is the current candidate profile, not final truth
- dynamic topology remains unproven
- next phase is organ falsification

Pass condition:
- a fresh reader can identify v7.4 as active without being told verbally

Fail condition:
- docs still imply v7.1 is operational default

## Phase 1: v7.4 Organ Ablation Matrix

Goal:
- determine whether v7.4 organs are causal

Variants:

1. `full`
   - current v7.4 default or selected candidate kwargs

2. `fixed_step`
   - dynamic step disabled:
   - `dynamic_step_min=1.0`
   - `dynamic_step_max=1.0`
   - all dynamic step acceleration/slowdown terms set to `0.0`

3. `nano_step`
   - current candidate:
   - `dynamic_step_min=0.96`
   - `dynamic_step_max=1.01`
   - `dynamic_step_hold_slowdown=0.03`
   - `dynamic_step_refusal_slowdown=0.02`
   - `dynamic_step_recovery_accel=0.015`
   - `dynamic_step_integrate_accel=0.010`

4. `no_self_policy_drive`
   - set `self_policy_scale=0.0`
   - preserve observation metrics if possible

5. `no_quarantine`
   - set `quarantine_scale=0.0`
   - set `quarantine_retention=0.0` or zero quarantine state each step

6. `no_topology_shadow`
   - preserve inherited `topology_state`
   - disable `topology_shadow` update and dynamic topology injection

7. `shadow_without_injection`
   - update `topology_shadow`
   - do not inject it into inherited topology

8. `injection_without_shadow_memory`
   - allow dynamic topology injection
   - zero `topology_shadow` after each step

9. `no_ownership_policy`
   - replace ownership with neutral constant `0.5`

10. `randomized_ownership_policy`
    - replace ownership with seeded random scalar per step
    - preserves distributional motion but destroys semantic coupling

Metrics:
- RSS-negation final cosine
- RSS-negation peak message ratio
- memory final cosine
- bottleneck reuse ratio
- bottleneck unique codes
- bottleneck entropy
- coupled final cosine
- coupled mean cosine
- resume capsule final cosine
- resume trajectory shape ratio
- route metric mean gaps against full v7.4:
  - `v74_viability_mean`
  - `v74_ownership_mean`
  - `v74_tension_mean`
  - `v74_hold_pressure_mean`
  - `v74_resolution_open_mean`
  - `v74_topology_state_norm`
  - `v74_dynamic_topology_injection_norm`

Seeds:
- first pass: `94,95,96,97`
- second pass if signal exists: `94-109`

Pass condition:
- at least one claimed v7.4 organ produces a specific, repeatable behavioral loss when ablated
- loss is visible in both external probes and route metrics

Fail condition:
- ablations behave indistinguishably from full v7.4
- route metrics move but behavior does not
- behavior changes but route metrics do not explain it

## Phase 2: Ownership/Viability Quadrant Probe

Goal:
- test whether the central ownership/viability signal has discriminative control meaning

First result:
- artifact: `data/substrate_lab/v74_ownership_quadrants_20260429/summary.json`
- script: `development/probe_v74_ownership_viability_quadrants.py`
- constructed states successfully separate ownership/viability quadrants
- pressure terms separate strongly:
  - high ownership + high viability drives integrate pressure
  - high ownership + low viability drives recover pressure
  - low ownership + high viability drives refuse pressure
  - low ownership + low viability drives reject pressure
- self-policy gates separate weakly:
  - dominant-gate margin from uniform is about `0.035`
  - high ownership + high viability selected recover instead of integrate
  - high ownership + low viability selected adapt instead of recover
  - low ownership + low viability selected adapt instead of quarantine/hold
- current read: the ownership/viability pressure algebra works better than the self-policy head that consumes it
- implication: do not treat v7.4 self-policy gates as proven organs yet

Pressure-policy follow-up:
- code: `pressure_policy_blend` in `DemianNativeV74Substrate`
- alias: `demian_native_v7.4_pressure_policy`
- quadrant artifact: `data/substrate_lab/v74_ownership_quadrants_pressure_policy_20260429/summary.json`
- ablation artifact: `data/substrate_lab/v74_pressure_policy_ablation_20260429/summary.json`
- pressure-routed gates separate quadrants cleanly:
  - high ownership + high viability selects integrate
  - high ownership + low viability selects recover
  - low ownership + high viability selects refuse
  - low ownership + low viability selects quarantine
  - dominant-gate margin from uniform rises from about `0.035` to about `0.394`
- behavior does not improve in first pass:
  - RSS recovery drops from about `0.622` to about `0.599`
  - memory cosine drops from about `0.753` to about `0.735`
  - resume fidelity drops slightly
  - dynamic topology injection rises
- current read: correct symbolic gate routing is not enough; the route actuators downstream of gates are not calibrated to tolerate strong pressure routing
- implication: pressure policy is a diagnostic lens, not the new default

Calibration follow-up:
- script: `development/sweep_v74_pressure_policy.py`
- artifact: `data/substrate_lab/v74_pressure_policy_sweep_20260429/summary.json`
- tested blends: `0, 0.1, 0.2, 0.35, 0.5`
- tested floors: `0.005, 0.01, 0.03`
- observed tradeoff:
  - blend `0.0`: gate accuracy `0.25`, RSS `0.622`, memory `0.753`
  - blend `0.1`: gate accuracy `0.50`, RSS `0.619`, memory `0.751`
  - blend `0.2`: gate accuracy `0.75`, RSS `0.616`, memory `0.749`
  - blend `0.35`: gate accuracy `1.00`, RSS `0.612`, memory `0.746`
  - blend `0.5`: gate accuracy `1.00`, RSS `0.608`, memory `0.743`
- score-max point was blend `0.5`, floor `0.005`, but this overweights symbolic gate correctness
- current calibrated elbow:
  - `pressure_policy_blend=0.2`
  - `pressure_policy_floor=0.005`
  - alias: `demian_native_v7.4_calibrated_pressure_policy`
- current read: pressure policy gives a tunable semantic/behavior tradeoff, but every tested increase in gate correctness costs recovery and memory
- implication: calibrated pressure routing should be tested as a probe variant, not promoted to default

Construct four synthetic state classes:

1. high ownership, high viability
2. high ownership, low viability
3. low ownership, high viability
4. low ownership, low viability

Expected gate tendencies:

- high ownership + high viability:
  - `integrate_gate` should rise
  - `resolution_open` should remain high
  - quarantine should remain low

- high ownership + low viability:
  - `recover_gate` should rise
  - recovery write should affect active state

- low ownership + high viability:
  - `refuse_gate` should rise
  - boundary/projection separation should increase

- low ownership + low viability:
  - `quarantine_gate` or `hold_gate` should rise
  - topology authority should remain bounded

Pass condition:
- gate distributions separate by quadrant across seeds
- separation survives small perturbations
- response is not only a scalar norm increase

Fail condition:
- policy gates remain near uniform
- gates respond mostly to norm magnitude rather than quadrant
- random ownership produces the same policy patterns

## Phase 3: Dynamic Topology Causality

Goal:
- decide whether `topology_shadow` is an organ or telemetry

Arms:

1. full dynamic topology
2. dynamic topology coupling `0.0`
3. `topology_shadow` updated but never injected
4. topology injection active but `topology_shadow` zeroed every step
5. inherited topology frozen
6. inherited topology active, shadow frozen

Probes:
- perturbation stress
- memory pair
- bottleneck run
- coupled pair
- resume continuity

Pass condition:
- full dynamic topology creates a repeatable advantage in at least one probe
- advantage disappears under a targeted lesion
- route metrics identify when authority opens and when topology injection matters

Fail condition:
- all topology arms behave similarly
- nonzero injection only hurts memory/recovery
- topology signal is measurable but not behaviorally causal

## Phase 4: Dynamic Step Decision

Goal:
- decide whether dynamic step remains in v7.4 default

Compare:
- fixed
- nano
- micro
- broad profiles only as stress references

Decision rule:
- prefer `nano` only if it keeps resume fidelity and improves at least one structural behavior without weakening memory below threshold
- otherwise keep fixed step and demote dynamic step to experimental profile

Current tentative read:
- `nano` is worth keeping as candidate
- broad dynamic step is too aggressive

Pass condition:
- `nano` beats fixed on bottleneck reuse or coupling response while preserving:
  - memory final cosine >= `0.78`
  - RSS-negation final cosine >= `0.84`
  - resume capsule final cosine >= `0.999`

Fail condition:
- fixed step remains better across seeds
- nano advantage only appears on seed `94`

## Phase 5: Claim Promotion

Only promote claims after Phase 1-4.

Possible promoted claims:

1. v7.4 self-policy is causal
2. v7.4 quarantine separates refusal from recovery
3. v7.4 dynamic topology is causal
4. v7.4 dynamic topology is not causal and should be cut or redesigned
5. `nano` dynamic step improves behavior under bounded conditions
6. fixed step remains the correct default

Every promoted claim needs:
- artifact path
- compared variants
- seeds
- falsifier
- residual risk

## Do Not Do Yet

Do not build v7.5 yet.

Do not add another organ to compensate for unclear v7.4 behavior.

Do not optimize for a single summary score.

Do not treat named variables as evidence.

Do not let a route metric become a claim unless a lesion changes behavior.

## Implementation Targets

Likely new scripts:

- `development/ablate_native_v74_organs.py`
- `development/probe_v74_ownership_viability_quadrants.py`
- `development/probe_v74_topology_causality.py`
- `development/report_native_v74_falsification.py`

Likely new artifacts:

- `data/substrate_lab/v74_organ_ablation_20260429/summary.json`
- `data/substrate_lab/v74_ownership_quadrants_20260429/summary.json`
- `data/substrate_lab/v74_topology_causality_20260429/summary.json`
- `data/substrate_lab/v74_falsification_report_20260429/summary.json`

Existing helpers to reuse:

- `perturbation_stress_test`
- `memory_pair`
- `bottleneck_run`
- `coupled_pair`
- `resume_continuity_probe`
- `trajectory_map`
- `_manual_step_trace`

## Working Standard

This phase is not optimization.

This phase is organ accountability.

Every v7.4 component must answer:
- what behavior does it change?
- what stress makes it matter?
- what lesion removes that behavior?
- what metric saw it before the summary score did?

If an organ cannot answer those questions, it is not yet an organ.
