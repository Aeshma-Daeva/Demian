# Native Controller Revision Blueprint

Date: 2026-04-24

Status:
- working plan
- not yet promoted to live truth
- partially instantiated in `demian_native_v6`, now including multi-horizon critic and retained conflict-potential state

Scope:
- native substrate line only
- focus on `demian_native_v3` through `demian_native_v5.3`
- convert the current controller stack into a cleaner next revision target

## Current Read

What is already structurally good:
- the native route anatomy established by `demian_native_v0` and refined through `v3`
- explicit separation of:
  - `fast`
  - `slow`
  - `long_carrier`
  - `short_support`
  - `packet`
  - `control_short`
  - `control_long`
- `tightness` in `v3` as a compact governance variable over consolidation versus looseness

What is not working cleanly enough:
- `v5+` controller learning acts on too few actuators
- `v5.1+` credit is still mostly a hand-authored observer with a tiny learned head
- `v5.2b+` anti-locking is still dominated by threshold logic and schedule logic
- `v5.3` phase control changes behavior over long horizons, but not yet with a strong enough gain in memory quality to justify the added controller complexity

Observed risk:
- controller drift without strong productive memory gains
- more meta-control logic than actual actuator authority
- phase and challenge machinery behaving like a scripted supervisor rather than a learned internal governor

## Keep

Keep these from the current line:

1. Native route anatomy from `v3`
   - this is the stable scaffold
   - do not collapse back into inherited-cell abstractions

2. `tightness` as a first-class governance variable
   - it is the cleanest controller added so far
   - it already maps to meaningful route effects

3. Route-local observability
   - keep exposing:
   - route activity
   - release strength
   - carrier residual
   - packet recruitment
   - slow takeover
   - memory divergence

4. Route-local adaptation idea from `v5`
   - learning on slow-governing paths is directionally right
   - only the current actuation surface is too narrow

## Cut

Cut or demote these in the next revision:

1. Periodic challenge scheduling as a primary mechanism
   - fixed challenge intervals are external script logic
   - anti-locking should be driven by state, not clock time

2. Hard phase switching as an actuator
   - phase can remain as a diagnostic readout
   - it should not remain a large threshold-controlled policy router unless it proves necessary after simpler revisions

3. Controller logic that only changes learning-rate multipliers
   - too much of `v5.2b` and `v5.3` is just scaling `lr` and decay
   - that is meta-policy without enough direct control authority

4. Tiny critic with overly aggregated features as the main observer
   - the current 11-feature credit head is too compressed
   - batch-mean pressure summaries are not enough for robust policy shaping

## Main Diagnosis

The current controller stack is missing a clean separation between:
- observer state
- credit/value estimate
- actuator outputs

Right now these are entangled:
- observer summaries directly feed a tiny credit head
- that head only modulates three learned route deltas
- extra heuristics then clamp, phase, or challenge the result

This creates a bad regime:
- too little learned control surface
- too much hand-authored supervisory logic

## Next Architecture

The next revision should branch from `demian_native_v3`, not from `v5.3` as-is.

Working name:
- `demian_native_v6`

Base:
- start from `v3`
- preserve route anatomy and tightness governor
- rebuild controller stack cleanly on top

Structure:

1. Substrate core
   - native route dynamics
   - same channel ownership as `v3`
   - no challenge scheduler
   - no hard phase machine in the forward path

2. Observer state
   - separate recurrent controller state
   - tracks recent route activity and internal mismatch without immediately becoming actuator logic

3. Credit / value head
   - predicts whether recent governance improved continuity and memory behavior
   - delayed update remains acceptable
   - but observer inputs should be richer and explicitly structured

4. Actuator head
   - outputs bounded governance deltas
   - should drive more than the current three route-to-slow scalars

5. Conflict potential
   - retain unresolved route opposition as structured internal state
   - do not treat disagreement as default error
   - distinguish bounded plurality from destructive monopoly

Current extension:
- `v6` now carries explicit `conflict_potential`
- critic disagreement is no longer treated only as a suppression trigger
- `v6` now includes bounded conflict-to-route drives for packet, carrier, and slow channels
- next comparison target is whether retained conflict can become differentiated route structure without basin breakage

## Next Controller API

The next controller should not only control:
- `control_to_slow`
- `packet_to_slow`
- `carrier_to_slow`

It should be able to modulate a bounded subset of:

- `control_to_slow`
- `packet_to_slow`
- `carrier_to_slow`
- `control_to_packet`
- `support_to_packet`
- `release_gain`
- `release_threshold`
- `long_carrier_decay`
- `control_short_decay` or write pressure
- `control_long_decay` or write pressure

Constraints:
- all controller outputs must remain bounded
- each actuator delta should have a clear metric in `route_metrics`
- do not let the controller rewrite the whole substrate every step

## Observer Revision

The next observer should explicitly track:

- route activity by path, not just route mean
- route imbalance, not just route norm
- carrier pressure
- packet pressure
- control pressure
- `fast/slow` mismatch
- release intensity
- memory separation signals from matched pair probes when available
- recent trend, not just instantaneous value

Implementation direction:
- give the controller its own small recurrent state
- use that state to integrate observations over time
- avoid direct dependence on a single-step threshold stack

## Credit Revision

The next credit head should estimate:
- whether recent controller action improved stability
- whether recent controller action improved useful memory separation
- whether recent controller action reduced unproductive lock

Credit target should be built from:
- coherence change
- mismatch change
- release change
- route balance change
- memory divergence quality

Important distinction:
- more divergence is not automatically better
- the target should penalize destructive divergence that destroys continuity without preserving structured separation

## Anti-Locking Revision

Anti-locking should become:
- state-driven
- local
- reversible

Anti-locking should stop being:
- periodic
- phase-scripted
- mostly an `lr/decay` rescaler

Preferred mechanism:
- derive a continuous lock pressure estimate
- let that estimate directly open specific actuators
- keep challenge-like behavior as a soft outcome, not as a named scheduled phase

## Phase Revision

Phase can remain as:
- a diagnostic summary
- a downstream classifier over trajectories

Phase should not remain as:
- the main decision mechanism in the recurrent controller

Interpretation:
- if phase is real, it should be recoverable from trajectory and controller state
- it should not need to be hand-installed as a governing switchboard

## Concrete Build Plan

Step 1:
- branch a new controller from `demian_native_v3`
- keep `tightness`
- remove challenge and phase routing logic

Step 2:
- add a small controller state
- example role: short history of route pressure, mismatch, and release context

Step 3:
- replace the current 3-scalar route delta actuator with a bounded actuator vector over the selected governance parameters

Step 4:
- keep credit delayed
- but train/update it against explicit multi-term viability targets instead of mostly hand-shaped heuristic carryover

Step 5:
- expose all new actuator outputs in `route_metrics`
- make the new controller inspectable from the start

Step 6:
- add direct comparisons:
  - `v3` vs next revision
  - `v5.2c` vs next revision
  - `v5.3` vs next revision

## Required New Analysis Hooks

Add these before promoting any new claim:

1. same-seed controller trace comparison
   - actuator outputs over time
   - lock pressure over time
   - route ownership over time

2. memory quality comparison
   - not just final cosine
   - also continuity retention versus destructive spread

3. drift quality report
   - classify long-horizon change into:
   - productive adaptation
   - neutral drift
   - destructive divergence

4. controller-state readout
   - if the observer/controller has recurrent state, log it in reduced form

## Initial Rejection Rules

Reject the next controller revision if:
- it only wins on best-case seeds
- it increases drift without improving memory quality
- it needs scheduled challenge windows to appear alive
- it cannot explain its own actuator changes through exposed metrics

Promote only if:
- same-seed comparisons show better typical-case memory behavior
- the controller improves lock handling without collapsing continuity
- gains survive matched seed sweeps

## Working Conclusion

The next revision should not be:
- `v5.3` plus more controller logic

The next revision should be:
- `v3` substrate
- clearer observer/credit/actuator split
- richer but bounded actuator surface
- softer, state-driven anti-locking
- phase treated as readout before phase is treated as policy
