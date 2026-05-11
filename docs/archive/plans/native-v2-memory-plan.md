# Native V2 Memory Plan

Last updated: 2026-04-23

## Purpose

This is the immediate experiment map for `demian_native_v2`.

The target is no longer retention in the weak sense of "some state remains nonzero".
The target is memory behavior:

- what persists
- what is selective
- what is reused later
- what is transmissible
- what remains endogenous versus being dominated by intervention

Language, `lm_head`, and multi-agent symbolic exchange are out of scope for this phase.

## Working Definition

For the current phase, treat memory as:

- retained structure that later changes route choice, takeover timing, basin entry, recovery, or coupling

Do not treat memory as:

- high norm by itself
- slow decay by itself
- a channel name by itself

`long_carrier` is a candidate owner of memory, not proof of memory.

## Architecture Map

`demian_native_v2` lives in [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py:1664).

Relevant state routes:

- `fast`: exposed local surface
- `slow`: deep continuity basin
- `long_carrier`: long-lived persistence owner
- `short_support`: short-lived route stabilizer
- `packet`: transmissible recruitment surface
- `control_short`: local opening pressure
- `control_long`: slower route-history pressure

Key `v2` route biases relative to `v1`:

- stronger `carrier_to_slow`
- stronger `packet_to_slow`
- stronger `control_to_slow`
- slower `long_carrier_decay`
- lower generic release pressure
- weaker support traffic

That means the immediate memory question is:

- does retained structure actually move recruitment into `slow`, or do the memory-like channels only persist passively?

## Existing Harness Pieces

Use these existing functions first:

- `trajectory_map(...)`
  - per-step trajectory with `route_metrics`, `entry_markers`, and summary
- `memory_pair(...)`
  - existing memory sensitivity baseline
- `perturbation_pair(...)`
  - recovery under perturbation
- `coupled_pair(...)`
  - transmissibility / cross-instance influence
- `compare_native_v0_vs_v1(...)`
  - same-seed ancestry comparison
- `compare_native_v1_vs_v2(...)`
  - same-seed ancestry comparison
- `rank_native_v0_onset_predictors(...)`
  - onset-window route predictors
- `rank_native_v1_onset_predictors(...)`
  - onset-window route predictors
- `rank_native_v2_onset_predictors(...)`
  - onset-window route predictors

Relevant route metrics already exposed in `trajectory_map` for `demian_native_v2`:

- `control_to_packet_norm`
- `control_to_carrier_norm`
- `control_to_slow_norm`
- `carrier_to_slow_norm`
- `packet_to_carrier_norm`
- `packet_to_slow_norm`
- `carrier_long_residual_norm`
- `carrier_short_residual_norm`
- `release_strength_mean`
- `control_short_write_mean`
- `control_long_write_mean`
- `packet_write_mean`
- `message_write_mean`

## Main Questions

### 1. Persistence

Question:

- which route carries delayed influence for the longest useful horizon?

Run:

- `trajectory_map("demian_native_v2", ...)`
- `memory_pair("demian_native_v2", ...)`
- `rank_native_v2_onset_predictors(...)`

Look for:

- whether `carrier_long_residual_norm` predicts earlier or stronger slow takeover
- whether `carrier_to_slow_norm` stays predictive later than packet and control metrics
- whether high retained route mass changes later behavior instead of merely remaining large

Failure mode:

- long-carrier stays large but has weak predictive relation to later route changes

### 2. Selectivity

Question:

- does the system keep distinct prior influences separate, or merge them into generic route heat?

Use current harness as baseline:

- compare multiple `memory_pair(...)` and `perturbation_pair(...)` runs under different seeds and injection scales

Immediate comparison:

- small `initial_delta` versus larger `initial_delta`
- early perturbation versus later perturbation

Look for:

- whether delayed influence depends on timing
- whether later dynamics reflect specific earlier differences or simply saturate into the same class

Failure mode:

- all disturbances collapse into the same interior behavior with no route-specific memory signature

### 3. Functional Reuse

Question:

- when retained structure matters later, which future event does it actually change?

Primary outcomes:

- `onset_step`
- `slow_norm_takeoff_step`
- dominant route near onset
- route mass around takeover window

Run:

- `compare_native_v1_vs_v2(...)`
- `compare_native_v0_vs_v1(...)`
- `rank_native_v2_onset_predictors(...)`

Look for:

- whether `v2` shifts onset earlier than `v1`
- whether route mass moves from packet-heavy to slow/carrier-heavy regimes
- whether retained carrier and control terms predict takeover timing more strongly than generic update means

Failure mode:

- retention metrics are present, but future behavior is mostly explained by generic update magnitude

### 4. Route Ownership

Question:

- which route owns usable memory under matched conditions?

Immediate candidates:

- `long_carrier`
- `slow`
- `packet`
- `control_long`

Proxy rule for now:

- a route owns memory if its local metric predicts later transition structure better than competing routes

Run:

- `rank_native_v2_onset_predictors(...)`
- compare top correlations against `rank_native_v1_onset_predictors(...)`

Look for:

- whether `carrier_to_slow_norm`, `packet_to_slow_norm`, or `carrier_long_residual_norm` outrank generic gates
- whether `control_long_write_mean` beats `control_short_write_mean` for delayed effects

Failure mode:

- the apparent memory owner changes arbitrarily across seeds with no stable ranking pattern

### 5. Endogenous Versus Induced Balance

Question:

- are interventions scaffolding endogenous openings, or replacing them?

Use current harness:

- baseline `trajectory_map(...)`
- `perturbation_pair(...)`
- `memory_pair(...)`
- `coupled_pair(...)`

Compare:

- no intervention
- single early intervention
- stronger intervention
- coupled influence

Look for:

- whether intervention shifts timing while preserving native route hierarchy
- whether induced structure decays into endogenous route ownership
- whether intervention simply dominates the whole rollout

Failure mode:

- route activity after intervention mostly reflects the perturbation itself rather than substrate-native routing

## Immediate Run Set

These are the next runs worth doing before inventing new machinery.

### A. Same-family memory behavior check

Goal:

- verify that `v2` changed memory-relevant behavior, not only architecture description

Commands:

```bash
./venv/bin/python - <<'PY'
from development.substrate_lab import compare_native_v1_vs_v2
import json
payload = compare_native_v1_vs_v2(hidden_size=64, steps=512, seeds=list(range(7, 19)))
print(json.dumps(payload["aggregate"], indent=2))
PY
```

Readout:

- `v2_earlier_onset_count`
- `v2_earlier_takeoff_count`
- `route_change_count`

### B. Native v2 onset-memory ranking

Goal:

- identify which local route metrics are actually predictive of later takeover

Commands:

```bash
./venv/bin/python - <<'PY'
from development.substrate_lab import rank_native_v2_onset_predictors
import json
payload = rank_native_v2_onset_predictors(hidden_size=64, steps=512, seeds=list(range(7, 31)))
print(json.dumps(payload["top_rankings"], indent=2))
PY
```

Readout:

- top correlations for `carrier_to_slow_norm`
- `packet_to_slow_norm`
- `carrier_long_residual_norm`
- `control_long_write_mean`
- compare against generic metrics like `fast_update_mean`

### C. Endogenous-versus-induced check

Goal:

- test whether memory-like behavior survives intervention without being replaced by it

Commands:

```bash
./venv/bin/python development/run_substrate_stress_tests.py --substrate demian_native_v2
```

Readout:

- compare perturbed versus unperturbed trajectory summaries
- inspect whether route ordering remains native after intervention

### D. Native ancestry comparison

Goal:

- place `v2` in the native line instead of comparing against old GRU scaffolds

Commands:

```bash
./venv/bin/python - <<'PY'
from development.substrate_lab import compare_native_v0_vs_v1, compare_native_v1_vs_v2
import json
print("v0->v1")
print(json.dumps(compare_native_v0_vs_v1(hidden_size=64, steps=512, seeds=list(range(7, 19)))["aggregate"], indent=2))
print("v1->v2")
print(json.dumps(compare_native_v1_vs_v2(hidden_size=64, steps=512, seeds=list(range(7, 19)))["aggregate"], indent=2))
PY
```

Readout:

- whether movement from `v0 -> v1 -> v2` is actually increasing early slow ownership and useful persistence

## Decision Rules

Treat the current phase as successful only if at least one of these becomes true:

- retained carrier or control structure predicts later takeover better than generic update metrics
- `v2` consistently shifts recruitment into `slow` relative to `v1`
- induced perturbations alter timing without replacing native route ownership
- coupled or memory-pair structure produces delayed behavioral effects that are not reducible to immediate norm differences

Treat the phase as unresolved if:

- the top predictors are only generic gate/update magnitudes
- route ownership is unstable across seeds
- retained structure does not change later route behavior
- intervention dominates endogenous dynamics

## What Not To Do Yet

- do not bring back `lm_head`
- do not optimize for text or dialogue quality
- do not interpret human-readable symbolic behavior as progress yet
- do not move to multi-agent native language experiments before route ownership and memory behavior are clear

## Next If This Works

If memory behavior becomes real in the stricter sense above, the next layer is:

- explicit delayed-reuse tests
- transmissibility tests across paired native instances
- only later, symbolic emergence or language-like bridging
