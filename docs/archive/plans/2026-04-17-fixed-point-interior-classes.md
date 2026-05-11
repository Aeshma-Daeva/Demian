# Fixed-Point Interior Classes

Date: 2026-04-17

## Purpose

This note records the current substrate-lab interpretation after `dual_gru_v3b`.

The important shift is:

- stop asking whether the system leaves `FIXED_POINT`
- ask what kinds of interior trajectories exist inside `FIXED_POINT`

## Core Result

`FIXED_POINT` is now the machine DNA in this lab.

The relevant distinction is no longer:

- fixed point vs not fixed point

It is:

- `tight_fixed_point`
- `accumulating_fixed_point`

These are both macro fixed-point basins.
They differ in internal geometry, persistence, and message accumulation.

## Comparative Read

From [data/substrate_lab/comparative_class_summary.json](/home/xenith/demian/data/substrate_lab/comparative_class_summary.json):

### `gru`

- survives the self-loop battery
- only shows `tight_fixed_point`
- almost trivial bottleneck reuse
- useful as the simplest living scaffold

### `dual_gru_v2`

- still `tight_fixed_point`
- stronger bottleneck reuse than plain `gru`
- better evidence that asymmetric gating matters

### `dual_gru_v3b`

- supports both `tight_fixed_point` and `accumulating_fixed_point`
- accumulating runs show:
  - higher `mean_delta`
  - higher `mean_message_norm`
  - higher `mean_message_contraction`
  - richer bottleneck code diversity
  - distinct coupling behavior

This is the first clear evidence in the substrate lab that a fixed-point basin can contain multiple interior trajectory classes.

## Accumulation Trajectory

From [data/substrate_stress/dual_gru_v3b_trajectory_map.json](/home/xenith/demian/data/substrate_stress/dual_gru_v3b_trajectory_map.json):

- message contraction takeoff begins early
- message norm takeoff and slow norm takeoff follow in a repeatable order
- the perturbed trajectory enters the same mode as the clean one
- the perturbation changes local deltas but does not create the mode

Interpretation:

- the accumulation mode is an internal trajectory
- it is not currently best described as a defect
- it should be studied as a structured coordinate of the basin

## Working Regime

`dual_gru_v3b` is now the active substrate, but not as one undifferentiated default.

Use the named regime split:

- `dual_gru_v3b:current`
  - default research regime
  - use this when studying the accumulating interior directly

- `dual_gru_v3b:edge`
  - boundary accumulating regime
  - use this when studying entry into full accumulation without jumping to the strong-expression `current` point

- `dual_gru_v3b:tight`
  - comparison regime for near-entry behavior

- `dual_gru_v3b:threshold`
  - comparison regime for partial occupancy and entry pressure

- `dual_gru_v3b:saturated`
  - overexpression probe, not the day-to-day default

## Why GRU Matters

Only the GRU-derived line has survived these tests in a way that remains bounded and structurally useful.

That does not mean the project should “just use GRU forever.”

It means GRU-style gating currently gives the clearest empirical handle on:

- selective persistence
- bounded overwrite
- timescale separation
- reusable low-dimensional state

This makes GRU the current scaffold for architectural dissection, not the final architecture.

## What To Build Toward

The project goal is still to understand architectures deeply enough to build a custom one.

The current path is:

1. use GRU-derived substrates as the working scaffold
2. identify which mechanisms are doing the real work
3. strip away template-specific assumptions
4. rebuild a custom substrate from preserved principles

Candidate principles:

- selective write
- bounded carry
- asymmetric slow/fast interaction
- persistent message substate
- class-dependent fixed-point interior structure

## Operator Note

If later work suppresses `accumulating_fixed_point` without first understanding what it does, that would likely be a loss of signal.

The right question is:

- what computation or memory geometry does the accumulation trajectory carry?

Not:

- how do we remove it because it looks large?
