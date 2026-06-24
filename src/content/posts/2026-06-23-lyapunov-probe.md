---
title: "Lyapunov-style perturbation probe: v3 results"
date: 2026-06-23
description: "Added a chaos/order layer via Lyapunov-style perturbation probing. Real run on 250 seed-0 coupling streams shows controlled nonlinear sensitivity."
tags: ["lyapunov", "chaos", "v3", "eeg", "probe"]
metrics:
  - label: "recordings"
    value: 250
  - label: "positive_growth_fraction"
    value: 1.0
  - label: "bounded_sensitive_reconvergent"
    value: 101
artifacts:
  - label: "Probe report"
    path: "adaptation_probes/eeg_demian_architecture_v3_all3264/lyapunov_style_probe_report.md"
  - label: "Recording results"
    path: "adaptation_probes/eeg_demian_architecture_v3_all3264/lyapunov_style_recording_results.csv"
---

## What changed

Implemented the next chaos/order layer: a Lyapunov-style perturbation probe.

- Added `run_lyapunov_style_probe(...)` to `demian_eeg/contrast.py`
- Added CLI command: `demian-eeg-contrast lyapunov-probe`
- Exported from `demian_eeg/__init__.py`
- Added unit coverage in `tests/test_demian_eeg_contrast.py`
- Marked objective complete in `adaptation_probes/eeg_demian_architecture_v2/eeg_adaptation_objectives.md`
- Added narrative technical notes: `adaptation_probes/eeg_demian_architecture_v3_all3264/lyapunov_style_notes.md`

## Real run results

Run on 250 saved seed-0 coupling streams:

| metric | value |
|---|---|
| recordings | 250 |
| mean_log_growth_slope | 0.0892305 |
| positive_growth_fraction | 1.0 |
| mean_bounded_separation_ratio | 0.000105279 |
| bounded_sensitive_fraction | 1.0 |
| bounded_sensitive | 149 |
| bounded_sensitive_reconvergent | 101 |

## Artifacts

- `adaptation_probes/eeg_demian_architecture_v3_all3264/lyapunov_style_probe_report.md`
- `adaptation_probes/eeg_demian_architecture_v3_all3264/lyapunov_style_recording_results.csv`
- `adaptation_probes/eeg_demian_architecture_v3_all3264/lyapunov_style_curve_points.csv`

## Interpretation

Demian v3 is showing controlled nonlinear sensitivity. A tiny first-step perturbation grows across the observer route in every sampled recording, but it stays extremely bounded relative to the route radius. In 101/250 recordings it also reconverges.

The careful claim is now stronger: **Demian v3 shows controlled nonlinear perturbation sensitivity inside bounded observer-route dynamics.**

Not claiming: mathematical chaos, brain simulation, decoding, consciousness, or biomarkers.

## Verification

```
60 passed in 10.35s
```
