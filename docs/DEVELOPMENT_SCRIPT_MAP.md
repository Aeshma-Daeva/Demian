# Development Script Map

Last updated: 2026-05-11

## Purpose

This map classifies `development/*.py` before any code movement. It is a
cleanup aid, not a claim ledger. Use it to decide whether a script should stay
active, move behind a workbench facade, or eventually archive.

Classification values:

| Class | Meaning |
| --- | --- |
| `active` | Current v9 five-channel or Demian v1 substrate/evolution work. |
| `baseline` | Still useful comparison surface for v8, v7.4, or native lineage context. |
| `historical` | Older ancestry or completed investigation; keep for traceability. |
| `operational` | Tooling, generated-doc support, summaries, or generic runners. |
| `legacy_core` | Large compatibility implementation that should not be casually moved. |

## Active v9 Five-Channel / Demian v1 Work

| File | Class | Keep first-read? | Notes |
| --- | --- | --- | --- |
| `development/evolve_v9_5ch_release.py` | `active` | yes | Main v9 five-channel and Demian v1 predecessor release evolution surface: scoring, lineage, diagnostics, CUDA worker behavior, CLI. |
| `development/probe_v9_message_carrier_strange.py` | `active` | yes | Message/carrier/release probe surface and experimental five-channel scaffold. |
| `development/probe_v9_5ch_release_params.py` | `active` | targeted | Release scalar sensitivity for archived candidates. |
| `development/export_v9_5ch_evo_trajectory_3d.py` | `active` | targeted | Exports evolved v9 five-channel candidates to 3D trajectory schema. |
| `development/export_v9_release_gate_trajectory_3d.py` | `active` | targeted | Exports release-gate probe trajectories; supports custom seed and perturb-scale CSV arguments for broader comparison sweeps. |
| `development/summarize_v9_5ch_evolution.py` | `active` | targeted | Rebuilds compact summaries from per-island v9 five-channel and predecessor evolution outputs. |
| `development/cross_eval_v10_frozen.py` | `active` | targeted | Held-out CPU cross-evaluation and release ablation driver for archived v10.0 frozen predecessor candidates. |

## Workbench And Operations

| File | Class | Keep first-read? | Notes |
| --- | --- | --- | --- |
| `development/update_docs.py` | `operational` | targeted | Generated status/link/manifest validator. |
| `development/summarize_results.py` | `operational` | targeted | Local compact result summary utility. |
| `development/render_machine_visuals.py` | `operational` | targeted | Renders machine-native diagnostic PNGs from `trajectory_3d.json` exports: event windows, deltas, recurrence, covariance, phase portraits, spectra, and candidate frontiers. |
| `development/run_substrate_tests.py` | `operational` | targeted | Generic self-loop empirical battery runner. |
| `development/run_substrate_stress_tests.py` | `operational` | targeted | Generic stress-test runner. |
| `development/sweep_substrate_regimes.py` | `operational` | targeted | Generic substrate-regime sweep. |

## Current Baseline And Comparison Work

| File | Class | Keep first-read? | Notes |
| --- | --- | --- | --- |
| `development/run_v8_vs_v74.py` | `baseline` | no | Saved v8/v7.4 comparison context. |
| `development/run_v8_vs_v74_coupling_stress.py` | `baseline` | no | Coupling-stress comparison for v8 and v7.4. |
| `development/run_v8_battery.py` | `baseline` | no | Consolidated v8 channel/bottleneck/population battery. |
| `development/run_v8_multi_ablation.py` | `baseline` | no | v8 multi-channel ablation. |
| `development/run_v8_coupled_decoupled.py` | `baseline` | no | v8 coupling trace probe. |
| `development/run_v8_developmental_trajectory.py` | `baseline` | no | v8 per-step channel development probe. |
| `development/sweep_v8_tightness.py` | `baseline` | no | v8 tightness phase-space sweep. |
| `development/compare_v8_v85_steps.py` | `baseline` | no | v8 versus v8.5 step-length comparison. |
| `development/ablate_native_v74_organs.py` | `baseline` | no | v7.4 organ ablation/falsification runner. |
| `development/ablate_v74_seed95.py` | `baseline` | no | v7.4 seed-95 collapse ablation. |
| `development/probe_v74_ownership_viability_quadrants.py` | `baseline` | no | v7.4 ownership/viability quadrant probe. |
| `development/run_v74_resume_probe.py` | `baseline` | no | v7.4 resume-continuity probe. |
| `development/sweep_v74_dynamic_step.py` | `baseline` | no | v7.4 dynamic step sweep. |
| `development/sweep_v74_dynamic_topology.py` | `baseline` | no | v7.4 topology sweep. |
| `development/sweep_v74_pressure_policy.py` | `baseline` | no | v7.4 pressure-policy sweep. |
| `development/trace_v74_lineage_collapse.py` | `baseline` | no | Native lineage collapse trace through v3/v6/v7/v7.1/v7.2/v7.4. |
| `development/evolve_native_v72_metabolism.py` | `baseline` | no | v7.2 metabolic parameter evolution; keep as constrained-environment ancestry. |

## Historical Dual-GRU And Interior-Class Work

| File | Class | Keep first-read? | Notes |
| --- | --- | --- | --- |
| `development/report_dual_gru_family.py` | `historical` | no | Dual-GRU architecture progression report. |
| `development/report_dual_gru_v3b_regimes.py` | `historical` | no | Controllable `dual_gru_v3b` regime report. |
| `development/report_dual_gru_v3b_autonomy_modes.py` | `historical` | no | Autonomy-mode comparison for `dual_gru_v3b`. |
| `development/ablate_dual_gru_v3b_message.py` | `historical` | no | Message-channel ablations for `dual_gru_v3b`. |
| `development/map_dual_gru_v3b_message_transitions.py` | `historical` | no | Message-control interior transition mapping. |
| `development/map_interior_transitions.py` | `historical` | no | Generic fixed-point interior transition mapping. |
| `development/compare_dual_gru_v3b_boundary_mechanisms.py` | `historical` | no | Edge/threshold boundary mechanism comparison. |
| `development/probe_dual_gru_v3b_edge_window.py` | `historical` | no | Edge-boundary timing probe. |
| `development/probe_dual_gru_v3b_internal_trigger.py` | `historical` | no | Internal trigger schedule probe. |
| `development/scan_dual_gru_v3b_edge_anomalies.py` | `historical` | no | Perturbation timing anomaly scan. |
| `development/search_dual_gru_v3b_internal_trigger_controller.py` | `historical` | no | Internal trigger controller search. |
| `development/search_dual_gru_v3b_state_trigger_controller.py` | `historical` | no | State-conditioned trigger controller search. |

## Legacy Core

| File | Class | Keep first-read? | Notes |
| --- | --- | --- | --- |
| `development/substrate_lab.py` | `legacy_core` | no | 10k+ line compatibility source for native lineage, registry, generic runners, and historical helpers. Use targeted symbol ranges only. |
| `development/__init__.py` | `operational` | no | Package marker. |

## Movement Rules

- Do not move `active` files until the lightweight workbench exposes replacement entry points.
- Do not move `baseline` files until their artifact outputs are indexed in `data/MANIFEST.json`.
- `historical` files are archive candidates, but only after import/reference maps and doc links are checked.
- `legacy_core` remains in place until focused modules cover active behavior and compatibility tests pass.

## Default Output Paths

These paths are extracted from script constants or CLI defaults. They are
written as code paths rather than links because not every historical output is
guaranteed to exist locally.

### Active v9 Five-Channel / Demian v1

| File | Default output |
| --- | --- |
| `development/evolve_v9_5ch_release.py` | `data/evolution/v9_5ch_release_20260509` |
| `development/probe_v9_message_carrier_strange.py` | `data/substrate_lab/v9_message_carrier_strange_20260508/summary.json` |
| `development/probe_v9_5ch_release_params.py` | `data/substrate_lab/v9_5ch_release_param_probe` |
| `development/export_v9_5ch_evo_trajectory_3d.py` | `data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509` |
| `development/export_v9_release_gate_trajectory_3d.py` | `data/substrate_lab/v9_release_gate_trajectory_3d_20260509/trajectory_3d.json` |
| `development/summarize_v9_5ch_evolution.py` | caller-provided summary JSON path |
| `development/cross_eval_v10_frozen.py` | `data/substrate_lab/v10_frozen_cross_eval_20260511` |

### Baseline v8/v7.4/v7.2

| File | Default output |
| --- | --- |
| `development/run_v8_vs_v74.py` | `data/substrate_lab/v8_baseline_compare_20260505/summary.json` |
| `development/run_v8_vs_v74_coupling_stress.py` | `data/substrate_lab/v8_v74_coupling_stress_interval1_20260506/summary.json` |
| `development/run_v8_battery.py` | `data/substrate_lab/*` consolidated v8 battery outputs |
| `development/run_v8_multi_ablation.py` | `data/substrate_lab/v8_multi_ablation_20260507/summary.json` |
| `development/run_v8_coupled_decoupled.py` | `data/substrate_lab/v8_coupled_decoupled_20260505/summary.json` |
| `development/run_v8_developmental_trajectory.py` | `data/substrate_lab/v8_development_trajectory_20260506/summary.json` |
| `development/sweep_v8_tightness.py` | `data/substrate_lab/v8_tightness_sweep_20260505/summary.json` |
| `development/compare_v8_v85_steps.py` | `data/substrate_lab/v8_v85_step_compare_20260505/summary.json` |
| `development/ablate_native_v74_organs.py` | `data/substrate_lab/v74_organ_ablation_20260429` |
| `development/ablate_v74_seed95.py` | `data/substrate_lab/v74_seed95_ablation_20260505/summary.json` |
| `development/probe_v74_ownership_viability_quadrants.py` | `data/substrate_lab/v74_ownership_quadrants_20260429` |
| `development/run_v74_resume_probe.py` | `data/substrate_lab/v74_resume_probe_20260429` |
| `development/sweep_v74_dynamic_step.py` | `data/substrate_lab/v74_dynamic_step_sweep_20260429` |
| `development/sweep_v74_dynamic_topology.py` | `data/substrate_lab/v74_dynamic_topology_sweep_20260429` |
| `development/sweep_v74_pressure_policy.py` | `data/substrate_lab/v74_pressure_policy_sweep_20260429` |
| `development/trace_v74_lineage_collapse.py` | `data/substrate_lab/v74_lineage_collapse_20260505/summary.json` |
| `development/evolve_native_v72_metabolism.py` | `data/evolution/v72_metabolism_probe` |

### Historical dual-GRU/interior

| File | Default output |
| --- | --- |
| `development/report_dual_gru_family.py` | `data/substrate_lab/dual_gru_family_report.json` |
| `development/report_dual_gru_v3b_regimes.py` | `data/substrate_lab/dual_gru_v3b_regimes.json` |
| `development/report_dual_gru_v3b_autonomy_modes.py` | `data/substrate_stress/dual_gru_v3b_autonomy_modes.json` |
| `development/ablate_dual_gru_v3b_message.py` | `data/substrate_lab/dual_gru_v3b_message_ablations.json` |
| `development/map_dual_gru_v3b_message_transitions.py` | `data/interior_transitions/dual_gru_v3b_message_transitions.json` |
| `development/map_interior_transitions.py` | `data/interior_transitions` |
| `development/compare_dual_gru_v3b_boundary_mechanisms.py` | `data/substrate_stress/dual_gru_v3b_boundary_mechanism_compare.json` |
| `development/probe_dual_gru_v3b_edge_window.py` | `data/substrate_stress` |
| `development/probe_dual_gru_v3b_internal_trigger.py` | `data/substrate_stress/dual_gru_v3b_internal_trigger_probe.json` |
| `development/scan_dual_gru_v3b_edge_anomalies.py` | `data/substrate_stress/dual_gru_v3b_edge_anomalies.json` |
| `development/search_dual_gru_v3b_internal_trigger_controller.py` | `data/substrate_stress/dual_gru_v3b_internal_trigger_controller_search.json` |
| `development/search_dual_gru_v3b_state_trigger_controller.py` | `data/substrate_stress/dual_gru_v3b_state_trigger_controller_search.json` |

### Operational

| File | Default output |
| --- | --- |
| `development/update_docs.py` | `docs/AUTO_STATUS.md` and generated status blocks |
| `development/summarize_results.py` | terminal summary only |
| `development/render_machine_visuals.py` | `docs/assets/machine_visuals` |
| `development/run_substrate_tests.py` | `data/substrate_lab` |
| `development/run_substrate_stress_tests.py` | `data/substrate_stress` |
| `development/sweep_substrate_regimes.py` | `data/substrate_sweeps` |

## Next Classification Work

- Add a generated or semi-manual import/reference map for scripts that depend
  on `development.substrate_lab`.
- Add a machine-readable script manifest only if manual classification starts drifting.
- After `demian-v1-cross-eval` stabilizes the predecessor evidence, decide whether `evolve_v9_5ch_release.py` should split into scoring, lineage, evaluation, and CLI modules.
