# Substrate Lab Dependencies

Last updated: 2026-05-11

## Purpose

This map records which `development/*.py` scripts import symbols from
`development/substrate_lab.py`. Use it before moving research scripts or
extracting active code from the legacy lab.

This is generated manually from import inspection. It is intentionally compact:
the goal is movement safety, not complete call-graph analysis.

## High-Dependency Surfaces

These scripts depend on many private or semi-private helpers and should not be
moved or refactored casually:

| File | Imported legacy symbols |
| --- | --- |
| `development/ablate_native_v74_organs.py` | `_code_stats`, `_decode_code`, `_encode_state`, `_fixed_projection`, `_make_substrate`, `_manual_step_trace`, `_mean_abs_metric_gap`, `_clone_substrate_state` |
| `development/run_v8_vs_v74_coupling_stress.py` | `DemianNativeV8Substrate`, `DemianNativeV74Substrate`, `SelfLoopRunner`, `_make_substrate`, `_fixed_projection`, `_encode_state`, `_decode_code`, `_resolve_substrate_spec` |
| `development/run_substrate_tests.py` | `basin_map`, `bottleneck_run`, `coupled_pair`, `list_substrate_specs`, `memory_pair`, `perturbation_pair`, `save_json`, `summarize_dual_gru_family`, `summarize_dual_gru_v3b_regimes`, `summarize_by_interior_class` |
| `development/sweep_substrate_regimes.py` | `SUBSTRATE_REGISTRY`, `basin_map`, `bottleneck_run`, `coupled_pair`, `memory_pair`, `perturbation_pair`, `regime_score` |

## Active v9 Five-Channel / Demian v1 Dependencies

| File | Imported legacy symbols | Cleanup reading |
| --- | --- | --- |
| `development/probe_v9_message_carrier_strange.py` | via `development.substrates.legacy`: `DemianNativeV9Substrate`, `SelfLoopRunner` | Active probe still subclasses legacy v9 through the extraction boundary. |
| `development/export_v9_release_gate_trajectory_3d.py` | via `development.substrates.legacy`: `SelfLoopRunner` | Exporter depends on runner only through the extraction boundary. |
| `development/substrates/current.py` | via `development.substrates.legacy`: v9/v8/v7.4 classes, `_make_substrate`, `compare_native_v9_vs_v8` | Main workbench no longer imports the 10k-line lab directly. |
| `development/substrates/trajectory_export.py` | via `development.substrates.legacy`: `compare_native_v9_vs_v8`, `trajectory_map` | Export path no longer imports the 10k-line lab directly. |

`development/evolve_v9_5ch_release.py` does not import `development.substrate_lab`;
it carries its own active five-channel evolution implementation.

## Extraction Boundary

[development/substrates/legacy.py](/home/xenith/demian/development/substrates/legacy.py)
is the compatibility boundary. It re-exports the small set of legacy lab
symbols needed by active v9 five-channel and Demian v1 work and injects the legacy substrate registry
into the extracted runner.

[development/substrates/runtime.py](/home/xenith/demian/development/substrates/runtime.py)
now owns the active `SelfLoopRunner`, `StepMetrics`, `RunSummary`, FFT summary,
and fixed-point interior classifier. New active code should import runtime
helpers from focused modules, not directly from `development.substrate_lab`.

## Baseline Dependencies

| File | Imported legacy symbols |
| --- | --- |
| `development/ablate_v74_seed95.py` | `DemianNativeV74Substrate`, `SelfLoopRunner`, `perturbation_stress_test`, `coupling_stress_test` |
| `development/compare_v8_v85_steps.py` | `DemianNativeV8Substrate`, `DemianNativeV85Substrate`, `SelfLoopRunner` |
| `development/evolve_native_v72_metabolism.py` | `coupling_stress_test`, `perturbation_stress_test`, `save_json` |
| `development/probe_v74_ownership_viability_quadrants.py` | `_make_substrate` |
| `development/run_v74_resume_probe.py` | `resume_continuity_probe` |
| `development/run_v8_battery.py` | `DemianNativeV8Substrate`, `SelfLoopRunner`, `perturbation_stress_test` |
| `development/run_v8_coupled_decoupled.py` | `DemianNativeV8Substrate`, `SelfLoopRunner`, `_fixed_projection` |
| `development/run_v8_developmental_trajectory.py` | `DemianNativeV8Substrate`, `SelfLoopRunner` |
| `development/run_v8_multi_ablation.py` | `DemianNativeV8Substrate`, `SelfLoopRunner` |
| `development/run_v8_vs_v74.py` | `DemianNativeV8Substrate`, `DemianNativeV74Substrate`, `SelfLoopRunner`, `perturbation_stress_test`, `coupling_stress_test` |
| `development/sweep_v74_dynamic_step.py` | `bottleneck_run`, `coupled_pair`, `memory_pair`, `perturbation_stress_test`, `resume_continuity_probe` |
| `development/sweep_v74_dynamic_topology.py` | `_make_substrate`, `bottleneck_run`, `coupled_pair`, `memory_pair`, `perturbation_stress_test` |
| `development/sweep_v74_pressure_policy.py` | `_make_substrate` |
| `development/sweep_v8_tightness.py` | `DemianNativeV8Substrate`, `SelfLoopRunner`, `perturbation_stress_test`, `coupling_stress_test` |
| `development/trace_v74_lineage_collapse.py` | `perturbation_stress_test`, `coupling_stress_test` |

## Historical Dependencies

| File | Imported legacy symbols |
| --- | --- |
| `development/ablate_dual_gru_v3b_message.py` | `dual_gru_v3b_message_ablation_suite`, `save_json` |
| `development/compare_dual_gru_v3b_boundary_mechanisms.py` | `save_json`, `trajectory_map_pair` |
| `development/map_dual_gru_v3b_message_transitions.py` | `map_dual_gru_v3b_message_transitions`, `save_json` |
| `development/map_interior_transitions.py` | `map_interior_class_transitions`, `save_json` |
| `development/probe_dual_gru_v3b_edge_window.py` | `amplify_dual_gru_v3b_edge_window`, `probe_dual_gru_v3b_edge_step`, `save_json` |
| `development/probe_dual_gru_v3b_internal_trigger.py` | `save_json`, `self_coupling_schedule_pair` |
| `development/report_dual_gru_family.py` | `save_json`, `summarize_dual_gru_family` |
| `development/report_dual_gru_v3b_autonomy_modes.py` | `save_json`, `state_conditioned_self_trigger_pair`, `classify_dual_gru_v3b_autonomy_mode` |
| `development/report_dual_gru_v3b_regimes.py` | `save_json`, `summarize_dual_gru_v3b_regimes` |
| `development/scan_dual_gru_v3b_edge_anomalies.py` | `save_json`, `scan_dual_gru_v3b_edge_anomalies` |
| `development/search_dual_gru_v3b_internal_trigger_controller.py` | `save_json`, `search_dual_gru_v3b_internal_trigger_controller` |
| `development/search_dual_gru_v3b_state_trigger_controller.py` | `save_json`, `search_dual_gru_v3b_state_trigger_controller` |

## Extraction Guidance

Before extracting from `development/substrate_lab.py`:

1. Keep public helpers behind focused boundaries. `SelfLoopRunner`,
   `StepMetrics`, and `RunSummary` are already extracted into
   `development/substrates/runtime.py`; `_make_substrate`, `save_json`, and
   generic battery functions still need compatibility coverage before movement.
2. Keep private helper movement conservative. Scripts using `_encode_state`,
   `_decode_code`, `_fixed_projection`, or `_manual_step_trace` are tightly
   coupled to legacy internals.
3. Do not change historical scripts just to make imports prettier. Prefer a
   compatibility layer until active behavior has direct tests.
4. Active v9 five-channel and Demian v1 extraction should start from the current workbench and
   `evolve_v9_5ch_release.py`, not from a broad split of the legacy file.
