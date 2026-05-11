# Repo Inventory

Last updated: 2026-05-11

## Purpose

This document maps the repo into active surfaces, historical ancestry, generated
artifacts, and operational tooling. Use it before moving files. It is not a
claims document and should not carry experiment interpretation.

## Active Restart Surface

Read these first:

1. [docs/WORKING_STATE.md](/home/xenith/demian/docs/WORKING_STATE.md)
2. [docs/SUBSTRATE_ANATOMY.md](/home/xenith/demian/docs/SUBSTRATE_ANATOMY.md)
3. [docs/EXPERIMENT_NAMING.md](/home/xenith/demian/docs/EXPERIMENT_NAMING.md)
4. [docs/LABBOOK.md](/home/xenith/demian/docs/LABBOOK.md)
5. [docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md)
6. [data/MANIFEST.json](/home/xenith/demian/data/MANIFEST.json)
7. [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py)
8. [docs/DEVELOPMENT_SCRIPT_MAP.md](/home/xenith/demian/docs/DEVELOPMENT_SCRIPT_MAP.md)
9. [docs/SUBSTRATE_LAB_DEPENDENCIES.md](/home/xenith/demian/docs/SUBSTRATE_LAB_DEPENDENCIES.md)

## Code Buckets

| Bucket | Current role | Cleanup direction |
| --- | --- | --- |
| `demian/` | Importable package code and baseline mechanisms | Keep package-only; avoid adding one-off research scripts here. |
| `development/substrates/` | Compact active substrate workbench, runtime helpers, and focused import surfaces | Expand as the lightweight substrate lab facade. |
| `development/evolution/` | v9 five-channel and Demian v1 helpers for metadata, artifact validation, scoring, lineage, and diagnostics | Keep as the active evolution package. |
| `development/substrate_lab.py` | Legacy full substrate lineage and compatibility API | Keep stable until focused workbench coverage is complete. |
| `development/evolve_v9_5ch_release.py` | Compatibility CLI for v9 five-channel release evolution | Keep runnable while helpers move into `development/evolution/`. |
| `development/summarize_v9_5ch_evolution.py` | Reproducible summary builder for island outputs | Keep active; use instead of ad hoc aggregation snippets. |
| `development/probe_v9_message_carrier_strange.py` | Active five-channel probe surface | Keep active while message/carrier/release remains the center. |
| `development/run_*`, `development/sweep_*`, `development/probe_*`, `development/report_*` | Mixed historical and active research scripts | Classify before moving; do not bulk archive by filename alone. |
| `scripts/` | Operational launchers, renderers, dashboards, model routing | Keep operational tools here. |
| `visualization/` | Browser 3D trajectory viewer | Keep visualization assets separate from experiment code. |
| `tests/` | Active verification surface | Split only after code surfaces split. |

## Active Development Entry Points

- [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py)
  - compact workbench for active substrate work
  - exposes Demian v1 metadata, current substrate specs, entrypoints, v9/v8 comparison, predecessor summary loader, and manifest loader
- [development/evolution/](/home/xenith/demian/development/evolution)
  - current Demian v1 metadata, predecessor artifact validation, scoring, lineage, and generation diagnostics
- [development/substrates/legacy.py](/home/xenith/demian/development/substrates/legacy.py)
  - extraction boundary over selected legacy lab symbols used by active work
- [development/substrates/runtime.py](/home/xenith/demian/development/substrates/runtime.py)
  - extracted runner, per-step metrics, run summaries, and fixed-point interior classification
- [development/evolve_v9_5ch_release.py](/home/xenith/demian/development/evolve_v9_5ch_release.py)
  - compatibility CLI for v9 five-channel release evolution
  - still owns substrate evaluation, CUDA worker handling, reproduction, and CLI orchestration
- [development/probe_v9_message_carrier_strange.py](/home/xenith/demian/development/probe_v9_message_carrier_strange.py)
  - message/carrier/release probe surface
- [development/substrates/trajectory_export.py](/home/xenith/demian/development/substrates/trajectory_export.py)
  - current 3D trajectory export path
- [development/summarize_v9_5ch_evolution.py](/home/xenith/demian/development/summarize_v9_5ch_evolution.py)
  - rebuilds compact summaries from island candidate files

## Legacy Lab Boundary

[development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py)
is still the implementation source for most native classes and comparison
helpers. The active runner layer has moved to
[development/substrates/runtime.py](/home/xenith/demian/development/substrates/runtime.py),
with legacy registry wiring in
[development/substrates/legacy.py](/home/xenith/demian/development/substrates/legacy.py).
The old lab is too large for default session context. Use targeted symbol
reads:

```bash
rg -n "class DemianNativeV9Substrate|compare_native_v9_vs_v8" development/substrate_lab.py
rg -n "SUBSTRATE_REGISTRY|def _make_substrate" development/substrate_lab.py
```

Do not split this file until:

- active workbench tests cover the target behavior
- docs and manifest no longer cite broad legacy ranges as primary evidence
- import/reference maps identify dependent scripts

## Data Buckets

| Bucket | Current role |
| --- | --- |
| `data/MANIFEST.json` | Machine-readable run and artifact map. |
| `data/INDEX.md` | Human-readable artifact index. |
| `data/evolution/` | v9 five-channel and Demian v1 predecessor raw runs, summaries, diagnostics, candidates. |
| `data/substrate_lab/` | substrate comparison summaries and trajectory exports. |
| `data/substrate_stress/` | older stress/autonomy artifacts. |
| `data/reservoir_batch/` | transformer self-reference baseline. |
| `data/mamba_batch/` | Mamba self-reference baseline. |
| `data/competition*` | population/competition runs and checkpoints. |

Raw data remains ignored by default. Track only curated manifest/index files
unless a raw artifact is explicitly the deliverable.

## Documentation Buckets

| File | Role |
| --- | --- |
| `WORKING_STATE.md` | Restart orientation and current priorities. |
| `CURRENT_STATE_AND_ROUTING.md` | Operational session/model routing. |
| `SUBSTRATE_ANATOMY.md` | Stable channel, metric, regime, and lineage anatomy. |
| `EXPERIMENT_NAMING.md` | Scaffold, artifact, and custom-substrate naming rules. |
| `LABBOOK.md` | Append-only experiment chronology. |
| `CLAIMS.md` | Promoted claims with evidence and falsifiers. |
| `EXPERIMENT_RULES.md` | Interpretation and doc ownership rules. |
| `RESEARCH_MAP.md` | Stable conceptual spine and historical map. |
| `REPO_CLEANUP_PLAN.md` | Cleanup execution plan. |
| `REPO_INVENTORY.md` | File and responsibility map. |
| `DEVELOPMENT_SCRIPT_MAP.md` | Classification of `development/*.py` before file movement. |
| `SUBSTRATE_LAB_DEPENDENCIES.md` | Import map for scripts depending on `development/substrate_lab.py`. |
| `docs/archive/` | Historical notes and superseded plans. |

## Next Cleanup Slices

1. Continue splitting `development/evolve_v9_5ch_release.py` by moving substrate
   evaluation, CUDA worker handling, reproduction, and CLI orchestration after
   each slice has focused tests.
2. Move only archive-safe historical scripts after import/reference checks.
