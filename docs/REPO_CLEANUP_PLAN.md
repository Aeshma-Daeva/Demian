# Repo Cleanup Plan

Last updated: 2026-05-11

## Purpose

The repo needs a deliberate organization pass. Do not combine broad cleanup with
live experiment interpretation or scoring changes. Cleanup should preserve the
ability to reproduce cited artifacts and keep active research surfaces obvious.

## Current Problem

The repo currently mixes:

- legacy runtime package code, formerly under `demian/`
- active substrate/evolution research code under `development/`
- historical substrate lab code
- live docs and generated docs
- archived plans and notes
- large/generated experiment artifacts under `data/`
- visualization assets
- OpenClaude/model-routing scripts
- root-era scratch runners and baseline CLIs

This makes restart context fragile and increases the chance that old ancestry
is mistaken for active direction.

## Cleanup Rules

- Preserve all paths cited by [docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md) and [data/INDEX.md](/home/xenith/demian/data/INDEX.md), or add a migration index before moving them.
- Separate active research code from historical or scratch code.
- Keep generated data out of source-code review unless the artifact itself is the deliverable.
- Do not move files before checking imports, command references, and doc links.
- Keep non-anthropocentric research framing in live docs; archive old human-facing or speculative notes.

## Proposed Target Buckets

- `legacy/demian_runtime/`: old importable runtime package and baseline mechanisms.
- `legacy/root_cli/`: old root-level CLIs and config.
- `development/substrates/`: active substrate implementations and focused helpers.
- `development/evolution/`: current v9 five-channel and Demian v1 helpers plus extracted evolutionary search code.
- `development/archive/`: old probes and one-off scripts retained for ancestry.
- `scripts/`: operational launchers, renderers, dashboards, and model-routing utilities.
- `docs/`: live restart docs, claims, experiment rules, routing, and cleanup notes.
- `docs/archive/`: historical plans, notes, and superseded architecture logs.
- `data/`: artifacts only, with summaries indexed in `data/INDEX.md`.
- `visualization/`: browser/Blender visualization surfaces and static assets.
- `tests/`: active tests matching active code surfaces.

## First Pass

1. Create stable knowledge surfaces: `docs/SUBSTRATE_ANATOMY.md`, `docs/LABBOOK.md`, `docs/REPO_INVENTORY.md`, `docs/DEVELOPMENT_SCRIPT_MAP.md`, `docs/SUBSTRATE_LAB_DEPENDENCIES.md`, and `data/MANIFEST.json`.
2. Update README and restart docs to point to anatomy, labbook, and manifest rather than duplicating latest-run interpretation.
3. Update `development/update_docs.py` to validate the new docs and manifest while avoiding broad manual timestamp churn.
4. Generate a file inventory by bucket: package, development, docs, scripts, tests, visualization, data summaries, generated bulk artifacts.
5. Build an import/reference map for root scripts and `development/*.py`.
6. Identify files that are active entry points versus historical ancestry.
7. Move only archive-safe docs first, because docs already have an archive pattern.
8. Add or update README pointers after each move.
9. Run focused tests after each code move.

## Open Decisions

- Whether to keep reproduction, substrate evaluation, worker orchestration, and
  CLI in `development/evolve_v9_5ch_release.py` or continue moving them into
  `development/evolution/`.
- Whether to physically move native substrate classes out of `development/substrate_lab.py` after the extracted runner has enough coverage.
- Whether large `data/evolution/*/candidates` and `diagnostics` trees should be tracked, ignored, compressed, or summarized only.
- Whether root one-off scripts should move into `scripts/` or `development/archive/`.

## Implemented Slices

- Stable docs and manifest created for anatomy, labbook, repo inventory, script
  classification, and substrate lab dependencies.
- v10 summary generation moved from ad hoc aggregation into
  `development/summarize_v9_5ch_evolution.py`.
- Active substrate workbench added under `development/substrates/current.py`.
- Active runner/runtime layer extracted to
  `development/substrates/runtime.py`, with legacy registry compatibility in
  `development/substrates/legacy.py`.
- Current Demian v1 metadata, predecessor artifact validation, scoring,
  lineage, and generation diagnostics extracted into `development/evolution/`.
- Root-era CLIs and the old `demian/` runtime package moved under `legacy/` so
  the GitHub front page starts with active research, docs, data, tests, and
  operational tools instead of historical entry points.
