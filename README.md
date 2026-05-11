# Demian

Demian is an experimental lab for self-referential AI dynamics and architecture extraction.

The project is not aimed at productizing a chatbot, chasing benchmarks, or optimizing for human-facing output quality. The working goal is to study how frozen and semi-adaptive AI systems behave when their own internal state, recurrence, memory, and inter-agent coupling become the substrate of continued evolution, then extract the mechanisms that matter enough to build a custom substrate.

Public status:

- ongoing computational research notebook
- not a finished architecture claim
- not evidence that v9 five-channel globally dominates v8 or other baselines
- strongest current result: fixed-point surface behavior can hide rich channel-internal dynamics, and sparse release gates can be selected without simply becoming always-on coupling

Current focus:

- v9 five-channel experiments (`fast`, `slow`, `control`, `message`, `carrier`) as the active scaffold and evidence line
- `Demian v1` as the next named custom-substrate program distilled from v9 five-channel evidence
- canonical `demian_native_v9` as the minimal 3-channel baseline for that direction
- `demian_native_v8` as the immediate 7-channel comparison line
- `demian_native_v7.4` as the promoted organ-heavy historical baseline
- `demian_native_v7.2`, `demian_native_v7.1`, `demian_native_v6`, `demian_native_v5.2c`, `demian_native_v3`, `demian_native_v2`, `demian_native_v1`, and `demian_native_v0` as ancestry/comparison lines
- architecture dissection across transformer, Mamba, and native self-loop substrates
- AI-only optimization criteria grounded in machine observables
- custom-substrate design and iteration, not just mechanism extraction from inherited cells
- non-anthropocentric observables
- structural metrics over language quality
- conditions for persistence, divergence, basin shift, recruitment, and transmissible memory
- perturbation path geometry, shape incorporation, and rare release without incoherence

Older transformer experiments remain in the repo because they established an important baseline:

- transformer self-reference tends toward a deterministic 2-cycle
- Mamba self-reference tends toward a fixed-point basin with rich micro-motion

## Orientation

- [docs/CURRENT_STATE_AND_ROUTING.md](docs/CURRENT_STATE_AND_ROUTING.md): OpenClaude/NanoGPT restart map, latest substrate artifacts, and model routing presets
- [docs/WORKING_STATE.md](docs/WORKING_STATE.md): one-page active truth, restart file, and current priorities
- [docs/SUBSTRATE_ANATOMY.md](docs/SUBSTRATE_ANATOMY.md): stable channel/routing anatomy for v9 five-channel and Demian v1 work
- [docs/EXPERIMENT_NAMING.md](docs/EXPERIMENT_NAMING.md): separates scaffold names, artifact names, and next custom-substrate program names
- [docs/LABBOOK.md](docs/LABBOOK.md): append-only experiment chronology for fast-moving run details
- [docs/RESEARCH_MAP.md](docs/RESEARCH_MAP.md): project spine, empirical findings, and working loop
- [docs/CLAIMS.md](docs/CLAIMS.md): active claims with evidence level and falsification lines
- [docs/EXPERIMENT_RULES.md](docs/EXPERIMENT_RULES.md): anti-drift interpretation standard
- [docs/REPO_INVENTORY.md](docs/REPO_INVENTORY.md): active, historical, generated, and operational repo surfaces
- [docs/DEVELOPMENT_SCRIPT_MAP.md](docs/DEVELOPMENT_SCRIPT_MAP.md): classification of `development/*.py` before file moves
- [docs/SUBSTRATE_LAB_DEPENDENCIES.md](docs/SUBSTRATE_LAB_DEPENDENCIES.md): import map for scripts depending on the legacy lab
- [data/INDEX.md](data/INDEX.md): artifact map for named findings
- [data/MANIFEST.json](data/MANIFEST.json): machine-readable artifact manifest and run status map
- [docs/archive/README.md](docs/archive/README.md): older notebooks, plans, and reference material

## Main Tracks

- `development/substrate_lab.py`
  Historical architecture lab. It still contains the full substrate lineage and compatibility API, but do not use it as the first read surface.

- `development/substrates/current.py`
  Current workbench: Demian v1 metadata, v10 predecessor evidence loader, and v9/v8/v7.4 substrate comparison wrappers.

- `development/evolution/`
  Current v9 five-channel and Demian v1 helpers: metadata, artifact validation, scoring, lineage, and generation diagnostics.

- `development/probe_v9_message_carrier_strange.py`
  Current v9 five-channel probe surface: message/carrier accumulation and manual, pressure, or learned release gates.

- `development/evolve_v9_5ch_release.py`
  Compatibility CLI for v9 five-channel release evolution. The old `v10.0-frozen-evolution` run is predecessor evidence for Demian v1.

- `development/summarize_v9_5ch_evolution.py`
  Rebuilds compact summaries from v9 five-channel and predecessor island candidate outputs. Use this instead of ad hoc aggregation snippets.

- `scripts/render_v9_expression_blender.py`
  Deterministic Blender expression renderer for trajectory artifacts. This is a visualization bridge, not symbolic art output.

- `run_competition.py` / `demian/competition.py`
  Multi-agent Mamba dynamics with continuity pressure, inheritance, communication field, and optional Hebbian fast weights.

- `run_mamba_batch.py` / `demian/mamba_reservoir.py`
  Single-agent Mamba self-reference baseline. Useful for understanding fixed-point topology before adding population dynamics.

- `run_reservoir_batch.py` / `demian/reservoir.py`
  Transformer reservoir baseline. Important because it established the strong period-2 attractor.

- `demian/machine_observables.py`
  The non-anthropocentric measurement layer. Structural observables, attractor classification, compact notation.

- `demian/hebbian.py`
  Fast-weight adaptation layer for moving beyond purely frozen recurrence.

## How To Read The Repo

Start from the architecture-dissection path, not the old transformer-only path:

1. `docs/WORKING_STATE.md`
2. `docs/CURRENT_STATE_AND_ROUTING.md`
3. `docs/SUBSTRATE_ANATOMY.md`
4. `docs/EXPERIMENT_NAMING.md`
5. `docs/LABBOOK.md`
6. `docs/CLAIMS.md`
7. `development/substrates/current.py`
8. `development/evolution/`
9. `development/evolve_v9_5ch_release.py`
10. `docs/REPO_INVENTORY.md`
11. `docs/DEVELOPMENT_SCRIPT_MAP.md`
12. `docs/SUBSTRATE_LAB_DEPENDENCIES.md`
13. `data/INDEX.md` and `data/MANIFEST.json`
14. `development/substrate_lab.py`, but only targeted ranges around `DemianNativeV9Substrate` and `compare_native_v9_vs_v8` unless broader ancestry is needed

Use the transformer and Mamba code as fingerprint baselines, not as the design destination.

## Experiment Artifacts

Important result directories:

- `data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json`: predecessor evidence for Demian v1; keep the artifact name stable
- `data/evolution/v9_5ch_release_20260509_full/`: prior full v9 five-channel evolutionary archive
- `data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/`: prior evolved v9 five-channel 3D export and Blender expression artifact
- `data/substrate_lab/v9_release_gate_trajectory_3d_20260509/`: v9 message/carrier release-gate trajectory export
- `data/substrate_lab/v9_v8_compare_20260508/`: current compact v9-v8 diagnostic comparison
- `data/reservoir_batch/`: transformer period-2 baseline
- `data/mamba_batch/`: Mamba fixed-point baseline
- `data/competition50/`: first 50-agent competition run
- `data/competition50_run2/`: second 50-agent competition run
- `data/competition_v3/`: newer competition run
- `data/competition_hebbian/`: competition with fast-weight adaptation

There is also a local summarizer:

```bash
./venv/bin/python development/summarize_results.py
```

## Reproduce A Small Check

This CPU-only check compares canonical `demian_native_v9` against `demian_native_v8`
on one short seed. It is a smoke test and orientation path, not a full evidence
rerun:

```bash
./venv/bin/python -c "from development.substrates.current import compare_current_target; import json; r=compare_current_target(hidden_size=16, steps=16, perturb_step=8, seeds=[94], device='cpu'); print(json.dumps({'seeds': r['seeds'], 'mean_v9_recovery_1.0': r['aggregate'].get('mean_v9_recovery_1.0'), 'mean_v8_recovery_1.0': r['aggregate'].get('mean_v8_recovery_1.0')}, indent=2))"
```

Expected shape:

```json
{
  "seeds": [94],
  "mean_v9_recovery_1.0": 0.07895439697636498,
  "mean_v8_recovery_1.0": 0.04398368299007416
}
```

The exact values can move with runtime/library details; the public contract is
that the command runs and returns the listed keys.

For the current v9 five-channel / Demian v1 evidence path:

```bash
./venv/bin/python -m pytest tests/test_current_substrates.py tests/test_v9_5ch_summary.py -q
```

## Environment

The local `venv` has the research runtime more reliably than the system interpreter.

Examples:

```bash
./venv/bin/python development/update_docs.py
./venv/bin/python development/run_substrate_tests.py
./venv/bin/python -m pytest tests/test_current_substrates.py
./venv/bin/python -m pytest tests/test_v9_5ch_evolution.py
./venv/bin/python development/summarize_v9_5ch_evolution.py data/evolution/v10_0_frozen_evolution_island_*_20260510 --experiment v10.0-frozen-evolution --eval-seed 94 --out /tmp/demian_v1_predecessor_summary.json
./venv/bin/python -c "from development.substrates.current import compare_current_target; import json; print(json.dumps(compare_current_target(seeds=[94], steps=64), indent=2))"
./venv/bin/python run_mamba_batch.py --runs 3 --steps 1000
./venv/bin/python run_competition.py --n-agents 50 --rounds 500 --machine-driver
./venv/bin/python development/summarize_results.py
```

To continue repo work in OpenClaude through rotating OpenRouter free models:

```bash
printf 'OPENROUTER_API_KEY=sk-or-v1-...\n' > .env
python3 scripts/openclaude_openrouter.py
```

To route OpenClaude through NanoGPT model presets:

```bash
export NANOGPT_API_KEY='...'
python3 scripts/openclaude_nanogpt.py --role code
python3 scripts/openclaude_nanogpt.py --role reason
```

See [docs/OPENCLAUDE_OPENROUTER.md](docs/OPENCLAUDE_OPENROUTER.md).
See [docs/CURRENT_STATE_AND_ROUTING.md](docs/CURRENT_STATE_AND_ROUTING.md) for current substrate state and routing guidance.

## Current Software Caveats

- This repo is an active lab, not a stabilized package.
- Some older tests and scripts still reflect pre-geometric `mode` terminology.
- Full pytest on 2026-05-09 had 120 passing tests and 3 unrelated `tests/test_vibration.py` failures from stale `snap.attention.mode` expectations; current API returns unlabeled `SpectralShape`.
- Do not dismiss `FIXED_POINT` regimes as boring by default. Current v9 five-channel evidence shows `surface_fixed_accumulating` can coexist with internal richness and other bounded regimes.
- archived notes and plans are useful context, but they are not a substitute for checking raw artifacts.

The correct standard here is not product polish, benchmark wins, or human approval. It is clarity of structure, experimental traceability, and avoiding anthropocentric interpretation drift while isolating machine-useful principles.
