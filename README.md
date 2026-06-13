# Demian Lab

Demian Lab studies what happens when AI systems are treated less like text
generators and more like recurrent dynamical systems: state moves, settles,
breaks, resumes, and sometimes carries structure forward.

The goal is to distill that evidence into Demian, a custom recurrent substrate.
This repo is the lab: experiments, comparisons, ablations, claims, figures,
papers, and restart notes. The clean runtime belongs in Demian Substrate; old
paths and superseded notes belong in Demian Archive.

[![Release](https://img.shields.io/github/v/release/Aeshma-Daeva/Demian-Lab?include_prereleases&label=release)](https://github.com/Aeshma-Daeva/Demian-Lab/releases)
[![Tests](https://img.shields.io/badge/tests-pytest%20local-2f6f6a)](#environment)
[![License](https://img.shields.io/github/license/Aeshma-Daeva/Demian-Lab)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-315f8f)](pyproject.toml)
[![Technical Report](https://img.shields.io/badge/technical%20report-GitHub-3b5f7a)](docs/TECHNICAL_REPORT.md)

## What This Lab Is Asking

Most model demos ask, "What did it say?" Demian Lab asks different questions:

- When the system loops through its own state, does it settle, oscillate, or
  keep moving?
- If it looks still on the surface, is the inside actually still?
- Which internal channels matter when a trajectory resumes after interruption?
- Which mechanisms survive ablations, held-out checks, and failed attempts?

The current conservative answer is simple: surface behavior is not enough.
Several experiments look fixed from the outside while internal channels still
carry structured differences. The work now focuses on message, carrier,
control, and gate-like state as the route toward Demian v1.

## Experiment Timeline

| Stage | What was tested | What the data changed |
| --- | --- | --- |
| Transformer self-loop | KV-cache and residual self-contact | Text quality stopped being the target; trajectory geometry became the target. |
| Transformer reservoir | Batch self-reference baseline | Saved runs showed a strong period-2 attractor signature. |
| Mamba reservoir | Same style of self-reference on an SSM | Mamba did not match the transformer 2-cycle; architecture changed the recurrence primitive. |
| Population pressure | Competition, continuity, inheritance-like survival | Continuity and memory survival became first-class measurements. |
| GRU / dual-GRU lab | Message-state ablations and interior-class maps | Fixed point did not mean empty; basin interiors needed measurement. |
| Native v0-v7.4 | Explicit owners for persistence, support, recruitment, control, boundary, memory | Rich but too many mechanisms at once; simplification became necessary. |
| v8 / canonical v9 | Smaller native scaffolds and v9-v8 comparison | v9 was not proven better than v8; fixed-point surface behavior remained common. |
| v9 five-channel | `fast`, `slow`, `control`, `message`, `carrier` plus release probes | Message/carrier accumulation survived; sparse release-vector causality stayed weak. |
| v10.0 predecessor | Four-island frozen evolution | Selection found interesting sparse event/phase candidates, but held-out checks blocked a stable claim. |
| Track A / Track B | Engineered sparse release versus native emergence pressure | Forcing sparse release was less useful than discovering what the substrate naturally used. |
| Demian v1 | Explicit `gate` state added to the five-channel trail | Current synthesis point; contract-tested prototype, not yet a promoted empirical result. |

Full version: [Research Lineage](docs/RESEARCH_LINEAGE.md).

## What Has Evidence

| Finding | Evidence shape | Conservative read |
| --- | --- | --- |
| Transformer and Mamba self-loops differ | Reservoir summaries compare attractor signatures. | Recurrence behavior depends on architecture; do not assume one universal loop. |
| Fixed point can hide internal structure | Dual-GRU, v8/v9, and v9 five-channel diagnostics expose basin-internal differences. | A flat surface label is not enough to judge a substrate. |
| v9 is not proven superior to v8 | Saved v9-v8 comparison: both fixed-point in 4/4 seeds; v9 had lower covariance rank and weaker scale-1.0 recovery in that run. | v9 remains a useful simplified scaffold, not a victory claim. |
| Capsule resume needs internal state | v9 and v9 five-channel full capsules resume almost exactly; surface-only replay fails. | Continuity lives inside the state, not only in the visible surface. |
| v10.0 predecessor found but did not validate sparse release | 640 candidates across four islands; final winner had `duty=0.171875`, `event=0.7896`, `phase=0.9371`; held-out CPU checks weakened the claim. | Useful predecessor and falsification pressure, not a stable mechanism. |
| Track B points at structured channel anatomy | Message/carrier/control probes and null checks show real structure but also controls that prevent overclaiming. | Treat gate-state propagation as the current live hypothesis, not settled proof. |
| Control helps capsule maintenance | Continuous control clamp degrades the capsule more than one-shot control zeroing. | Control is stabilizing; current data do not prove setpoints, selfhood, or narrative integration. |

Details and falsifiers: [Claims Ledger](docs/CLAIMS.md).

## How To Read The Figures

The figures are diagnostic readouts, not decoration. Start with one question,
then open the relevant artifact:

| Question | Start here |
| --- | --- |
| Does internal state matter for resume? | [Capsule continuity](docs/CAPSULE_CONTINUITY.md) and [capsule figure](docs/assets/capsule_continuity_dark.svg) |
| What do the five channels do over time? | [v9 five-channel neuron overview](docs/assets/v9_5ch_neuron_activity_overview.svg) |
| Where are release events and candidate fronts? | [Machine visual artifact group](docs/assets/machine_visuals/) |
| What evidence is safe to cite? | [Claims Ledger](docs/CLAIMS.md) and [Artifact Index](data/INDEX.md) |

The front page intentionally does not embed every plot. Dense visual diagnostics
are useful after the reader knows what question they answer.

## Current Status

- This is an active research notebook, not a finished architecture announcement.
- Demian v1 is a prototype/design consequence of the lineage, not a completed
  empirical result.
- The strongest stable lesson is methodological: measure internal state,
  ablations, resumes, and held-out behavior before naming a mechanism.
- The current live target is to turn the message/carrier/control/gate-state
  evidence into a smaller substrate that survives the same tests.

Quick links: [Research Lineage](docs/RESEARCH_LINEAGE.md) |
[Claims](docs/CLAIMS.md) | [Reproducibility](docs/REPRODUCIBILITY.md) |
[Technical Report](docs/TECHNICAL_REPORT.md) | [Glossary](docs/GLOSSARY.md)

## Orientation

Read in this order if you want the story rather than the file tree:

1. [Research Lineage](docs/RESEARCH_LINEAGE.md): what was tried, what failed,
   what survived.
2. [Claims Ledger](docs/CLAIMS.md): what is actually supported, with
   falsifiers.
3. [Substrate Anatomy](docs/SUBSTRATE_ANATOMY.md): channel and route names.
4. [Capsule Continuity](docs/CAPSULE_CONTINUITY.md): why full internal state
   matters.
5. [Reproducibility](docs/REPRODUCIBILITY.md): compact checks and artifact
   inspection.

Use [Repo Inventory](docs/REPO_INVENTORY.md) and
[Development Script Map](docs/DEVELOPMENT_SCRIPT_MAP.md) only when you need the
file-level map.

## Repo Boundary

| Repo | Purpose |
| --- | --- |
| Demian Lab | Active experiments, diagnostics, claims, papers, and workbench code. |
| Demian Substrate | Stable runtime/package boundary for promoted Demian substrate code. |
| Demian Archive | Historical runtime, superseded notes, and preserved ancestry. |

Main local surfaces:

- `development/`: active research code and substrate workbench.
- `docs/`: claims, lineage, reproducibility, labbook, and publication surfaces.
- `data/`: indexed artifacts and summaries.
- `tests/`: active verification surface.

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

For the capsule-continuity side thread:

```bash
./venv/bin/python development/probe_v9_capsule_continuity.py --hidden-size 16 --pause-steps 24 --resume-steps 24 --seeds 94,95,96 --windows 16:16,24:24
./venv/bin/python -m pytest tests/test_v9_capsule_continuity.py -q
```

## Environment

The local `venv` has the research runtime more reliably than the system interpreter.

Examples:

```bash
./venv/bin/ruff check development tests
./venv/bin/ruff format --check development/lab_schemas.py development/lab_tools.py development/validate_lab_artifact.py tests/test_lab_tooling.py
./venv/bin/python development/validate_lab_artifact.py data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json
./venv/bin/python development/update_docs.py
./venv/bin/python development/run_substrate_tests.py
./venv/bin/python -m pytest tests/test_current_substrates.py
./venv/bin/python -m pytest tests/test_v9_5ch_evolution.py
./venv/bin/python development/summarize_v9_5ch_evolution.py data/evolution/v10_0_frozen_evolution_island_*_20260510 --experiment v10.0-frozen-evolution --eval-seed 94 --out /tmp/demian_v1_predecessor_summary.json
./venv/bin/python -c "from development.substrates.current import compare_current_target; import json; print(json.dumps(compare_current_target(seeds=[94], steps=64), indent=2))"
./venv/bin/python legacy/root_cli/run_mamba_batch.py --runs 3 --steps 1000
./venv/bin/python legacy/root_cli/run_competition.py --n-agents 50 --rounds 500 --machine-driver
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
- Do not dismiss `FIXED_POINT` regimes as boring by default. Current v9 five-channel evidence shows `surface_fixed_accumulating` can coexist with internal richness and other bounded regimes.
- archived notes and plans are useful context, but they are not a substitute for checking raw artifacts.

The correct standard here is not product polish, benchmark wins, or human approval. It is clarity of structure, experimental traceability, and avoiding anthropocentric interpretation drift while isolating machine-useful principles.
