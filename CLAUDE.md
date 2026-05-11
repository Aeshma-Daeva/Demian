# Demian OpenClaude Instructions

## Start Here

Current center of gravity:
- `demian_native_v9` is the active experimental line.
- `demian_native_v8` is the immediate comparison line.
- `demian_native_v7.4` is now a promoted historical baseline, not the default work target.
- current v9 result: [data/substrate_lab/v9_v8_compare_20260508/summary.json](/home/xenith/demian/data/substrate_lab/v9_v8_compare_20260508/summary.json)

Read only these first:
1. [docs/WORKING_STATE.md](/home/xenith/demian/docs/WORKING_STATE.md)
2. [docs/CURRENT_STATE_AND_ROUTING.md](/home/xenith/demian/docs/CURRENT_STATE_AND_ROUTING.md)
3. [docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md)
4. [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py)
5. [data/INDEX.md](/home/xenith/demian/data/INDEX.md)

Do not start by reading all of [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py). For v9, jump to:
- `DemianNativeV9Substrate`
- `compare_native_v9_vs_v8`
- v9 tests in [tests/test_substrate_lab.py](/home/xenith/demian/tests/test_substrate_lab.py)
- the compact v9-v8 result summary, not raw artifacts

Avoid broad work in `development/substrate_lab.py`; it is the large historical
substrate lab. The current objective is v9 with 3D visuals, so use focused v9
modules, current docs, and saved artifacts first.

Current workbench:
- [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py): v9/v8/v7.4 constants, constructors, comparison wrapper, and artifact writer
- [development/substrates/native_v9.py](/home/xenith/demian/development/substrates/native_v9.py): focused v9 import/pointer
- [development/substrates/native_v8.py](/home/xenith/demian/development/substrates/native_v8.py): focused v8 import/pointer
- [development/substrates/native_v74.py](/home/xenith/demian/development/substrates/native_v74.py): focused v7.4 baseline import/pointer

## Worker And Reasoner

Default session role:
- Treat the active OpenClaude session as the worker.
- The worker handles repo navigation, targeted reads, code edits, tests, artifact extraction, and doc updates.

Reasoner escalation:
- When a task requires hard interpretation, claim promotion, architecture tradeoff debate, or synthesis from experiment results, call the reasoner helper yourself.
- Use:

```bash
python3 scripts/ask_reasoner.py "Question and compact evidence here"
```

or pipe compact evidence:

```bash
jq '.aggregate' data/substrate_lab/v8_baseline_compare_20260505/summary.json |
  python3 scripts/ask_reasoner.py --stdin "Should this support promoting a v8 claim?"
```

Rules:
- Do not send giant raw artifacts to the reasoner. Extract compact metrics first with `jq`, `rg`, or small scripts.
- The reasoner gives an advisory checkpoint, not automatic truth.
- Final answers should say when a recommendation is based on worker-only analysis versus a reasoner checkpoint.
- Keep durable claims in [docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md) and promoted working truth in [docs/WORKING_STATE.md](/home/xenith/demian/docs/WORKING_STATE.md).

Current model routing:
- worker preset: `openclaude code` -> `deepseek/deepseek-v4-flash`
- reasoner helper default: `deepseek/deepseek-v4-pro`
- Qwen remains available for comparison through `openclaude qwen` or `openclaude nanogpt --role qwen`

## Token Efficiency

- Prefer targeted `rg`, `jq`, and narrow file reads over broad context loading.
- Treat `data/` as artifact storage. Read `data/INDEX.md` and specific `summary.json` files first; do not dump raw logs or large JSON into chat.
- Treat [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py) as the default current-code surface.
- Use `rg -n "DemianNativeV9Substrate|compare_native_v9_vs_v8"` before opening code ranges.
- Redirect large command output to a file, then inspect compact slices or `jq` aggregates.
- Start a fresh OpenClaude session after broad exploration or large tool output.
- Use the reasoner only at checkpoints where its output can change the decision.
- Keep `OPENAI_SHIM_TOOL_MODE=minify` enabled unless explicitly testing tool schema behavior.
