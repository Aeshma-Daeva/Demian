# Current State And Routing

Last updated: 2026-05-11

Purpose:
- provide one short restart document for OpenClaude/NanoGPT sessions
- point to the live substrate state, claims, experiment rules, and latest result artifacts
- keep model choice explicit instead of using one model for every task

## Start Here

Read in this order:
1. [docs/WORKING_STATE.md](/home/xenith/demian/docs/WORKING_STATE.md)
2. [docs/SUBSTRATE_ANATOMY.md](/home/xenith/demian/docs/SUBSTRATE_ANATOMY.md)
3. [docs/EXPERIMENT_NAMING.md](/home/xenith/demian/docs/EXPERIMENT_NAMING.md)
4. [docs/LABBOOK.md](/home/xenith/demian/docs/LABBOOK.md)
5. [docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md)
6. [docs/EXPERIMENT_RULES.md](/home/xenith/demian/docs/EXPERIMENT_RULES.md)
7. [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py)
8. [data/INDEX.md](/home/xenith/demian/data/INDEX.md) and [data/MANIFEST.json](/home/xenith/demian/data/MANIFEST.json)
9. targeted symbols in [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py)

Operational rule:
- use this file as a session map, not as the claims ledger
- promote durable conclusions through [docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md) and [docs/WORKING_STATE.md](/home/xenith/demian/docs/WORKING_STATE.md)
- put fast-moving experiment details in [docs/LABBOOK.md](/home/xenith/demian/docs/LABBOOK.md), not in repeated current-state prose

## Current Substrate State

Active experimental target:
- Demian v1 program on the v9 five-channel scaffold: `fast`, `slow`, `control`, `message`, `carrier`
- evidence: [development/probe_v9_message_carrier_strange.py](/home/xenith/demian/development/probe_v9_message_carrier_strange.py), [development/evolve_v9_5ch_release.py](/home/xenith/demian/development/evolve_v9_5ch_release.py), [tests/test_v9_5ch_evolution.py](/home/xenith/demian/tests/test_v9_5ch_evolution.py)
- current status: v9 five-channel implemented, evolved for 8 generations in the original archive, then run as `v10.0-frozen-evolution` predecessor evidence for 20 generations across four islands with lineage/scatter/duty diagnostics
- naming rule: do not promote this to `demian_native_v10`; the next named custom-substrate program is `demian-v1`

Current code surface:
- [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py)
- [development/probe_v9_message_carrier_strange.py](/home/xenith/demian/development/probe_v9_message_carrier_strange.py)
- [development/evolve_v9_5ch_release.py](/home/xenith/demian/development/evolve_v9_5ch_release.py)
- use these before opening the full 10k+ line [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py)

Immediate comparison line:
- canonical `demian_native_v9`: 3-channel baseline for the v9 direction
- `demian_native_v8`: 7-channel genotype scaffold with the latest saved result artifacts

Promoted historical baseline:
- `demian_native_v7.4`: organ-heavy ownership/viability line; keep it as a baseline, not as default restart focus

Ancestry/comparison lines:
- `demian_native_v7.2`: constrained-environment metabolic-resource comparison
- `demian_native_v7.1`: trajectory-memory ancestry comparison
- `demian_native_v6`: endogenous-controller comparison
- `demian_native_v5.2c`: same-family anti-locking comparison
- `demian_native_v3`: pre-plasticity tightness-governed comparison

## Latest Result Artifacts

May 10, 2026 artifacts:
- [data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json)
- [data/evolution/v10_0_frozen_evolution_island_1_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_1_20260510/diagnostics.json)
- [data/evolution/v10_0_frozen_evolution_island_2_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_2_20260510/diagnostics.json)
- [data/evolution/v10_0_frozen_evolution_island_3_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_3_20260510/diagnostics.json)
- [data/evolution/v10_0_frozen_evolution_island_4_20260510/diagnostics.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_island_4_20260510/diagnostics.json)

May 9, 2026 artifacts:
- [data/evolution/v9_5ch_release_20260509_full/archive.json](/home/xenith/demian/data/evolution/v9_5ch_release_20260509_full/archive.json)
- [data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/trajectory_3d.json](/home/xenith/demian/data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/trajectory_3d.json)
- [data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/blender_expression/expression.blend](/home/xenith/demian/data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/blender_expression/expression.blend)
- [data/substrate_lab/v9_release_gate_trajectory_3d_20260509/trajectory_3d.json](/home/xenith/demian/data/substrate_lab/v9_release_gate_trajectory_3d_20260509/trajectory_3d.json)

May 8, 2026 artifact:
- [data/substrate_lab/v9_v8_compare_20260508/summary.json](/home/xenith/demian/data/substrate_lab/v9_v8_compare_20260508/summary.json)

May 7, 2026 artifact:
- [data/substrate_lab/v8_multi_ablation_20260507/summary.json](/home/xenith/demian/data/substrate_lab/v8_multi_ablation_20260507/summary.json)

May 6, 2026 artifacts:
- [data/substrate_lab/v8_v74_coupling_stress_20260506/summary.json](/home/xenith/demian/data/substrate_lab/v8_v74_coupling_stress_20260506/summary.json)
- [data/substrate_lab/v8_development_trajectory_20260506/summary.json](/home/xenith/demian/data/substrate_lab/v8_development_trajectory_20260506/summary.json)
- [data/substrate_lab/v8_population_development_20260506/summary.json](/home/xenith/demian/data/substrate_lab/v8_population_development_20260506/summary.json)
- [data/substrate_lab/v8_phase_perturbation_20260506/summary.json](/home/xenith/demian/data/substrate_lab/v8_phase_perturbation_20260506/summary.json)

May 5, 2026 artifacts:
- [data/substrate_lab/v8_baseline_compare_20260505/summary.json](/home/xenith/demian/data/substrate_lab/v8_baseline_compare_20260505/summary.json)
- [data/substrate_lab/v8_tightness_sweep_20260505/summary.json](/home/xenith/demian/data/substrate_lab/v8_tightness_sweep_20260505/summary.json)
- [data/substrate_lab/v74_seed95_ablation_20260505/summary.json](/home/xenith/demian/data/substrate_lab/v74_seed95_ablation_20260505/summary.json)
- [data/substrate_lab/v74_lineage_collapse_20260505/summary.json](/home/xenith/demian/data/substrate_lab/v74_lineage_collapse_20260505/summary.json)
- [data/substrate_lab/v8_channel_ablation_20260505/summary.json](/home/xenith/demian/data/substrate_lab/v8_channel_ablation_20260505/summary.json)
- [data/substrate_lab/v8_bottleneck_sweep_20260505/summary.json](/home/xenith/demian/data/substrate_lab/v8_bottleneck_sweep_20260505/summary.json)
- [data/substrate_lab/v8_population_20260505/summary.json](/home/xenith/demian/data/substrate_lab/v8_population_20260505/summary.json)
- [data/substrate_lab/v85_baseline_20260505/summary.json](/home/xenith/demian/data/substrate_lab/v85_baseline_20260505/summary.json)
- [data/substrate_lab/v8_coupled_decoupled_20260505/summary.json](/home/xenith/demian/data/substrate_lab/v8_coupled_decoupled_20260505/summary.json)

Current read:
- canonical v9 is a 3-channel collapse of the v8 direction: `fast`, `slow`, and `control`
- the active v9 five-channel experiment adds explicit `message` and `carrier` accumulation plus learned/manual/pressure release probes
- v9 five-channel evolution mutates scalar hyperparameters and selected gate/routing matrices with bounded rank-2 runtime low-rank deltas
- in the saved four-seed v9-v8 comparison, both v9 and v8 remain fixed-point across all seeds
- saved aggregate: v9 has lower mean covariance rank than v8, lower compression ratio than v8, and weaker scale-1.0 recovery than v8 in this run
- do not promote v9 superiority from the current artifact; use it as a falsification/diagnostic baseline
- the full v9 five-channel archive preserved multiple regimes: `surface_fixed_accumulating`, `bounded_strange`, `edge_of_chaos`, and small limit-cycle pockets
- release became rarer in archived candidates but release-local causal effect is still weak; next work should improve release consequence, not just increase gain
- `v10.0-frozen-evolution` tested selection pressure over time rather than continued rank tweaking; mean duty fell from early `0.283` to late `0.214`, while event rose from `0.084` to `0.266` and phase rose from `0.152` to `0.450`
- the final predecessor winner was a sparse-to-borderline release candidate, not a flood candidate: `duty=0.171875`, `event=0.7896`, `phase=0.9371`, `rank=5.8039`
- predecessor top-10 lineage diversity contracted while mutation depth rose, supporting heritable selection under eval seed 94; cross-validation is still required before treating the ordering as stable
- fixed-point regimes are not boring by default; inspect internal richness, message/carrier signature retention, and path geometry before judging
- v8 and v7.4 both remain in `FIXED_POINT` / `accumulating_fixed_point` in the saved four-seed baseline compare
- v8 has slightly higher mean covariance rank than v7.4 in that compare, but this is not a promoted superiority claim
- v8 tightness sweeps remain fixed-point; perturbation recovery depends strongly on tightness
- v8 channel ablation suggests `slow` removal hurts least in the saved aggregate, while `control_short` removal hurts recovery most among listed channels
- v8 bottleneck dimension 4 is the saved best point in the bottleneck sweep
- v8 population sweep shows uniform fixed-point class across 16 seeds but nontrivial recovery spread
- v8.5 baseline has low saved perturbation recovery at scale 1.0; treat it as unresolved or currently weaker
- v8 coupled/decoupled runs show small mean coupling shift and persistence with high variance

## Active Experiment Questions

Priority 1:
- `demian-v1-cross-eval`: cross-validate [data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json](/home/xenith/demian/data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json) on held-out eval seeds before promoting genotype-level conclusions

Priority 2:
- `demian-v1-release-causality`: inspect predecessor final and runner-up lineages to isolate how sparse release produces event/phase gains

Priority 3:
- keep canonical v9-v8 comparison as baseline context; do not confuse it with the five-channel evolutionary scaffold

Priority 4:
- keep v7.4 questions available as baseline/falsification work, but do not let them displace v9 as the restart target

Priority 5:
- clean and organize the repo as a separate workstream; preserve active docs/artifact links while separating scratch experiments, archived notes, generated outputs, visualization assets, and OpenClaude utilities

## OpenClaude Token Defaults

Demian launchers default to:

```bash
OPENAI_SHIM_TOOL_MODE=minify
```

This preserves all tools while stripping verbose schema prose. Override only when testing tool-schema behavior:

```bash
OPENAI_SHIM_TOOL_MODE=off python3 scripts/openclaude_nanogpt.py
OPENAI_SHIM_TOOL_MODE=lazy python3 scripts/openclaude_nanogpt.py
```

For usage diagnostics:

```bash
OPENCLAUDE_LOG_TOKEN_USAGE=verbose python3 scripts/openclaude_nanogpt.py --role code
```

## NanoGPT Model Routing

OpenClaude uses one model per session. Route by launching separate sessions with role presets:

```bash
openclaude code
openclaude reason
openclaude cheap
```

Current presets:
- `default`: `deepseek/deepseek-v4-flash` for long-context general work
- `reason`: `deepseek/deepseek-v4-pro` for complex synthesis and hard experiment interpretation
- `code`: `deepseek/deepseek-v4-flash` for the main worker session
- `qwen`: `qwen/qwen3-coder` for comparison runs and Qwen-specific coding trials
- `edit`: `qwen3-coder-30b-a3b-instruct` for cheaper narrow code edits
- `cheap`: `zai-org/glm-4.7-flash` for low-cost search, summarization, and bookkeeping
- `general`: `minimax/minimax-m2.5` for general agent work with tools
- `scout`: `meta-llama/llama-4-scout` for broad-context inexpensive inspection

Routing discipline:
- use the `code` worker session for normal repo work
- use the reasoner helper for high-leverage reasoning checkpoints
- use Qwen Coder for comparison runs and narrow edit trials until it proves more efficient on Demian
- use GLM Flash or MiniMax for doc updates, search, and summarization
- use V4 Flash when the task needs long context but not top reasoning
- keep expensive models out of repetitive read/search/edit loops

Within a worker session, call the reasoner directly instead of switching sessions:

```bash
python3 scripts/ask_reasoner.py "Given these compact metrics, should this claim be promoted?"
jq '.aggregate' data/substrate_lab/v8_baseline_compare_20260505/summary.json |
  python3 scripts/ask_reasoner.py --stdin "Compare v8 and v7.4 conservatively."
```

## Reasoner Checkpoint Protocol

Use the reasoner when the answer changes research direction:
- claim promotion or demotion
- interpreting experiment results
- choosing between architecture branches
- deciding whether an ablation result is causal or telemetry
- debating privacy/cost tradeoffs for model routing

Do not use the reasoner for routine work:
- finding files
- making straightforward edits
- running tests
- summarizing already-promoted docs
- formatting or bookkeeping

Handoff shape:
1. worker gathers evidence with targeted tools
2. worker compresses the evidence with `jq`, `rg`, or a small script
3. worker calls [scripts/ask_reasoner.py](/home/xenith/demian/scripts/ask_reasoner.py)
4. worker compares reasoner output with local analysis
5. final answer states whether the recommendation is worker-only or reasoner-checked

Good handoff:

```bash
jq '{v8:.aggregate.demian_native_v8, v74:.aggregate["demian_native_v7.4"]}' \
  data/substrate_lab/v8_baseline_compare_20260505/summary.json |
  python3 scripts/ask_reasoner.py --stdin \
    "Should this support promoting a v8 claim? Separate observation, inference, risk, next check."
```

Bad handoff:
- sending all of `development/substrate_lab.py`
- sending large raw JSON artifacts when aggregate metrics exist
- asking the reasoner to rediscover file paths the worker can find locally
- treating reasoner output as automatic truth

## Useful Commands

NanoGPT presets:

```bash
openclaude nanogpt --list
openclaude code --help
openclaude nanogpt --model qwen/qwen3-coder-next
```

OpenRouter free rotation:

```bash
openclaude openrouter --list
openclaude openrouter --model minimax/minimax-m2.5:free
```

Model comparison:

```bash
python3 scripts/benchmark_nanogpt_openclaude.py --group core
python3 scripts/benchmark_nanogpt_openclaude.py --group deepseek --task artifact_reasoning
python3 scripts/benchmark_nanogpt_openclaude.py --group coding --task code_navigation
```

Substrate verification:

```bash
./venv/bin/python -m pytest tests/test_current_substrates.py
./venv/bin/python -m pytest tests/test_v9_5ch_evolution.py
./venv/bin/python tests/test_substrate_lab.py
./venv/bin/python -c "from development.substrates.current import compare_current_target; import json; print(json.dumps(compare_current_target(seeds=[94], steps=64), indent=2))"
./venv/bin/python development/run_substrate_stress_tests.py --substrate demian_native_v9
```

V9 five-channel live visualization:

```bash
bash scripts/watch_v9_live.sh
bash scripts/open_v9_blender_expression.sh
```

Known verification caveat:
- full pytest on 2026-05-09: 120 passed, 3 unrelated `tests/test_vibration.py` failures because stale tests expect `snap.attention.mode`; current API returns unlabeled `SpectralShape`
