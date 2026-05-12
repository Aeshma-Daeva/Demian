# Reproducibility

Last updated: 2026-05-12

These are compact CPU checks for orientation and artifact inspection. They do
not rerun the full evolutionary searches. Use the repo virtual environment when
available:

Public repository: <https://github.com/Aeshma-Daeva/Demian>

```bash
./venv/bin/python -m pytest tests/test_v9_capsule_continuity.py tests/test_gate_state_propagation_characterization.py tests/test_v9_5ch_evolution.py tests/test_demian_v1_gate_state.py -q
```

Expected shape: the focused test suite passes and exercises capsule continuity,
gate-state characterization helpers, v9 five-channel evolution contracts, and
the Demian v1 gate-state prototype.

## Capsule-Continuity Probe

```bash
./venv/bin/python development/probe_v9_capsule_continuity.py --hidden-size 16 --pause-steps 24 --resume-steps 24 --seeds 94,95,96 --windows 16:16,24:24 --device cpu --out /tmp/demian_capsule_continuity_summary.json
```

Expected shape:

- JSON is written to `/tmp/demian_capsule_continuity_summary.json`.
- `aggregate.demian_native_v9.n_runs` and `aggregate.v9_five_channel.n_runs`
  are positive.
- Full capsules are exact or near-exact under the deterministic probe.
- Surface-only replay is worse than full capsule resume.

This supports the internal-state continuity claim in
[CLAIMS.md C19](CLAIMS.md#c19-v9-and-v9-five-channel-resume-from-internal-state-capsules-while-surface-only-replay-fails).

## v9/v8 Smoke Comparison

```bash
./venv/bin/python -c "from development.substrates.current import compare_current_target; import json; r=compare_current_target(hidden_size=16, steps=16, perturb_step=8, seeds=[94], device='cpu'); print(json.dumps({'seeds': r['seeds'], 'mean_v9_recovery_1.0': r['aggregate'].get('mean_v9_recovery_1.0'), 'mean_v8_recovery_1.0': r['aggregate'].get('mean_v8_recovery_1.0')}, indent=2))"
```

Expected shape:

- Output is a JSON object with `seeds`, `mean_v9_recovery_1.0`, and
  `mean_v8_recovery_1.0`.
- The command is a smoke comparison, not a superiority claim.

This is the compact baseline/falsification path referenced in
[README.md](../README.md#reproduce-a-small-check).

## Gate-State Artifact Inspection

```bash
./venv/bin/python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json").read_text())
print(json.dumps({
    "replication_count": summary.get("replication_count"),
    "passed_replication_count": summary.get("passed_replication_count"),
    "message_carrier_top2_count": summary.get("message_carrier_top2_count"),
    "mean_heldout_routes_divergence": summary.get("mean_heldout_routes_divergence"),
    "mean_heldout_gain_zero_divergence": summary.get("mean_heldout_gain_zero_divergence"),
}, indent=2, sort_keys=True))
PY
```

Expected shape:

- The JSON includes replication counts and mean held-out divergence fields.
- The strong claim is the 3/3 replicated Track B mechanism summarized in
  [CLAIMS.md C20](CLAIMS.md#c20-gate-state-causal-propagation-replicates-in-track-b-native-emergence-runs).
- This is artifact inspection, not a new evolution run.

For a small characterization smoke run that writes temporary artifacts:

```bash
./venv/bin/python development/gate_state_propagation_characterization.py --smoke --steps 16 --perturb-step 8 --capsule-pause-step 8 --hidden-size 16 --device cpu --output-dir /tmp/demian_gate_state_smoke
```

Expected shape: the command writes `/tmp/demian_gate_state_smoke/summary.json`
and prints an `evidence_gate` object. The smoke run checks the code path; use
the saved replication artifacts for the publication claim.

## Demian v1 Gate-State Prototype

```bash
./venv/bin/python -m pytest tests/test_demian_v1_gate_state.py -q
```

Expected shape:

- Tests verify the six-channel prototype exposes `gate` as internal state.
- Surface-only resume preserves the surface while clearing the internal capsule.
- Gate-disabled/frozen behavior and route modulation contracts are exercised.

This is a design-contract check. It does not validate Demian v1 as a finished
architecture.
