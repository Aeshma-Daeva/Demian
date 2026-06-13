# Gate-State Causal Propagation Article Plan

Last updated: 2026-05-15

Purpose: define the smaller first arXiv article focused only on the replicated
Gate-State Causal Propagation result.

## Article Position

Recommended title:

> Gate-State Causal Propagation in a Structured Recurrent Substrate

Alternate titles:

- Discovering Gate-State Causal Propagation in Structured Recurrent Substrates
- Causal Gate-State Propagation in a Five-Channel Recurrent Substrate
- Internal Gate-State Propagation Under Ablation in Recurrent Substrates

Recommended article type:

> Compact empirical mechanism paper.

This should not be the full Demian methods paper. It should introduce only the
minimum Demian vocabulary needed to understand one result, then spend most of
its space on the diagnostic protocol, replication, caveats, and artifact-backed
evidence.

Core thesis:

> In a structured five-channel recurrent substrate, native-emergence search
> repeatedly discovered a mechanism where gate/release-state history changes
> downstream dynamics even when explicit release gain is zeroed. The mechanism
> replicated in 3/3 Track B runs, with repeated message/carrier necessity and
> full internal-state capsule continuity.

## Claim Boundary

Claim directly:

- Gate-State Causal Propagation is an observed mechanism in the current v9
  five-channel Track B protocol.
- Three Track B native-emergence replications passed the mechanism gate.
- Held-out route-disabled and gain-zero divergence were positive in 3/3
  replications.
- Mean held-out route-disabled and gain-zero divergence were both `0.2997`.
- Gain-zero diagnostics remained clean in 3/3 top-candidate held-out
  classifications.
- `message` and `carrier` were the top two necessary channels in 3/3
  replications.
- Full internal-state resume was exact in 3/3, while surface-only resume had a
  positive gap in 3/3.

Do not claim:

- that the mechanism is universal across recurrent architectures;
- that the substrate is benchmark-superior;
- that sparse delayed release was validated;
- that high-duty Track B candidates satisfy the original sparse-release target;
- that Demian v1 is validated;
- that the mechanism implies general intelligence or consciousness.

One-sentence caveat to repeat in abstract, results, and conclusion:

> This is not a sparse-release success claim; the replicated candidates are
> high-duty native-emergence phenotypes under the current protocol.

## Paper Structure

### 1. Introduction

Goal: make the reader care about the mechanism without needing the full Demian
history.

Draft logic:

- Recurrent systems can expose stable or similar surface behavior while
  differing in internal state dynamics.
- Mechanism discovery therefore needs interventions that separate surface
  output, route effects, internal channels, and resumable state.
- This paper reports a compact result: Gate-State Causal Propagation in a
  five-channel recurrent substrate.
- Contributions:
  - a diagnostic framing for gate-state propagation;
  - a replicated empirical finding in 3/3 Track B runs;
  - channel-necessity and capsule-continuity evidence;
  - a negative boundary: the result is not sparse delayed release.

### 2. Substrate And Terminology

Keep this section short.

Required terms:

- substrate: recurrent dynamical system;
- surface: observable readout vector;
- channel: factorized recurrent state component;
- route: transformation between state components;
- release/gate state: internal route-modulating state;
- capsule continuity: pause/resume comparison using full internal state vs
  surface-only replay.

Substrate:

- v9 five-channel scaffold;
- channels: `fast`, `slow`, `control`, `message`, `carrier`;
- study substrate only, not the final Demian v1 architecture.

Avoid long historical lineage. Put historical detail in a short paragraph or
appendix pointer.

### 3. Diagnostic Protocol

This section is the technical center of the article.

Protocol elements:

- Track B native-emergence search selects candidates for internally useful
  mechanisms rather than sparse/timed release alone.
- Characterization grid:
  - seeds `94-102`;
  - perturb scales `0.2`, `0.35`, `0.7`;
  - motifs `basis:0` and `gaussian:0`;
  - `128` steps;
  - perturb step `64`.
- Interventions:
  - original;
  - routes-disabled;
  - gain-zero;
  - per-channel disabled arms.
- Channel-disabled semantics:
  - advance one step;
  - zero the requested channel;
  - record the row;
  - feed the clamped state into the next step.
- Capsule probe:
  - uninterrupted continuation;
  - full internal-state resume;
  - surface-only replay;
  - channel-only resumes where relevant.

Define the mechanism gate compactly:

> A candidate passes when route-disabled and gain-zero divergence remain
> positive, gain-zero cleanliness holds, message/carrier necessity repeats, and
> capsule evidence separates full internal-state resume from surface-only
> replay.

### 4. Results

Recommended subsections:

1. Replication summary
   - 3/3 Track B native-emergence replications passed.
   - Mean held-out route/gain-zero divergence: `0.2997`.
   - Held-out route/gain-zero divergence range: `0.2422-0.3375`.

2. Channel necessity
   - `message` and `carrier` were top two in 3/3.
   - Run 1 order: `carrier,message,slow,control`.
   - Run 2 order: `message,carrier,control,slow`.
   - Run 3 order: `carrier,message,slow,control`.
   - `slow` support remained high.

3. Gain-zero and route-disabled behavior
   - Gain-zero diagnostics clean in 3/3.
   - Positive divergence under both route-disabled and gain-zero comparisons.
   - Interpretation: gate/release-state history changes downstream dynamics
     beyond simple additive release gain.

4. Capsule continuity
   - Full internal-state resume exact in 3/3.
   - Surface-only final gap positive in 3/3:
     `0.6637`, `0.7789`, `0.3565`.
   - Interpretation: surface state is insufficient to resume the computation
     under this probe.

5. Negative boundary
   - The result does not validate sparse delayed release.
   - Top Track B candidates are high-duty native-emergence phenotypes.
   - The mechanism should be separated from route-specific sparse-release
     success.

### 5. Discussion

Discuss what the result means:

- The replicated effect supports treating gate/release history as an internal
  computational state, not only as an output-side switch.
- `message` and `carrier` look like necessary routed state channels under the
  current diagnostics.
- Surface-only replay failing supports internal-state-centered evaluation.
- The result motivates explicit gate state in Demian v1, but does not validate
  Demian v1.

Discuss what remains unresolved:

- breadth across architectures;
- independent implementation;
- stronger held-out protocols;
- whether a lower-duty version can preserve the mechanism;
- whether compressed capsules can retain the continuity effect.

### 6. Reproducibility

Point to compact checks, not full evolutionary reruns.

Minimum reproducibility text:

```bash
./venv/bin/python -m pytest tests/test_gate_state_propagation_characterization.py tests/test_v9_capsule_continuity.py -q
```

Artifact inspection:

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

Primary artifact paths:

- `data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json`
- `data/diagnostics/gate_state_propagation_characterization_20260511/summary.json`
- `data/diagnostics/gate_state_track_b_replication_1_heldout_20260511/summary.json`
- `data/diagnostics/gate_state_track_b_replication_2_heldout_20260511/summary.json`
- `data/diagnostics/gate_state_track_b_replication_3_heldout_20260511/summary.json`

## Figures And Tables

Keep the article visually compact.

Required:

- Figure 1: five-channel scaffold and intervention points.
- Figure 2: diagnostic protocol schematic showing original, routes-disabled,
  gain-zero, and channel-disabled arms.
- Table 1: 3/3 replication summary.
- Table 2: channel necessity and capsule resume summary.

Optional:

- Figure 3: capsule continuity comparison.
- Appendix table: full per-run held-out divergence ranges.

Avoid:

- broad Demian lineage figures;
- Blender/3D inspection visuals;
- figures about identity continuity, control maintenance, or Demian v1 unless
  placed in future-work context.

## Draft Abstract

Structured recurrent systems can expose stable or similar readouts while
preserving distinct internal state dynamics. We report Gate-State Causal
Propagation, a replicated mechanism discovered in a five-channel recurrent
substrate with `fast`, `slow`, `control`, `message`, and `carrier` state. Under
a Track B native-emergence protocol, three independent replications passed a
mechanism gate: held-out route-disabled and gain-zero divergence remained
positive in 3/3 runs, gain-zero diagnostics remained clean, `message` and
`carrier` were the top two necessary channels in 3/3 channel-disabled
classifications, and full internal-state resume was exact while surface-only
resume left positive gaps. The result suggests that gate/release-state history
can alter downstream dynamics beyond simple additive route injection. This is
not a sparse-release success claim: the replicated candidates are high-duty
native-emergence phenotypes under the current protocol. We present the
diagnostic procedure, replication artifacts, and limitations, and frame
explicit gate state as a design hypothesis for later work rather than as a
validated architecture.

Metadata note: this abstract is intentionally claim-limited. Before arXiv
submission, count the final metadata abstract against arXiv's 1920-character
limit.

## Reference Strategy

Use fewer references than the full Demian methods paper. The article only needs
enough context to place the result.

Core references:

- RNN dynamical reverse engineering:
  <https://arxiv.org/abs/1906.10720>
- Recent RNN mechanistic interpretability:
  <https://arxiv.org/abs/2510.25674>
- RNN solution degeneracy across behavior/dynamics/weights:
  <https://arxiv.org/abs/2410.03972>
- RNNs and attractor reconstruction:
  <https://arxiv.org/abs/2306.04406>
- Dynamical comparison beyond static geometry:
  <https://arxiv.org/abs/2410.24070>

Optional adjacent framing:

- Transformer residual stream dynamics:
  <https://arxiv.org/abs/2502.12131>

Do not overload the result paper with broad AI, consciousness, or philosophy
references.

## Acceptance Criteria For The Manuscript

- The title and abstract mention Gate-State Causal Propagation directly.
- The introduction can be understood without reading the full Demian repo.
- Every numeric result in the paper appears in a cited artifact.
- The paper includes the high-duty caveat in the abstract or introduction.
- The result section reports all three replications, not only the mean.
- The capsule-continuity result is described as full-state resume, not
  compressed memory.
- Demian v1 appears only in discussion/future work.
- The paper has no benchmark-superiority, intelligence, consciousness, or
  finished-architecture claim.
- The reproducibility section includes compact commands and artifact paths.
