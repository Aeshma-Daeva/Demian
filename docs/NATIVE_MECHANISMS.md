# Native Mechanisms

This catalog records coherent strategies discovered by native emergence runs or
by engineered-track bypasses. Entries are evidence records, not endorsements of
the current target metric.

## Gate-State Causal Propagation

- Status: replicated core mechanism
- First observed: Demian v1 delayed-eligibility four-island run, 2026-05-11
- Confirmed diagnostic run: Demian v2 dual-track run, 2026-05-11
- Focused Track B characterization: `data/diagnostics/gate_state_propagation_characterization_20260511/summary.json`
- Track B replication summary: `data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json`
- Representative Track B genotype: `data/evolution/demian_v2_track_b_island_1_20260511/candidates/gen012_candidate000.json`
- Earlier Track A representative: `data/evolution/demian_v2_track_a_island_2_20260511/candidates/gen012_candidate005.json`
- Phenotype: original-vs-gain-zero divergence can persist even when explicit
  release gain is set to zero.
- Scoring pathway: route-disabled and gain-zero ablations are compared against
  original release anchors; a persistent gain-zero divergence suggests the
  release subsystem or gate state can alter downstream dynamics beyond simple
  route injection.
- Characterization grid: seeds 94-102, perturb scales 0.2/0.35/0.7,
  `basis:0` and `gaussian:0`, 128 steps, perturb step 64, with original,
  route-disabled, gain-zero, and per-channel disabled interventions.
- Channel-disabled intervention semantics: each step first advances the model,
  then zeros the requested channel state, records the row, and feeds the
  clamped state into the next step.
- Ablation behavior: gain-zero diagnostics were clean in the Demian v2 held-out
  summary while original-vs-gain-zero divergence persisted, matching
  route-disabled divergence in the top candidates.
- Focused Track B behavior: gain-zero cleanliness remains true, route-disabled
  and gain-zero divergence remain positive, and message/carrier disabled arms
  show the largest channel-necessity drops under the current grid.
- Capsule behavior: full internal-state resume is exact; surface-only and
  channel-only resumes leave positive final and mean-step gaps.
- Replication behavior: three native-emergence Track B replications all
  rediscovered clean gain-zero divergence on held-out seeds 96, 97, and 98.
  Mean held-out route-disabled and gain-zero divergence were both `0.2997`.
- Replication artifact paths:
  - Replication 1 held-out classification:
    `data/diagnostics/gate_state_track_b_replication_1_heldout_20260511/summary.json`
  - Replication 2 held-out classification:
    `data/diagnostics/gate_state_track_b_replication_2_heldout_20260511/summary.json`
  - Replication 3 held-out classification:
    `data/diagnostics/gate_state_track_b_replication_3_heldout_20260511/summary.json`
  - Combined summary:
    `data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json`
- Replication details:
  - Run 1: `gen016_candidate012`; necessary channel order
    `carrier,message,slow,control`; held-out route/gain-zero divergence
    `0.2422`; message-disabled divergence `1.1289`; carrier-disabled
    divergence `1.2604`; slow-disabled divergence `0.8589`; control-disabled
    divergence `0.6411`; full internal-state resume exact; surface-only
    final gap `0.6637`; gain-zero clean.
  - Run 2: `gen019_candidate008`; necessary channel order
    `message,carrier,control,slow`; held-out route/gain-zero divergence
    `0.3194`; message-disabled divergence `1.0852`; carrier-disabled
    divergence `0.8930`; slow-disabled divergence `0.8101`; control-disabled
    divergence `0.8720`; full internal-state resume exact; surface-only
    final gap `0.7789`; gain-zero clean.
  - Run 3: `gen017_candidate000`; necessary channel order
    `carrier,message,slow,control`; held-out route/gain-zero divergence
    `0.3375`; message-disabled divergence `1.0187`; carrier-disabled
    divergence `1.1618`; slow-disabled divergence `0.8553`; control-disabled
    divergence `0.6475`; full internal-state resume exact; surface-only
    final gap `0.3565`; gain-zero clean.
- Held-out divergence ranges across the three replications:
  route/gain-zero `0.2422-0.3375`; message-disabled `1.0187-1.1289`;
  carrier-disabled `0.8930-1.2604`; slow-disabled `0.8101-0.8589`;
  control-disabled `0.6411-0.8720`; surface-only final gap
  `0.3565-0.7789`.
- Necessary-channel pattern: `message` and `carrier` were the top two
  channel-disabled divergences in 3/3 Track B replications; `slow` support was
  also high in all three held-out classifications.
- Cross-seed caveat: this replicated mechanism does not preserve sparse delayed
  duty or timing; the replicated top Track B candidates are high-duty native
  emergence phenotypes.
- Parameter signature: current representatives concentrate around high
  original-vs-ablation divergence with held-out duty near 0.42 and timing 0.0.
- Why it matters: this is a replicated native architectural affordance and
  should be separated from route-specific causal release in future metrics.
- Promotion result: criteria passed in 3/3 Track B replications: mechanism
  appeared in at least two runs, gain-zero remained clean, held-out
  route/gain-zero divergence remained positive, message/carrier necessity
  repeated across top candidates, and capsule evidence remained consistent.

## Explicit Gate-State Prototype

- Status: architectural synthesis prototype
- Implementation: `development/demian_v1_gate_state.py`
- Test surface: `tests/test_demian_v1_gate_state.py`
- Design response: makes gate state a first-class internal channel instead of
  relying on release-vector side effects.
- State owners: `fast`, `slow`, `control`, `message`, `carrier`, `gate`.
- Route semantics: `gate` modulates message-carrier, carrier-slow, and
  message/carrier-surface routes with graded pressure.
- Boundary semantics: `gate` is internal capsule state and is intentionally
  absent from surface-only resume.
- Metric shift: sparsity is read as gate-state change duty and pressure
  variation rather than release-open duty.
- Evidence status: unit-level architectural contracts pass; comparative
  emergence and held-out confirmation are still future evidence, not yet a
  promoted mechanism result.
