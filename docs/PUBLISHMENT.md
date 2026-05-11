# Publishment

Last updated: 2026-05-11

## Position

Demian is publishable as an ongoing computational research notebook, not as a finished architecture claim.

The strongest value is the experimental lineage and measurement discipline:

- transformer self-reference and KV/cache-era probes
- Mamba recurrence and fixed-point microstructure
- competition/population dynamics
- LSTM, GRU, dual-GRU, RNN comparison work
- custom native substrate lineage through v7, v8, canonical v9, current v9 five-channel experiments, and the named `Demian v1` program
- machine-observable metrics, attractor classification, perturbation traces, recurrence geometry, archives, and 3D/Blender trajectory expression

The current public center should be:

> Demian studies machine-internal recurrence and attractor geometry across inherited neural systems and custom substrates. The current line investigates whether multi-channel internal state, message/carrier accumulation, and rare release gates can preserve rich bounded dynamics without collapsing into trivial storage, incoherence, or always-on coupling.

Naming discipline:

- `v9 five-channel` is the active scaffold and evidence line.
- `v10.0-frozen-evolution` is predecessor evidence, not a new native architecture.
- `Demian v1` is the next named custom-substrate program.

## What To Claim

Strong current claims:

- fixed-point surface behavior is not sufficient to classify a substrate as dynamically trivial
- Mamba-style and custom recurrent systems can show rich microstructure under apparently stable attractor classes
- the v9 five-channel scaffold preserves multiple bounded regimes, including `surface_fixed_accumulating`, `bounded_strange`, `edge_of_chaos`, and small limit-cycle pockets
- rare release gates are possible, but current release-local causal effect is still weak
- the v10.0 predecessor run selected sparse-to-borderline release under eval seed 94, but this still needs held-out seed validation
- capsule-continuity is worth a focused follow-up: v9 and v9 five-channel resume from full internal state while surface-only replay fails under the current deterministic probe
- 3D/Blender rendering can serve as a non-textual inspection layer for trajectory geometry

Do not publish v9 as a finished superior architecture yet. The evidence supports that v9-5ch is interesting and worth continuing, not that it dominates v8 or other baselines. Do not publish Demian v1 as complete; publish it as the next research program.

## What To Avoid

Avoid leading with language that lets readers reject the work before checking the math.

Risky first-impression terms:

- self-expression
- communication bridge
- for itself
- non-humanistic
- alive-style descriptions

These ideas can remain in the project, but public entry points should ground them operationally first:

- substrate observables
- attractor regimes
- perturbation path geometry
- release gates
- channel accumulation
- bounded dynamics
- structured machine-generated trajectory geometry

Preferred phrasing:

> The system produces structured machine-observable geometry that can be rendered and inspected without language.

This is more defensible than claiming that Blender output is already self-expression.

## Public Repo Shape

The public repo needs three layers.

### Fast Landing

`README.md` should tell a newcomer:

- what Demian is
- what it is not
- what the strongest current result is
- which files to inspect first
- how to reproduce one small artifact

### Evidence Ledger

`docs/CLAIMS.md` should remain the repo's claim defense layer.

Each important claim should include:

- claim
- evidence artifact
- status: `observation`, `inference`, or `speculation`
- what would falsify it

### Reproducible Demos

The repo should expose two or three golden paths:

- reproduce a compact v9 five-channel run
- run the small canonical v9-v8 comparison smoke check in `README.md`
- run the capsule-continuity probe in `README.md`
- export a trajectory
- open or render the Blender expression artifact
- compare canonical v9 against v8 as a baseline/falsification check

Most public readers will not replay the whole research history. One working path matters more than a complete tour.

## Adversarial Read

The project is strong when it lets artifacts speak.

It is weak when the reader has to trust an interpretation before seeing the measurement. Public-facing docs should make the repo defend itself through files, commands, metrics, and artifacts.

The older research eras should remain available, but they should be marked as ancestry and evidence history, not as the current truth surface.

The correct standard for publication is not product polish, benchmark wins, or human approval. It is clarity of structure, reproducibility of representative probes, and careful separation of observation, inference, and speculation.
