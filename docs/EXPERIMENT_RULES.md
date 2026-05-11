# Experiment Rules

Purpose:
- reduce interpretation drift
- keep live docs tied to artifacts
- stop best-run storytelling from becoming working truth

## Claim Discipline

Every nontrivial live claim should have:
- a label: `observation`, `inference`, or `speculation`
- at least one artifact path or script path
- one falsification line

If a claim does not meet that bar, it stays in notes or discussion, not in live truth.

## Comparison Discipline

When comparing two mechanisms, regimes, or substrates:
- keep seeds, step counts, and key hyperparameters matched when possible
- explicitly state the mismatch when not possible
- do not narrate over setup differences as if they were mechanism differences

## Best-Run Discipline

Always separate:
- `best_case`
- `typical_case`
- `stability_across_seeds`

Do not let a best run become the default interpretation without a replication check.

## Promotion Standard

Promote a finding into [WORKING_STATE.md](/home/xenith/demian/docs/WORKING_STATE.md) only when one of these is true:
- it survived a direct comparative check
- it survived a seed sweep or rerun check
- it is a narrow operational fact such as which regime is currently being used

## Interpretation Standard

Preferred order:
1. artifact
2. summary statistic
3. interpretation
4. mechanism hypothesis

Do not reverse that order.

## Naming Standard

Named effects should be conservative.

Before naming an effect, ask:
- is it stable across more than one run or seed?
- is it distinguishable from generic perturbation sensitivity?
- does the name describe the evidence, or does it smuggle in theory?

If unclear, use a descriptive temporary name instead of a mechanistic one.

## Notebook Boundary

Put these in archive notes, not live docs:
- broad theory
- philosophical framing
- speculative mechanism stories
- unresolved interpretations

Put these in live docs:
- current regime choice
- active open questions
- claims with explicit evidence links
- reproducible commands and tests

## Doc Ownership

Use each live document for one job:

- [WORKING_STATE.md](/home/xenith/demian/docs/WORKING_STATE.md): stable restart orientation and current priorities.
- [SUBSTRATE_ANATOMY.md](/home/xenith/demian/docs/SUBSTRATE_ANATOMY.md): channel roles, routing anatomy, metric meanings, and lineage deltas.
- [LABBOOK.md](/home/xenith/demian/docs/LABBOOK.md): append-only experiment chronology and fast-moving run details.
- [CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md): promoted claims with evidence and falsifiers.
- [data/INDEX.md](/home/xenith/demian/data/INDEX.md): human-readable artifact map.
- [data/MANIFEST.json](/home/xenith/demian/data/MANIFEST.json): machine-readable run and artifact manifest.
- [REPO_INVENTORY.md](/home/xenith/demian/docs/REPO_INVENTORY.md): file responsibility map before moving code or docs.
- [DEVELOPMENT_SCRIPT_MAP.md](/home/xenith/demian/docs/DEVELOPMENT_SCRIPT_MAP.md): classification of `development/*.py` before moving research scripts.
- [SUBSTRATE_LAB_DEPENDENCIES.md](/home/xenith/demian/docs/SUBSTRATE_LAB_DEPENDENCIES.md): dependency map before extracting legacy lab helpers.

Avoid copying the same latest-run interpretation into multiple docs. Put the
run detail in the labbook, promote the durable part into claims, and link to
the artifact from the index/manifest.

## Minimal Review Before Writing Conclusions

Before adding or revising a conclusion:
- check the artifact again
- check whether the result is best-case or replicated
- check whether a setup mismatch is driving the difference
- add or update the relevant item in [CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md)
