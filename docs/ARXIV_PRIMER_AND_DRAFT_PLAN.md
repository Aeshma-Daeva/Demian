# arXiv Primer And Draft Plan

Last updated: 2026-05-15

Purpose: explain the practical arXiv path for the first Demian paper and turn
the existing publication notes into a paper drafting checklist.

Working title:

> Demian: Discovering Native Mechanisms in Structured Recurrent Substrates

Recommended paper stance:

> A methods-first paper about discovering and validating architecture-internal
> mechanisms in structured recurrent substrates, centered on the replicated
> Gate-State Causal Propagation finding.

This document is a planning and orientation guide. It is not the manuscript
source, not legal advice, and not a substitute for checking arXiv's submission
screens during upload.

## arXiv Basics

arXiv is a public preprint repository with technical checks, moderation, and
versioned records. It is not peer review. Treat the first Demian arXiv paper as
a frozen, citable scientific statement, while GitHub remains the living lab and
artifact ledger.

Important operational facts:

- A first-time submitter, or a submitter entering a new subject area, may need
  endorsement before submitting. arXiv recommends using an institutional email
  when available; otherwise, a personal endorsement from an established arXiv
  author in the subject area may be needed.
- The submitter must ensure all authors consent and that author identity and
  affiliation metadata are accurate.
- arXiv metadata fields are stricter than the PDF. Title, author, and abstract
  metadata should be ASCII-safe. Avoid copied Unicode punctuation, opaque TeX
  macros, and formatting commands in metadata.
- The metadata abstract must be short. arXiv currently rejects abstracts longer
  than 1920 characters.
- arXiv assigns the final identifier only when the work is announced. It cannot
  be reserved or backdated.
- Quality checks can take one to four days, sometimes longer. Announcements
  normally happen Sunday through Thursday, with no Friday or Saturday
  announcements.
- A posted paper is part of the permanent scholarly record. New versions can be
  submitted, but the original version remains available.
- The license choice is irrevocable for that version. Check journal, funder,
  and collaborator constraints before choosing.

Official arXiv pages to check before submission:

- TeX/LaTeX submissions:
  <https://info.arxiv.org/help/submit_tex.html>
- PDF submissions:
  <https://info.arxiv.org/help/submit_pdf.html>
- Metadata:
  <https://info.arxiv.org/help/prep.html>
- Endorsement:
  <https://info.arxiv.org/help/endorsement.html>
- Availability and announcement schedule:
  <https://info.arxiv.org/help/availability.html>
- Licenses:
  <https://info.arxiv.org/help/license/index.html>
- Category taxonomy:
  <https://arxiv.org/category_taxonomy>

## Formatting And Source Packaging

Recommended format: LaTeX source upload.

arXiv generally expects TeX/LaTeX source when the paper was written in TeX.
PDF-only submission is mainly for papers that were not prepared in TeX, and a
PDF generated from TeX may be rejected unless an exception is granted.

Use a simple source tree:

```text
paper/
  main.tex
  references.bib
  figures/
    methodology_loop.pdf
    five_channel_scaffold.pdf
    gate_state_diagnostic.pdf
    capsule_continuity.pdf
```

Recommended LaTeX style:

- Use a standard article-style format unless a later target venue requires a
  specific template.
- Keep the arXiv version single-spaced, not referee/double-spaced.
- Use `graphicx` and `\includegraphics` for figures.
- Use `natbib` or standard BibTeX unless there is a strong reason for
  `biblatex`.
- Include either the `.bib` file or a matching generated `.bbl` file. For
  portability, include the `.bbl` in the final arXiv bundle after local build
  validation.
- Use a fixed date or omit the date; do not rely on `\today`.
- Keep custom macros small, local, and readable. Expand opaque macros in title
  and abstract metadata.
- Prefer PDF figures for diagrams and PNG/JPG for raster plots. Do not rely on
  arXiv to convert figure formats.

Do not include:

- `.aux`, `.log`, `.out`, `.toc`, `.synctex`, local build products, local PDFs
  produced from the same TeX source, editor backup files, hidden files, or
  unused figures;
- journal referee letters, submission notes, unrelated scripts, or unused
  templates;
- embedded JavaScript, animated GIFs, movies, or HTML inside the PDF.

Before uploading:

1. Build locally from a clean directory.
2. Verify all citations resolve.
3. Verify figures are readable in grayscale and at one-column/two-column print
   size.
4. Verify every figure used in the manuscript is in the source bundle.
5. Verify no source comments reveal private notes, credentials, or unrelated
   drafts.
6. Upload and inspect the arXiv-rendered PDF before final submission.

## Category And Metadata Recommendation

Recommended primary category: `cs.LG`.

Reason: the paper is about machine-learning methodology, recurrent substrate
analysis, explanation, and evidence discipline. The category taxonomy describes
`cs.LG` as covering machine-learning research, including explanation and
methodology.

Recommended cross-list: `cs.NE`.

Reason: Demian uses neural architecture work and evolutionary search. The
taxonomy describes `cs.NE` as covering neural networks, connectionism, genetic
algorithms, artificial life, and adaptive behavior.

Optional cross-list: `cs.AI`.

Use only if the final introduction emphasizes the broader AI-methodology frame.
The taxonomy routes most machine learning work to `cs.LG`, so `cs.AI` should
not be the primary category unless the paper is framed primarily as general AI
methodology rather than ML mechanism discovery.

Draft metadata:

```text
Title:
Demian: Discovering Native Mechanisms in Structured Recurrent Substrates

Comments:
Methods-first preprint; includes reproducibility artifacts and code repository.

Primary category:
cs.LG

Cross-list:
cs.NE
```

The final metadata abstract should be shorter than the manuscript abstract if
needed. Keep it plain, ASCII-safe, and claim-limited.

## Demian Paper Strategy

The paper should start from the existing split in
[PUBLISHMENT.md](PUBLISHMENT.md):

- GitHub is the canonical living lab.
- arXiv is the frozen scientific statement.
- The first arXiv paper is not a complete project autobiography.
- The paper centers one strong replicated finding: Gate-State Causal
  Propagation.
- Demian v1 is a design consequence, not a validated result.

### Claim Boundary

Strong enough for the paper:

- Demian is a methodology for discovering mechanisms in structured recurrent
  substrates.
- Surface behavior alone can be misleading; fixed readouts can hide internal
  channel dynamics.
- Causal ablations and capsule-continuity probes provide a useful evidence
  discipline.
- In the current Track B protocol, Gate-State Causal Propagation replicated in
  3/3 native-emergence runs.
- Message and carrier channels were repeatedly necessary under channel-disabled
  diagnostics.
- Full internal-state resume was exact in the current deterministic probes,
  while surface-only resume left gaps.

Do not claim:

- benchmark superiority;
- general intelligence;
- consciousness;
- a finished architecture;
- formal chaos for `bounded_strange`;
- stable sparse delayed release across seeds;
- Demian v1 validation.

### Section Map

1. Introduction
   - Problem: optimizing exposed behavior can miss the internal mechanism.
   - Response: treat recurrent substrate design as mechanism discovery.
   - Contribution: methodology plus a replicated gate-state mechanism.

2. Background and Vocabulary
   - Translate Demian terms using [GLOSSARY.md](GLOSSARY.md).
   - Position the paper near RNN dynamical reverse engineering, mechanistic
     interpretability, and evolutionary architecture search.

3. Substrate Under Study
   - Describe the v9 five-channel scaffold:
     `fast`, `slow`, `control`, `message`, and `carrier`.
   - Clarify that this is the study substrate, not the final Demian v1
     architecture.

4. Methodology
   - Dual-track search: engineered targets vs native-emergence search.
   - Multi-objective evolutionary search.
   - Causal ablations: routes-disabled, gain-zero, channel-disabled.
   - Capsule continuity: uninterrupted continuation, full internal-state
     resume, surface-only replay.
   - Evidence gates: observation, inference, speculation.

5. Results
   - Gate-State Causal Propagation.
   - 3/3 Track B replication summary.
   - Message/carrier necessity.
   - Gain-zero cleanliness and positive route/gain-zero divergence.
   - Full internal-state resume vs surface-only gap.

6. Negative Results
   - Sparse release did not generalize.
   - v10 predecessor was not backend/held-out stable.
   - Dynamic selection preserved some causal checks but failed sparse/timed
     criteria.
   - `bounded_strange` is an inspection label, not a formal chaos claim.

7. Synthesis: Demian v1
   - Make gate state explicit.
   - Preserve message/carrier.
   - Treat surface as output and internal state as computation.
   - State clearly that this is a hypothesis and design response.

8. Limitations
   - Small architecture family.
   - Custom vocabulary.
   - CPU-scale and archived search limitations.
   - Replication is within the current Track B protocol.

9. Reproducibility and Artifacts
   - Point to GitHub, [REPRODUCIBILITY.md](REPRODUCIBILITY.md), and
     [data/INDEX.md](../data/INDEX.md).
   - Include compact commands only; do not require readers to rerun full
     evolutionary searches.

### Figure And Table Plan

- Figure 1: Methodology loop: substrate, perturbation, ablation, capsule
  resume, evidence gate.
- Figure 2: v9 five-channel scaffold.
- Figure 3: Gate-State Causal Propagation diagnostic schematic.
- Figure 4: Capsule continuity: full internal-state resume vs surface-only
  replay.
- Table 1: Vocabulary translation from Demian terms to standard terminology.
- Table 2: Track B 3/3 replication summary.
- Table 3: Negative results and resulting design choices.

### Evidence Map

Use these repo files as the claim backbone:

- [ARXIV_OUTLINE.md](ARXIV_OUTLINE.md): paper structure and abstract draft.
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md): broader report to condense.
- [CLAIMS.md](CLAIMS.md): promoted claims and falsification conditions.
- [NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md): Gate-State Causal Propagation
  details.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): compact CPU checks and artifact
  inspection.
- [GLOSSARY.md](GLOSSARY.md): terminology translation.
- [data/INDEX.md](../data/INDEX.md): artifact map.

Primary result artifacts:

- `data/diagnostics/gate_state_track_b_replication_summary_20260511/summary.json`
- `data/diagnostics/gate_state_propagation_characterization_20260511/summary.json`
- `data/substrate_lab/v9_capsule_continuity_20260511/summary.json`
- `data/evolution/v9_5ch_release_20260509_full/archive.json`

## Real Paper Examples

Use these as writing and citation models, not as templates to copy.

### RNN Mechanism Discovery

Maheswaranathan et al., "Reverse engineering recurrent networks for sentiment
classification reveals line attractor dynamics", `arXiv:1906.10720`.

- Why it matters: it is a clean example of using dynamical-systems tools to
  reverse-engineer trained RNNs.
- Useful pattern: state the task, identify fixed points, linearize dynamics,
  then present the interpretable mechanism.
- Demian adaptation: replace task accuracy as the center with substrate
  perturbation, ablation, and capsule continuity.
- Link: <https://arxiv.org/abs/1906.10720>

Torre et al., "Mechanistic Interpretability of RNNs emulating Hidden Markov
Models", `arXiv:2510.25674`.

- Why it matters: recent RNN mechanistic-interpretability example with a clear
  mechanistic result and replication across related setups.
- Useful pattern: describe the apparent mismatch between continuous RNN state
  and discrete/stochastic behavior, then show the mechanism that resolves it.
- Demian adaptation: use the same "apparent surface vs internal mechanism"
  rhetorical shape.
- Link: <https://arxiv.org/abs/2510.25674>

Huang et al., "Measuring and Controlling Solution Degeneracy across
Task-Trained Recurrent Neural Networks", `arXiv:2410.03972`.

- Why it matters: strong example of comparing internal neural dynamics across
  many RNN solutions with similar behavior.
- Useful pattern: distinguish behavior, dynamics, and weights rather than
  collapsing all evidence into one score.
- Demian adaptation: use it to justify separating surface behavior from
  internal-state evidence.
- Link: <https://arxiv.org/abs/2410.03972>

### Dynamical-Systems Framing

Hess et al., "Generalized Teacher Forcing for Learning Chaotic Dynamics",
`arXiv:2306.04406`.

- Why it matters: an example of connecting RNN training, attractor
  reconstruction, and scientific interpretability.
- Useful pattern: be precise about when a model reconstructs dynamics and what
  is being measured.
- Demian adaptation: cite cautiously around attractor/regime vocabulary and
  avoid claiming formal chaos unless measured.
- Link: <https://arxiv.org/abs/2306.04406>

"Dynamical similarity analysis can identify compositional dynamics developing
in RNNs", `arXiv:2410.24070`.

- Why it matters: useful for framing temporal/dynamical comparison as distinct
  from static representation comparison.
- Useful pattern: explain why geometry-only measures can miss computation
  unfolding over time.
- Demian adaptation: supports the paper's preference for trajectories,
  resumes, and perturbation paths over final surface snapshots.
- Link: <https://arxiv.org/abs/2410.24070>

### Mechanistic Interpretability Around Transformers

Fernando and Guitchounts, "Transformer Dynamics: A neuroscientific approach to
interpretability of large language models", `arXiv:2502.12131`.

- Why it matters: shows the broader trend of treating model internals as
  dynamical systems.
- Useful pattern: connect mechanistic interpretability with trajectories and
  attractor-like behavior.
- Demian adaptation: cite as adjacent framing, not as core RNN evidence.
- Link: <https://arxiv.org/abs/2502.12131>

## Reference Seed List

Start the bibliography from these groups.

Core nearby examples:

- `arXiv:1906.10720`: RNN reverse engineering and line attractors.
- `arXiv:2510.25674`: mechanistic interpretability of RNNs emulating HMMs.
- `arXiv:2410.03972`: solution degeneracy across task-trained RNNs.
- `arXiv:2306.04406`: RNNs and chaotic dynamical-system reconstruction.
- `arXiv:2410.24070`: dynamical similarity analysis in RNNs.
- `arXiv:2502.12131`: transformer residual stream as a dynamical system.

Likely additional literature groups to fill during manuscript drafting:

- mechanistic interpretability and causal intervention methods;
- fixed-point and linearization analysis of recurrent networks;
- low-rank and structured recurrent dynamics;
- evolutionary search and multi-objective optimization;
- reproducibility and artifact-backed ML research practices.

## Drafting Checklist

Before writing LaTeX:

- Convert the abstract in [ARXIV_OUTLINE.md](ARXIV_OUTLINE.md) into a
  metadata-safe version under 1920 characters.
- Build a one-page "claim table" from [CLAIMS.md](CLAIMS.md) with status,
  evidence, and falsifier for each paper claim.
- Select the exact figure assets or scripts for the four planned figures.
- Decide which plots are manuscript figures and which remain repo-only
  inspection artifacts.
- Produce a first bibliography file with the seed references above.
- Write Section 2 with standard terminology first, then introduce Demian terms.
- Keep negative results in the main paper, not only in an appendix.
- Add an explicit limitations paragraph to every major result section.

Before submission:

- Run the focused reproducibility checks from
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- Confirm artifact paths exist and match the values reported in the paper.
- Confirm all figure captions state what is measured, not only what is shown.
- Confirm all claims in the abstract are supported in the result section.
- Confirm Demian v1 is written as a proposed design response, not as a result.
- Build the LaTeX source from a clean checkout or clean export directory.
- Inspect the arXiv-generated PDF before completing submission.

## Submission Checklist

Account and metadata:

- arXiv account ready.
- Endorsement ready for the chosen category, if required.
- Author names, affiliations, and consent confirmed.
- Title ASCII-safe.
- Metadata abstract ASCII-safe and under 1920 characters.
- Comments field includes page/figure count and repository link if desired.
- Primary category `cs.LG` selected; `cs.NE` cross-list considered.
- License chosen after checking collaborator, funder, and journal constraints.

Source bundle:

- `main.tex` builds locally.
- `references.bib` and/or matching `main.bbl` included.
- Only used figures included.
- No build logs, local PDFs, hidden files, private notes, or unrelated code.
- No embedded JavaScript or animated media inside the PDF.
- Source comments reviewed for privacy and relevance.

Scientific readiness:

- Abstract does not overclaim.
- Every result has an artifact or reproducibility link.
- Negative results and limitations are visible.
- The paper explains custom vocabulary in standard terms.
- The repo link points to reproducibility instructions and artifact index.
- The final PDF is checked after arXiv processing, not only locally.
