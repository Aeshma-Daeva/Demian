# Fixed-Point Internal Structure Paper

This directory contains the first arXiv-oriented draft for the Demian fixed-point
surface result.

Working title:

`Fixed Points Are Not Empty: Hidden Internal Structure Behind Apparently Static Recurrent Surfaces`

## Build

```bash
cd papers/fixed_point_internal_structure
make
```

The build produces `main.pdf`. Build products are not part of the source bundle.

## Source Bundle

For arXiv, upload the source files that are needed to rebuild the paper:

- `main.tex`
- `references.bib`
- any final figure files, if figures are added later
- optional evidence summaries if they are kept as supplemental source files:
  `evidence_summary.csv` and `evidence_summary.json`

Do not upload temporary build outputs such as `.aux`, `.log`, `.out`, `.toc`,
`.synctex.gz`, or local cache directories.

## arXiv Notes Checked

Official arXiv guidance used while setting up this draft:

- TeX submissions: https://info.arxiv.org/help/submit_tex.html
- PDF submissions: https://info.arxiv.org/help/submit_pdf.html
- submission preparation and metadata: https://info.arxiv.org/help/prep.html
- category taxonomy: https://arxiv.org/category_taxonomy

Likely metadata:

- primary category: `cs.LG`
- possible cross-list: `cs.NE`

The abstract in arXiv metadata must stay under arXiv's abstract limit. Keep the
metadata abstract close to the paper abstract, but remove line breaks and avoid
LaTeX commands where possible.

## Codex Workflow Notes

Codex is useful here as a paper-building assistant, not as a substitute for the
scientific claim. The practical workflow is:

- extract and compact evidence from local artifacts;
- build the LaTeX source and inspect warnings;
- audit claim wording against the evidence tables;
- keep negative controls and failed mechanism names visible;
- review diffs before committing.

The author still owns the claim, scope, authorship, and submission metadata.
