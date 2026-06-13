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
The Makefile also regenerates the paper figures from the compact evidence table.

## Source Bundle

For arXiv, upload the source files that are needed to rebuild the paper:

- `main.tex`
- `references.bib`
- final files under `figures/`
- `evidence_summary.csv`
- `evidence_summary.json`

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
- author: `Azael`
- affiliation: `Independent Researcher`
- reproducibility tag: `v0.2-fixed-point-paper`

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

## Release Checklist

- Build from a clean tree with `make clean && make`.
- Confirm the LaTeX log has no unresolved citations, undefined references, or
  overfull boxes.
- Commit the final source and figure files.
- Tag the final commit as `v0.2-fixed-point-paper`.
- Push the branch and tag before using the GitHub URL in arXiv metadata.
