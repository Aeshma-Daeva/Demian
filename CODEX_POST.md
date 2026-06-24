# Codex end-of-session posting instructions

At the end of every coding session, do the following automatically:

## 1. Create a post file

Create a new file at:
```
src/content/posts/YYYY-MM-DD-<short-slug>.md
```

Use today's date and a slug derived from the main thing that changed.

## 2. Frontmatter format

```markdown
---
title: "<what changed, one sentence>"
date: YYYY-MM-DD
description: "<one sentence summary for the index page>"
status: public
tags: ["relevant", "tags", "here"]
artifacts:
  - label: "Report"
    path: "adaptation_probes/.../report.md"
metrics:
  - label: "recordings"
    value: 250
---
```

Good tags to reuse: `eeg`, `chaos`, `lyapunov`, `v3`, `probe`, `cli`, `tests`, `architecture`, `fix`, `refactor`

## 3. Body structure (use what applies)

```markdown
## What changed

- bullet list of files/functions changed
- keep it concrete

## Results (if there was a real run)

| metric | value |
|---|---|
| ... | ... |

## Interpretation

1-3 sentences on what the results mean.
What can be claimed. What cannot.

## Verification

```
N passed in Xs
```
```

## 4. Commit and push

```bash
git add src/content/posts/ src/data/blackboard.json
git commit -m "devlog: <same as title>"
git push origin main
```

The site rebuilds automatically via GitHub Actions.

## Rules

- Never claim brain simulation, decoding, consciousness, or biomarkers unless explicitly told to
- Keep interpretation narrow and honest
- If there were no real runs this session, skip the Results section
- Keep the whole post under ~300 words unless the session was unusually complex

## Optional draft helper

```bash
npm run draft:post -- --source adaptation_probes/.../report.md --tags eeg,v3,probe
```

Review the draft, replace placeholder text, then change `status: draft` to `status: public`.
