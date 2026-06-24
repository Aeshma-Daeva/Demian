# Demian Lab - research site

Research log for the Demian Substrate EEG adaptation project.

Built with [Astro](https://astro.build), deployed to GitHub Pages automatically on push.

## Local setup

```bash
npm install
npm run dev
```

## GitHub Pages

Enable Pages in GitHub repo settings with Source set to **GitHub Actions**.
The workflow builds this Astro app from the repository root and publishes `dist/`.

```bash
git add .
git commit -m "add research site"
git push origin main
```

GitHub Actions builds and deploys automatically. Site goes live at:
`https://aeshma-daeva.github.io/Demian-Lab`

## Public blackboard and feeds

- `/blackboard` shows the current public status, objectives, artifacts, and claim boundaries.
- `/updates.json` is the stable Discord bot feed.
- `/rss.xml` is the standard RSS feed.

Discord bots should poll `/updates.json`, use `tags` for channel routing, and post the `title`, `description`, `url`, `metrics`, and `artifacts` fields.

## Adding posts

Drop a `.md` file in `src/content/posts/` with this frontmatter:

```markdown
---
title: "Your title"
date: 2026-06-23
description: "One sentence for the index"
status: public
tags: ["eeg", "v3"]
artifacts:
  - label: "Report"
    path: "adaptation_probes/.../report.md"
metrics:
  - label: "recordings"
    value: 250
---

Your content here.
```

Push and the site updates.

## Drafting from EEG artifacts

Generate an editable draft from a local artifact:

```bash
npm run draft:post -- --source adaptation_probes/eeg_demian_architecture_v3_all3264/lyapunov_style_probe_report.md --tags eeg,v3,probe
```

By default the script reads artifacts from `/home/xenith/demian_eeg`. Override with:

```bash
DEMIAN_EEG_ROOT=/path/to/demian_eeg npm run draft:post -- --source adaptation_probes/.../report.md
```

Drafts are created with `status: draft`; change to `status: public` after review.
