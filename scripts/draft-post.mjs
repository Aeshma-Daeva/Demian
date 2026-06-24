#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const ROOT = process.cwd();
const EEG_ROOT = process.env.DEMIAN_EEG_ROOT ?? '/home/xenith/demian_eeg';
const POSTS_DIR = path.join(ROOT, 'src/content/posts');
const CLAIM_BOUNDARY = 'Not claiming: brain simulation, decoding, consciousness, biomarkers, or medical use.';

const args = new Map();
for (let index = 2; index < process.argv.length; index += 1) {
  const arg = process.argv[index];
  if (!arg.startsWith('--')) continue;
  const key = arg.slice(2);
  const next = process.argv[index + 1];
  if (next && !next.startsWith('--')) {
    args.set(key, next);
    index += 1;
  } else {
    args.set(key, 'true');
  }
}

const source = args.get('source');
if (!source) {
  console.error('Usage: npm run draft:post -- --source adaptation_probes/.../report.md [--title "..."] [--tags "eeg,v3,probe"]');
  process.exit(1);
}

const sourcePath = path.isAbsolute(source) ? source : path.join(EEG_ROOT, source);
if (!fs.existsSync(sourcePath)) {
  console.error(`Source artifact not found: ${sourcePath}`);
  process.exit(1);
}

const raw = fs.readFileSync(sourcePath, 'utf8');
const relativeSource = path.relative(EEG_ROOT, sourcePath);
const today = new Date().toISOString().slice(0, 10);
const fallbackTitle = titleFromMarkdown(raw) ?? titleFromPath(sourcePath);
const title = args.get('title') ?? fallbackTitle;
const tags = (args.get('tags') ?? 'eeg,progress,draft')
  .split(',')
  .map(tag => tag.trim())
  .filter(Boolean);
const slug = `${today}-${slugify(title)}`;
const outPath = path.join(POSTS_DIR, `${slug}.md`);

if (fs.existsSync(outPath) && !args.has('force')) {
  console.error(`Draft already exists: ${outPath}`);
  console.error('Pass --force to overwrite it.');
  process.exit(1);
}

const metrics = extractMetrics(raw);
const excerpt = firstUsefulParagraph(raw);
const body = `---
title: "${escapeYaml(title)}"
date: ${today}
description: "${escapeYaml(excerpt)}"
status: draft
tags: [${tags.map(tag => `"${escapeYaml(tag)}"`).join(', ')}]
artifacts:
  - label: "Source artifact"
    path: "${escapeYaml(relativeSource)}"
${metrics.length ? `metrics:
${metrics.map(metric => `  - label: "${escapeYaml(metric.label)}"\n    value: ${formatMetricValue(metric.value)}`).join('\n')}
` : ''}---

## What changed

- Draft generated from \`${relativeSource}\`.
- Replace this section with the concrete implementation or experiment changes.

## Results

${metrics.length ? `| metric | value |
|---|---|
${metrics.map(metric => `| ${metric.label} | ${metric.value} |`).join('\n')}` : 'Add run metrics here if this update includes a real run.'}

## Interpretation

${excerpt}

${CLAIM_BOUNDARY}

## Verification

Add tests, build output, or run command evidence before publishing.
`;

fs.mkdirSync(POSTS_DIR, { recursive: true });
fs.writeFileSync(outPath, body);
console.log(`Created draft: ${path.relative(ROOT, outPath)}`);

function titleFromMarkdown(value) {
  const heading = value.match(/^#\s+(.+)$/m);
  return heading?.[1]?.trim();
}

function titleFromPath(value) {
  return path.basename(value, path.extname(value)).replaceAll('_', ' ');
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 60);
}

function firstUsefulParagraph(value) {
  const stripped = value
    .split('\n')
    .filter(line => !line.startsWith('#') && !line.startsWith('|') && !line.startsWith('```'))
    .join('\n');
  const paragraph = stripped
    .split(/\n\s*\n/)
    .map(part => part.trim())
    .find(part => part.length > 40 && !part.startsWith('-'));
  return (paragraph ?? 'Draft progress update generated from a Demian EEG artifact.')
    .replace(/\s+/g, ' ')
    .slice(0, 180);
}

function extractMetrics(value) {
  const metrics = [];
  const seen = new Set();
  const tableRow = /^\|\s*([A-Za-z0-9_. -]+)\s*\|\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*\|/gim;
  let match;
  while ((match = tableRow.exec(value)) && metrics.length < 8) {
    const label = match[1].trim();
    if (label.toLowerCase() === 'metric' || seen.has(label)) continue;
    seen.add(label);
    metrics.push({ label, value: match[2] });
  }
  return metrics;
}

function escapeYaml(value) {
  return String(value).replaceAll('\\', '\\\\').replaceAll('"', '\\"');
}

function formatMetricValue(value) {
  return /^[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?$/i.test(value) ? value : `"${escapeYaml(value)}"`;
}
