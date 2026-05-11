"""Update generated docs and run the local verification suite.

This script keeps the live docs tied to machine-checkable state:

- discovered substrate registry
- active substrate named in WORKING_STATE
- local test command status
- markdown evidence links that resolve to files in this repo

It deliberately does not invent conclusions. Claims still need human/research
judgment; this tool updates the derived ledger around them so stale evidence and
test status are visible without manual bookkeeping.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / "venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

DOCS_TO_VALIDATE = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "WORKING_STATE.md",
    REPO_ROOT / "docs" / "CURRENT_STATE_AND_ROUTING.md",
    REPO_ROOT / "docs" / "SUBSTRATE_ANATOMY.md",
    REPO_ROOT / "docs" / "EXPERIMENT_NAMING.md",
    REPO_ROOT / "docs" / "CAPSULE_CONTINUITY.md",
    REPO_ROOT / "docs" / "LABBOOK.md",
    REPO_ROOT / "docs" / "RESEARCH_MAP.md",
    REPO_ROOT / "docs" / "CLAIMS.md",
    REPO_ROOT / "docs" / "EXPERIMENT_RULES.md",
    REPO_ROOT / "docs" / "REPO_CLEANUP_PLAN.md",
    REPO_ROOT / "docs" / "REPO_INVENTORY.md",
    REPO_ROOT / "docs" / "DEVELOPMENT_SCRIPT_MAP.md",
    REPO_ROOT / "docs" / "SUBSTRATE_LAB_DEPENDENCIES.md",
    REPO_ROOT / "data" / "INDEX.md",
]

MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"

AUTO_START = "<!-- AUTO-GENERATED:doc-state:start -->"
AUTO_END = "<!-- AUTO-GENERATED:doc-state:end -->"


@dataclass
class CommandResult:
    command: str
    returncode: int
    duration_s: float
    output_tail: str


@dataclass
class LinkIssue:
    source: Path
    target: str
    line: int


@dataclass
class ManifestIssue:
    message: str


def run_command(args: list[str], timeout_s: int) -> CommandResult:
    started = dt.datetime.now(dt.timezone.utc)
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    duration = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    output = proc.stdout or ""
    tail = "\n".join(output.strip().splitlines()[-18:])
    return CommandResult(" ".join(args), proc.returncode, duration, tail)


def command_output(args: list[str]) -> str:
    proc = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return (proc.stdout or "").strip()


def discover_test_commands() -> list[list[str]]:
    test_dir = REPO_ROOT / "tests"
    commands: list[list[str]] = []
    run_tests = test_dir / "run_tests.py"
    if run_tests.exists():
        commands.append([str(PYTHON), str(run_tests.relative_to(REPO_ROOT))])
    for path in sorted(test_dir.glob("test_*.py")):
        commands.append([str(PYTHON), str(path.relative_to(REPO_ROOT))])
    return commands


def parse_substrate_registry() -> list[str]:
    src = (REPO_ROOT / "development" / "substrate_lab.py").read_text()
    match = re.search(r"SUBSTRATE_REGISTRY\s*=\s*\{(?P<body>.*?)\n\}", src, re.S)
    if not match:
        return []
    return re.findall(r'"([^"]+)"\s*:', match.group("body"))


def native_sort_key(name: str) -> tuple[int, ...]:
    suffix = name.removeprefix("demian_native_v")
    alias = re.fullmatch(r"(\d)(\d+)([a-z]*)", suffix)
    if alias:
        key = [int(alias.group(1)), int(alias.group(2))]
        key.extend(ord(ch) - 96 for ch in alias.group(3))
        return tuple(key)
    parts = re.split(r"[._-]", suffix)
    nums: list[int] = []
    for part in parts:
        number = re.match(r"(\d+)", part)
        if number:
            nums.append(int(number.group(1)))
        tail = re.search(r"([a-z]+)$", part)
        if tail:
            nums.append(ord(tail.group(1)[0]) - 96)
    return tuple(nums)


def latest_native_substrate(registry: Iterable[str]) -> str:
    native = [name for name in registry if re.fullmatch(r"demian_native_v[0-9][0-9.a-z]*", name)]
    if not native:
        return "unknown"
    native_set = set(native)
    canonical: list[str] = []
    for name in native:
        suffix = name.removeprefix("demian_native_v")
        compact = re.fullmatch(r"([0-9])([0-9]+)([a-z]*)", suffix)
        if compact:
            if int(compact.group(2)) == 0 and not compact.group(3):
                base = f"demian_native_v{compact.group(1)}"
                if base in native_set:
                    continue
            dotted = f"demian_native_v{compact.group(1)}.{compact.group(2)}{compact.group(3)}"
            if dotted in native_set:
                continue
        canonical.append(name)
    return sorted(canonical or native, key=native_sort_key)[-1]


def active_substrates_from_working_state() -> list[str]:
    path = REPO_ROOT / "docs" / "WORKING_STATE.md"
    text = path.read_text() if path.exists() else ""
    manual_text = re.sub(
        r"<!-- AUTO-GENERATED:doc-state:start -->.*?<!-- AUTO-GENERATED:doc-state:end -->",
        "",
        text,
        flags=re.S,
    )
    if re.search(r"Active experimental scaffold:.*?v9 five-channel", manual_text, re.S | re.I):
        return ["demian_native_v9"]
    if re.search(r"Demian v1:.*?v9 five-channel", manual_text, re.S | re.I):
        return ["demian_native_v9"]
    scaffold = re.search(r"Active scaffold:\s*\n(?P<body>(?:- `demian_native_[^`]+`\s*\n?)+)", text)
    if scaffold:
        return re.findall(r"`(demian_native_[^`]+)`", scaffold.group("body"))
    seen: list[str] = []
    for name in re.findall(r"`(demian_native_[^`]+)`", manual_text):
        if name not in seen:
            seen.append(name)
    return seen


def local_markdown_links(path: Path) -> Iterable[tuple[str, int]]:
    if not path.exists():
        return []
    links: list[tuple[str, int]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", line):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            links.append((target, line_no))
    return links


def resolve_markdown_target(source: Path, target: str) -> Path:
    target = target.strip("<>")
    target = target.split("#", 1)[0]
    target = re.sub(r":\d+$", "", target)
    if target.startswith("/"):
        return Path(target)
    return (source.parent / target).resolve()


def validate_links() -> list[LinkIssue]:
    issues: list[LinkIssue] = []
    for source in DOCS_TO_VALIDATE:
        for target, line_no in local_markdown_links(source):
            resolved = resolve_markdown_target(source, target)
            if not resolved.exists():
                issues.append(LinkIssue(source.relative_to(REPO_ROOT), target, line_no))
    return issues


def repo_relative_path(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def manifest_artifact_paths() -> set[str]:
    if not MANIFEST_PATH.exists():
        return set()
    payload = json.loads(MANIFEST_PATH.read_text())
    paths: set[str] = set()
    for run in payload.get("runs", []):
        primary = run.get("primary_summary")
        if primary:
            paths.add(str(primary))
        for artifact in run.get("artifacts", []):
            paths.add(str(artifact))
    return paths


def index_artifact_paths() -> set[str]:
    index = REPO_ROOT / "data" / "INDEX.md"
    paths: set[str] = set()
    for target, _line_no in local_markdown_links(index):
        resolved = resolve_markdown_target(index, target)
        rel = repo_relative_path(resolved)
        if rel and rel.startswith("data/"):
            paths.add(rel)
    return paths


def claim_data_links() -> set[str]:
    claims = REPO_ROOT / "docs" / "CLAIMS.md"
    paths: set[str] = set()
    for target, _line_no in local_markdown_links(claims):
        resolved = resolve_markdown_target(claims, target)
        rel = repo_relative_path(resolved)
        if rel and rel.startswith("data/"):
            paths.add(rel)
    return paths


def validate_manifest() -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []
    if not MANIFEST_PATH.exists():
        return [ManifestIssue("data/MANIFEST.json is missing")]
    try:
        payload = json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError as exc:
        return [ManifestIssue(f"data/MANIFEST.json is invalid JSON: {exc}")]

    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        issues.append(ManifestIssue("data/MANIFEST.json must contain a non-empty runs list"))
        return issues

    seen_ids: set[str] = set()
    artifact_paths: set[str] = set()
    for index, run in enumerate(runs):
        run_id = run.get("id")
        if not run_id:
            issues.append(ManifestIssue(f"run at index {index} is missing id"))
        elif run_id in seen_ids:
            issues.append(ManifestIssue(f"duplicate manifest run id: {run_id}"))
        else:
            seen_ids.add(run_id)

        primary = run.get("primary_summary")
        if not primary:
            issues.append(ManifestIssue(f"run {run_id or index} is missing primary_summary"))
        else:
            artifact_paths.add(str(primary))

        artifacts = run.get("artifacts", [])
        if not isinstance(artifacts, list) or not artifacts:
            issues.append(ManifestIssue(f"run {run_id or index} must list at least one artifact"))
        for artifact in artifacts:
            artifact_paths.add(str(artifact))

    for artifact in sorted(artifact_paths):
        if not (REPO_ROOT / artifact).exists():
            issues.append(ManifestIssue(f"manifest artifact does not exist: {artifact}"))

    indexed = artifact_paths | index_artifact_paths()
    for claim_path in sorted(claim_data_links()):
        if claim_path not in indexed:
            issues.append(ManifestIssue(f"claim data artifact is not indexed: {claim_path}"))

    return issues


def current_date() -> str:
    return dt.datetime.now().date().isoformat()


def update_last_updated(path: Path, date: str) -> None:
    if not path.exists():
        return
    text = path.read_text()
    text = re.sub(r"Last updated: \d{4}-\d{2}-\d{2}", f"Last updated: {date}", text, count=1)
    path.write_text(text)


def replace_auto_block(path: Path, block: str) -> None:
    text = path.read_text()
    wrapped = f"{AUTO_START}\n{block.rstrip()}\n{AUTO_END}"
    pattern = re.compile(re.escape(AUTO_START) + r".*?" + re.escape(AUTO_END), re.S)
    if pattern.search(text):
        text = pattern.sub(wrapped, text, count=1)
    else:
        marker = re.search(r"Last updated: .*\n", text)
        insert_at = marker.end() if marker else 0
        text = text[:insert_at] + "\n" + wrapped + "\n" + text[insert_at:]
    path.write_text(text)


def update_readme_active_focus(active: str) -> None:
    path = REPO_ROOT / "README.md"
    if not path.exists() or active == "unknown":
        return
    text = path.read_text()
    text = re.sub(
        r"- `demian_native_[^`]+` as the active custom architecture",
        f"- `{active}` as the active custom architecture",
        text,
        count=1,
    )
    text = re.sub(
        r"Active architecture lab\. `demian_native_[^`]+` is the primary substrate;",
        f"Active architecture lab. `{active}` is the primary substrate;",
        text,
        count=1,
    )
    text = re.sub(
        r"`demian_native_v0` to `demian_native_[^`]+`",
        f"`demian_native_v0` to `{active}`",
        text,
        count=1,
    )
    text = re.sub(
        r"development/run_substrate_stress_tests\.py --substrate demian_native_[^\s]+",
        f"development/run_substrate_stress_tests.py --substrate {active}",
        text,
        count=1,
    )
    path.write_text(text)


def update_research_map_active_focus(active: str) -> None:
    path = REPO_ROOT / "docs" / "RESEARCH_MAP.md"
    if not path.exists() or active == "unknown":
        return
    text = path.read_text()
    text = re.sub(
        r"Inside the substrate lab, the active working substrate is now `demian_native_[^`]+`\.",
        f"Inside the substrate lab, the active working substrate is now `{active}`.",
        text,
        count=1,
    )
    text = re.sub(
        r"- `demian_native_[^`]+` as the default research substrate",
        f"- `{active}` as the default research substrate",
        text,
        count=1,
    )
    text = text.replace("the active working substrate is now `demian_native_v5.3`", f"the active working substrate is now `{active}`")
    text = text.replace("`demian_native_v5.3` as the default research substrate", f"`{active}` as the default research substrate")
    text = text.replace("`demian_native_v5.3` is the active working substrate", f"`{active}` is the active working substrate")
    text = text.replace("override `demian_native_v5.3` as the default substrate", f"override `{active}` as the default substrate")
    text = text.replace("Keep `demian_native_v5.3` as the main scaffold", f"Keep `{active}` as the main scaffold")
    text = text.replace("`demian_native_v5.3`: current architectural scaffold", f"`{active}`: current architectural scaffold")
    path.write_text(text)


def update_claims_current_substrate(active: str) -> None:
    path = REPO_ROOT / "docs" / "CLAIMS.md"
    if not path.exists() or active == "unknown":
        return
    text = path.read_text()
    text = re.sub(
        r"### C3\. `demian_native_[^`]+` is the active custom architecture line",
        f"### C3. `{active}` is the active custom architecture line",
        text,
        count=1,
    )
    text = re.sub(
        r"### C5b\. `demian_native_[^`]+` is the operational default research substrate",
        f"### C5b. `{active}` is the operational default research substrate",
        text,
        count=1,
    )
    text = text.replace("`demian_native_v5.3` is the active custom architecture line", f"`{active}` is the active custom architecture line")
    text = text.replace("`demian_native_v5.3` is the operational default research substrate", f"`{active}` is the operational default research substrate")
    text = text.replace("`demian_native_v5.3` is the current phase-aware anti-locking refinement of the native line", f"`{active}` is the current endogenous-controller refinement of the native line")
    text = text.replace("`v5.3` adds explicit phase state, lock-risk tracking, recovery handling, and phase-scaled route and credit updates on top of the earlier `v5.x` line", "`v6` adds endogenous observer/value/actuator control, multi-horizon critic signals, and retained conflict-potential transformation on top of the earlier native line")
    text = text.replace("a newer native variant replacing `v5.3` as the active refinement", f"a newer native variant replacing `{active}` as the active refinement")
    path.write_text(text)


def build_status_markdown(
    registry: list[str],
    active_names: list[str],
    tests: list[CommandResult],
    link_issues: list[LinkIssue],
    manifest_issues: list[ManifestIssue],
) -> str:
    generated = current_date()
    branch = command_output(["git", "branch", "--show-current"]) or "unknown"
    active = active_names[0] if active_names else "unknown"
    latest = latest_native_substrate(registry)
    test_status = "not run" if not tests else ("pass" if all(result.returncode == 0 for result in tests) else "fail")
    link_ok = not link_issues
    manifest_ok = not manifest_issues
    native = [name for name in registry if name.startswith("demian_native_")]
    lines = [
        "# Auto Status",
        "",
        "<!-- This file is generated by development/update_docs.py. Do not edit by hand. -->",
        "",
        f"Generated: {generated}",
        f"Branch: `{branch}`",
        f"Active substrate from working state: `{active}`",
        f"Latest registered native substrate: `{latest}`",
        f"Registered substrates: `{len(registry)}` total, `{len(native)}` native aliases",
        "",
        "## Verification",
        "",
        f"- Tests: `{test_status}`",
        f"- Evidence links: `{'pass' if link_ok else 'fail'}`",
        f"- Artifact manifest: `{'pass' if manifest_ok else 'fail'}`",
        "",
        "## Test Commands",
        "",
    ]
    for result in tests:
        status = "pass" if result.returncode == 0 else "fail"
        lines.append(f"- `{status}` `{result.command}` ({result.duration_s:.1f}s)")
    if not tests:
        lines.append("- `not run` tests were skipped")
    lines.extend(["", "## Evidence Link Issues", ""])
    if link_issues:
        for issue in link_issues:
            lines.append(f"- `{issue.source}:{issue.line}` -> `{issue.target}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Manifest Issues", ""])
    if manifest_issues:
        for issue in manifest_issues:
            lines.append(f"- {issue.message}")
    else:
        lines.append("- none")
    lines.extend(["", "## Native Registry", ""])
    for name in native:
        lines.append(f"- `{name}`")
    return "\n".join(lines) + "\n"


def build_embedded_block(
    tests: list[CommandResult],
    link_issues: list[LinkIssue],
    manifest_issues: list[ManifestIssue],
    active: str,
    latest: str,
) -> str:
    test_status = "not run" if not tests else ("pass" if all(result.returncode == 0 for result in tests) else "fail")
    link_ok = not link_issues
    return "\n".join(
        [
            f"Generated by `development/update_docs.py` on {current_date()}.",
            "",
            f"- Active substrate: `{active}`",
            f"- Latest registered native substrate: `{latest}`",
            f"- Local verification: `{test_status}`",
            f"- Evidence link validation: `{'pass' if link_ok else 'fail'}`",
            f"- Artifact manifest validation: `{'pass' if not manifest_issues else 'fail'}`",
            f"- Full generated ledger: [docs/AUTO_STATUS.md](/home/xenith/demian/docs/AUTO_STATUS.md)",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Update generated docs and run local tests")
    parser.add_argument("--check", action="store_true", help="validate docs and manifest without writing files")
    parser.add_argument("--skip-tests", action="store_true", help="update docs without running tests")
    parser.add_argument("--timeout", type=int, default=240, help="timeout per test command in seconds")
    args = parser.parse_args()

    registry = parse_substrate_registry()
    active_names = active_substrates_from_working_state()
    active = active_names[0] if active_names else latest_native_substrate(registry)
    latest = latest_native_substrate(registry)

    tests: list[CommandResult] = []
    if not args.skip_tests:
        for command in discover_test_commands():
            tests.append(run_command(command, args.timeout))

    link_issues = validate_links()
    manifest_issues = validate_manifest()
    if not args.check:
        update_readme_active_focus(active)
        update_research_map_active_focus(active)
        update_claims_current_substrate(active)

        status = build_status_markdown(registry, active_names, tests, link_issues, manifest_issues)
        (REPO_ROOT / "docs" / "AUTO_STATUS.md").write_text(status)
        embedded = build_embedded_block(tests, link_issues, manifest_issues, active, latest)
        replace_auto_block(REPO_ROOT / "docs" / "WORKING_STATE.md", embedded)
        replace_auto_block(REPO_ROOT / "docs" / "CLAIMS.md", embedded)
        replace_auto_block(REPO_ROOT / "data" / "INDEX.md", embedded)

    failed_tests = [result for result in tests if result.returncode != 0]
    if failed_tests:
        print("doc update complete, but tests failed:")
        for result in failed_tests:
            print(f"- {result.command}")
            if result.output_tail:
                print(result.output_tail)
        return 1
    if link_issues:
        print("doc update complete, but evidence links are stale:")
        for issue in link_issues:
            print(f"- {issue.source}:{issue.line} -> {issue.target}")
        return 1
    if manifest_issues:
        print("doc update complete, but manifest validation failed:")
        for issue in manifest_issues:
            print(f"- {issue.message}")
        return 1
    print("doc check complete" if args.check else "doc update complete")
    if tests:
        print(f"tests passed: {len(tests)}/{len(tests)}")
    print(f"active substrate: {active}")
    print(f"latest native substrate: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
