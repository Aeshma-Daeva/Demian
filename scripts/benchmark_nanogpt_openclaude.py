#!/usr/bin/env python3
"""Compare NanoGPT models on Demian-specific OpenClaude tasks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "model_benchmarks"
NANOGPT_BASE_URL = "https://nano-gpt.com/api/v1"

MODEL_GROUPS = {
    "core": [
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro-cheaper",
        "qwen/qwen3-coder",
        "qwen3-coder-30b-a3b-instruct",
    ],
    "deepseek": [
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-pro-cheaper",
        "TEE/deepseek-v4-pro",
    ],
    "coding": [
        "deepseek/deepseek-v4-flash",
        "qwen/qwen3-coder",
        "qwen/qwen3-coder-next",
        "qwen3-coder-30b-a3b-instruct",
    ],
}

TASKS = [
    {
        "id": "orientation",
        "prompt": (
            "Read docs/CURRENT_STATE_AND_ROUTING.md and docs/WORKING_STATE.md. "
            "Report the active substrate, the unpromoted latest line, and the next two experiment questions. "
            "Do not edit files."
        ),
    },
    {
        "id": "code_navigation",
        "prompt": (
            "Inspect development/substrate_lab.py. Find DemianNativeV8Substrate and DemianNativeV74Substrate. "
            "Summarize the practical architectural difference in four bullets. Do not edit files."
        ),
    },
    {
        "id": "artifact_reasoning",
        "prompt": (
            "Read data/substrate_lab/v8_baseline_compare_20260505/summary.json and "
            "data/substrate_lab/v85_baseline_20260505/summary.json. "
            "Compare v8, v7.4, and v8.5 in practical terms. Keep claims conservative. Do not edit files."
        ),
    },
    {
        "id": "tool_efficiency",
        "prompt": (
            "Use targeted search and reads only. Identify the files that define NanoGPT/OpenClaude routing "
            "and the doc that explains current substrate state. Return only file paths and one-line purposes. "
            "Do not edit files."
        ),
    },
]


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _usage_from_result(result: dict[str, Any], model: str) -> dict[str, Any]:
    model_usage = result.get("modelUsage") or {}
    usage = model_usage.get(model) or next(iter(model_usage.values()), {})
    raw_usage = result.get("usage") or {}
    return {
        "input_tokens": usage.get("inputTokens") or raw_usage.get("input_tokens") or 0,
        "output_tokens": usage.get("outputTokens") or raw_usage.get("output_tokens") or 0,
        "cache_creation_tokens": usage.get("cacheCreationInputTokens") or 0,
        "cache_read_tokens": usage.get("cacheReadInputTokens") or 0,
        "cost_usd": usage.get("costUSD") or result.get("total_cost_usd") or 0,
        "turns": result.get("num_turns") or 0,
        "duration_ms": result.get("duration_ms") or 0,
    }


def _run_case(model: str, task: dict[str, str], timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    api_key = env.get("NANOGPT_API_KEY") or env.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set NANOGPT_API_KEY first.")

    env["CLAUDE_CODE_USE_OPENAI"] = "1"
    env["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL", NANOGPT_BASE_URL)
    env["OPENAI_API_KEY"] = api_key
    env["NANOGPT_API_KEY"] = api_key
    env["OPENAI_MODEL"] = model
    env.setdefault("OPENAI_SHIM_TOOL_MODE", "minify")

    command = [
        "openclaude",
        "--provider",
        "openai",
        "--model",
        model,
        "--bare",
        "-p",
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        task["prompt"],
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.time() - started) * 1000)
        stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "model": model,
            "task": task["id"],
            "ok": False,
            "exit_code": "timeout",
            "elapsed_ms": elapsed_ms,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "cost_usd": 0,
                "turns": 0,
                "duration_ms": elapsed_ms,
            },
            "result_preview": stdout.replace("\n", " ")[:500],
            "stderr_preview": stderr.replace("\n", " ")[:500],
        }
    elapsed_ms = int((time.time() - started) * 1000)

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    parsed: dict[str, Any] = {}
    if lines:
        try:
            parsed = json.loads(lines[-1])
        except json.JSONDecodeError:
            parsed = {}

    return {
        "model": model,
        "task": task["id"],
        "ok": proc.returncode == 0 and parsed.get("type") == "result" and not parsed.get("is_error"),
        "exit_code": proc.returncode,
        "elapsed_ms": elapsed_ms,
        "usage": _usage_from_result(parsed, model),
        "result_preview": str(parsed.get("result") or proc.stdout).replace("\n", " ")[:500],
        "stderr_preview": proc.stderr.replace("\n", " ")[:500],
    }


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Benchmark NanoGPT models on Demian OpenClaude tasks.")
    parser.add_argument("--group", choices=sorted(MODEL_GROUPS), default="core")
    parser.add_argument("--model", action="append", help="Explicit model ID; can be repeated")
    parser.add_argument("--task", choices=[task["id"] for task in TASKS], action="append")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    models = args.model or MODEL_GROUPS[args.group]
    task_ids = set(args.task or [task["id"] for task in TASKS])
    tasks = [task for task in TASKS if task["id"] in task_ids]

    rows = []
    for model in models:
        for task in tasks:
            print(f"running {model} :: {task['id']}", flush=True)
            rows.append(_run_case(model, task, timeout=args.timeout))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or RESULTS_DIR / f"nanogpt_openclaude_{int(time.time())}.json"
    out.write_text(json.dumps({"models": models, "tasks": tasks, "results": rows}, indent=2) + "\n")

    print(f"\nWrote {out}")
    print("model\ttask\tok\tinput\toutput\tcost\tturns\telapsed_ms")
    for row in rows:
        usage = row["usage"]
        print(
            "\t".join(
                [
                    row["model"],
                    row["task"],
                    str(row["ok"]),
                    str(usage["input_tokens"]),
                    str(usage["output_tokens"]),
                    f"{float(usage['cost_usd']):.6f}",
                    str(usage["turns"]),
                    str(row["elapsed_ms"]),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
