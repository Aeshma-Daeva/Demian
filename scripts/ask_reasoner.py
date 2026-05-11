#!/usr/bin/env python3
"""Ask the Demian reasoner model for a compact checkpoint opinion.

This is intentionally a direct API call, not a nested OpenClaude session. The
worker should pass compact metrics, diffs, or summaries instead of raw artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


NANOGPT_BASE_URL = "https://nano-gpt.com/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro-cheaper"

DEFAULT_SYSTEM_PROMPT = """You are the Demian reasoner checkpoint.

Evaluate compact evidence from the worker. Be conservative and practical.
Separate observation, inference, and speculation. Say whether a claim should be
promoted, held as unpromoted, or rejected. Prefer short answers with concrete
risks and next checks. Do not ask for broad raw context unless the evidence is
insufficient.
"""


def _load_dotenv(path: Path = Path(".env")) -> None:
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


def _read_prompt(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.prompt:
        parts.append(" ".join(args.prompt))
    if args.stdin or not parts:
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            parts.append(stdin_text)
    for path in args.context:
        parts.append(f"\n\nContext file: {path}\n{Path(path).read_text()}")
    return "\n\n".join(parts).strip()


def _extract_content(payload: dict[str, Any]) -> str:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text") or item) for item in content)
    return json.dumps(payload, indent=2)


def main() -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Ask the Demian reasoner model.")
    parser.add_argument("prompt", nargs="*", help="Question/evidence for the reasoner")
    parser.add_argument("--model", default=os.environ.get("DEMIAN_REASONER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("NANOGPT_BASE_URL", NANOGPT_BASE_URL))
    parser.add_argument("--context", action="append", default=[], help="Small context file to append")
    parser.add_argument("--stdin", action="store_true", help="Read additional prompt text from stdin")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--usage", action="store_true", help="Print token usage to stderr when available")
    args = parser.parse_args()

    prompt = _read_prompt(args)
    if not prompt:
        raise SystemExit("Provide a prompt, --context, or stdin.")

    api_key = os.environ.get("NANOGPT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set NANOGPT_API_KEY first.")

    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "demian-reasoner-checkpoint/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"Reasoner HTTP {exc.code}: {body_text}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Reasoner request failed: {exc}")

    print(_extract_content(payload).strip())
    if args.usage and payload.get("usage"):
        print(f"\n[reasoner usage] {json.dumps(payload['usage'], sort_keys=True)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
