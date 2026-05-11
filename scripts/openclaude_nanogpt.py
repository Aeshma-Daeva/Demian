#!/usr/bin/env python3
"""Launch OpenClaude through NanoGPT with role-based model presets."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


NANOGPT_BASE_URL = "https://nano-gpt.com/api/v1"

MODEL_PRESETS = {
    "default": "deepseek/deepseek-v4-flash",
    "reason": "deepseek/deepseek-v4-pro",
    "code": "deepseek/deepseek-v4-flash",
    "qwen": "qwen/qwen3-coder",
    "edit": "qwen3-coder-30b-a3b-instruct",
    "cheap": "zai-org/glm-4.7-flash",
    "general": "minimax/minimax-m2.5",
    "scout": "meta-llama/llama-4-scout",
}


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overriding the shell."""
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


def _has_option(args: list[str], *names: str) -> bool:
    return any(
        arg == name or any(arg.startswith(f"{name}=") for name in names)
        for arg in args
        for name in names
    )


def _build_env(model: str) -> dict[str, str]:
    env = os.environ.copy()
    api_key = env.get("NANOGPT_API_KEY") or env.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set NANOGPT_API_KEY first. Example:\n"
            "  export NANOGPT_API_KEY='...'"
        )

    env["CLAUDE_CODE_USE_OPENAI"] = "1"
    env["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL", NANOGPT_BASE_URL)
    env["OPENAI_API_KEY"] = api_key
    env["NANOGPT_API_KEY"] = api_key
    env["OPENAI_MODEL"] = model
    env.setdefault("OPENAI_SHIM_TOOL_MODE", "minify")
    return env


def _openclaude_command(model: str, openclaude_args: list[str]) -> list[str]:
    args = list(openclaude_args)
    if args and args[0] == "--":
        args = args[1:]

    command = ["openclaude"]
    if not _has_option(args, "--model"):
        command.extend(["--model", model])
    command.extend(args)
    return command


def main() -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run OpenClaude against NanoGPT with role-based model presets."
    )
    parser.add_argument(
        "--role",
        choices=sorted(MODEL_PRESETS),
        default="default",
        help="Model preset to use",
    )
    parser.add_argument("--model", help="Use an explicit NanoGPT model ID")
    parser.add_argument("--list", action="store_true", help="List preset roles and exit")
    parser.add_argument(
        "openclaude_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to openclaude after --",
    )
    args = parser.parse_args()

    if args.list:
        for role, model in MODEL_PRESETS.items():
            print(f"{role}\t{model}")
        return 0

    model = args.model or MODEL_PRESETS[args.role]
    env = _build_env(model)
    print(f"OpenClaude via NanoGPT: {model} ({args.role})", flush=True)
    return subprocess.call(_openclaude_command(model, args.openclaude_args), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
