#!/usr/bin/env python3
"""Launch OpenClaude through OpenRouter with rotating free models.

The script intentionally stores only model/cache state. API keys stay in the
environment, so this can live in the repo without committing credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CACHE_DIR = Path(".openrouter")
CACHE_FILE = CACHE_DIR / "free-models.json"
STATE_FILE = CACHE_DIR / "openclaude-rotation.json"
CACHE_TTL_SECONDS = 6 * 60 * 60

STATIC_FALLBACK_MODELS = [
    "openrouter/owl-alpha",
    "poolside/laguna-m.1:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "inclusionai/ling-2.6-1t:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "poolside/laguna-xs.2:free",
    "minimax/minimax-m2.5:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3n-e4b-it:free",
]

FORCED_FREE_MODELS = [
    "openrouter/owl-alpha",
]

ALLOWED_MODELS = set(STATIC_FALLBACK_MODELS)

CODING_HINTS = (
    "coder",
    "coding",
    "code",
    "agent",
    "gpt-oss",
    "poolside",
    "laguna",
    "qwen",
    "deepseek",
    "kimi",
    "glm",
    "nemotron",
    "mistral",
    "llama",
)

PREFERRED_MODEL_BONUS = {
    "openrouter/owl-alpha": 100,
    "poolside/laguna-m.1:free": 90,
    "liquid/lfm-2.5-1.2b-instruct:free": 80,
    "inclusionai/ling-2.6-1t:free": 70,
    "nvidia/nemotron-3-nano-30b-a3b:free": 60,
    "nvidia/nemotron-3-super-120b-a12b:free": 50,
    "poolside/laguna-xs.2:free": 40,
    "minimax/minimax-m2.5:free": 30,
    "google/gemma-3-4b-it:free": 20,
    "google/gemma-3n-e4b-it:free": 10,
}

EXCLUDED_MODELS = {
    "openai/gpt-oss-20b:free",
    "qwen/qwen3-coder:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "tencent/hy3-preview:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3n-e2b-it:free",
    "baidu/qianfan-ocr-fast:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "z-ai/glm-4.5-air:free",
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


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _is_zero(value: Any) -> bool:
    try:
        return float(value or 0) == 0
    except (TypeError, ValueError):
        return False


def _fetch_models(timeout: float) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        OPENROUTER_MODELS_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "demian-openclaude-openrouter-rotation/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("data", [])


def _is_free_model(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    model_id = str(model.get("id") or "")
    return (
        bool(model_id)
        and model_id.endswith(":free")
        and _is_zero(pricing.get("prompt"))
        and _is_zero(pricing.get("completion"))
    )


def _score_model(model: dict[str, Any]) -> tuple[int, int, int, str]:
    model_id = str(model.get("id") or "")
    name = str(model.get("name") or "")
    haystack = f"{model_id} {name} {model.get('description') or ''}".lower()
    supported = set(model.get("supported_parameters") or [])
    top_provider = model.get("top_provider") or {}
    context = int(top_provider.get("context_length") or model.get("context_length") or 0)
    expires = str(model.get("expiration_date") or "")

    hint_score = sum(1 for hint in CODING_HINTS if hint in haystack)
    known_good_score = PREFERRED_MODEL_BONUS.get(model_id, 0)
    tool_score = 3 if {"tools", "tool_choice"} <= supported else 0
    expiry_penalty = -4 if expires else 0
    context_bucket = min(context // 32768, 16)
    return (known_good_score + tool_score + hint_score + expiry_penalty, context_bucket, context, model_id)


def _rank_free_models(models: list[dict[str, Any]]) -> list[str]:
    free_models = [
        model
        for model in models
        if (
            _is_free_model(model)
            and str(model.get("id") or "") in ALLOWED_MODELS
            and str(model.get("id") or "") not in EXCLUDED_MODELS
        )
    ]
    ranked = sorted(free_models, key=_score_model, reverse=True)
    return [str(model["id"]) for model in ranked]


def _configured_models() -> list[str]:
    value = os.environ.get("OPENROUTER_FREE_MODELS", "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_forced_models(models: list[str]) -> list[str]:
    active_models = list(models)
    for forced in FORCED_FREE_MODELS:
        if forced not in active_models:
            active_models.insert(0, forced)
    return active_models


def _models_from_cache_or_network(refresh: bool, timeout: float) -> list[str]:
    configured = _configured_models()
    if configured:
        return configured

    cache = _load_json(CACHE_FILE, {})
    cache_age = time.time() - float(cache.get("fetched_at") or 0)
    if not refresh and cache.get("models") and cache_age < CACHE_TTL_SECONDS:
        return _apply_forced_models(list(cache["models"]))

    try:
        models = _rank_free_models(_fetch_models(timeout))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        cached = list(cache.get("models") or [])
        if cached:
            print(f"openrouter model refresh failed; using cache: {exc}", file=sys.stderr)
            return _apply_forced_models(cached)
        print(f"openrouter model refresh failed; using static fallback: {exc}", file=sys.stderr)
        return _apply_forced_models(STATIC_FALLBACK_MODELS)

    if not models:
        return _apply_forced_models(STATIC_FALLBACK_MODELS)

    models = _apply_forced_models(models)
    _write_json(
        CACHE_FILE,
        {
            "fetched_at": time.time(),
            "models": models,
            "source": OPENROUTER_MODELS_URL,
        },
    )
    return models


def _choose_model(models: list[str], explicit_model: str | None) -> tuple[str, int]:
    if explicit_model:
        return explicit_model, -1

    state = _load_json(STATE_FILE, {})
    previous_index = int(state.get("index") or -1)
    index = (previous_index + 1) % len(models)
    model = models[index]
    _write_json(
        STATE_FILE,
        {
            "index": index,
            "last_model": model,
            "last_started_at": time.time(),
            "model_count": len(models),
        },
    )
    return model, index


def _build_env(model: str) -> dict[str, str]:
    env = os.environ.copy()
    api_key = env.get("OPENROUTER_API_KEY") or env.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set OPENROUTER_API_KEY first. Example:\n"
            "  export OPENROUTER_API_KEY='sk-or-v1-...'"
        )

    env["CLAUDE_CODE_USE_OPENAI"] = "1"
    env["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL", OPENROUTER_BASE_URL)
    env["OPENAI_API_KEY"] = api_key
    env["OPENROUTER_API_KEY"] = api_key
    env["OPENAI_MODEL"] = model
    env.setdefault("OPENAI_SHIM_TOOL_MODE", "minify")
    return env


def _probe_model(model: str, timeout: float) -> tuple[bool, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENROUTER_API_KEY is not set"

    request = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Test model compatibility with Demian."}
                ],
                "max_tokens": 1,
                "temperature": 0,
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "demian-openclaude-openrouter-probe/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return True, json.dumps(payload if isinstance(payload, dict) else payload, indent=2)[:1000]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return False, f"HTTPError {exc.code}: {body}"
    except urllib.error.URLError as exc:
        return False, f"URLError: {exc}"
    except Exception as exc:
        return False, f"Probe failure: {exc}"


def _has_option(args: list[str], *names: str) -> bool:
    return any(arg == name or any(arg.startswith(f"{name}=") for name in names) for arg in args for name in names)


def _openclaude_command(model: str, openclaude_args: list[str]) -> list[str]:
    args = list(openclaude_args)
    if args and args[0] == "--":
        args = args[1:]

    command = ["openclaude"]
    if not _has_option(args, "--provider"):
        command.extend(["--provider", "openai"])
    if not _has_option(args, "--model"):
        command.extend(["--model", model])
    command.extend(args)
    return command


def main() -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run OpenClaude against rotating OpenRouter free models."
    )
    parser.add_argument("--list", action="store_true", help="List candidate models and exit")
    parser.add_argument("--refresh", action="store_true", help="Refresh OpenRouter model cache")
    parser.add_argument("--probe", action="store_true", help="Probe each candidate model with a tiny OpenRouter request")
    parser.add_argument("--model", help="Use one model without advancing rotation")
    parser.add_argument("--timeout", type=float, default=10.0, help="OpenRouter fetch timeout")
    parser.add_argument(
        "openclaude_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to openclaude after --",
    )
    args = parser.parse_args()

    models = _models_from_cache_or_network(refresh=args.refresh, timeout=args.timeout)
    if args.list:
        for model in models:
            print(model)
        return 0

    if args.probe:
        for model in models:
            print(f"Probing {model}...")
            ok, message = _probe_model(model, timeout=args.timeout)
            print("  OK" if ok else "  FAIL", message)
        return 0

    model, index = _choose_model(models, args.model)
    env = _build_env(model)

    if index >= 0:
        print(f"OpenClaude via OpenRouter: {model} ({index + 1}/{len(models)})", flush=True)
    else:
        print(f"OpenClaude via OpenRouter: {model}", flush=True)

    return subprocess.call(_openclaude_command(model, args.openclaude_args), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
