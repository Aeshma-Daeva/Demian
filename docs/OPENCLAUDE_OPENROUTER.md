# OpenClaude Through OpenRouter Free Models

This repo has a local launcher for continuing work in OpenClaude when Codex usage is constrained:

```bash
printf 'OPENROUTER_API_KEY=sk-or-v1-...\n' > .env
python3 scripts/openclaude_openrouter.py
```

You can also export `OPENROUTER_API_KEY` in your shell. Shell variables take precedence over `.env`.

The launcher sets OpenClaude's OpenAI-compatible environment:

```bash
CLAUDE_CODE_USE_OPENAI=1
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=$OPENROUTER_API_KEY
OPENAI_MODEL=<rotated-free-model>
OPENAI_SHIM_TOOL_MODE=minify
```

On each start it rotates to the next free OpenRouter model. It fetches the live model list from `https://openrouter.ai/api/v1/models`, keeps zero-priced `:free` models, ranks coding/tool-friendly models first, pins `openrouter/owl-alpha` at the top, and stores only cache/rotation state under `.openrouter/`.

`OPENAI_SHIM_TOOL_MODE=minify` is the default unless you set another value in the shell. This keeps all tools available while stripping verbose tool-schema prose.

For NanoGPT model presets and the current Demian restart map, see [docs/CURRENT_STATE_AND_ROUTING.md](/home/xenith/demian/docs/CURRENT_STATE_AND_ROUTING.md).

Useful commands:

```bash
# Show current candidate rotation list.
python3 scripts/openclaude_openrouter.py --list

# Refresh OpenRouter's model cache now.
python3 scripts/openclaude_openrouter.py --refresh --list

# Probe each candidate model with a tiny OpenRouter request.
python3 scripts/openclaude_openrouter.py --probe

# Pin one model for this launch without advancing rotation.
python3 scripts/openclaude_openrouter.py --model poolside/laguna-m.1:free

# Pass arguments through to OpenClaude.
python3 scripts/openclaude_openrouter.py -- --help
```

To override the dynamic list, set a comma-separated model list:

```bash
export OPENROUTER_FREE_MODELS="poolside/laguna-m.1:free,poolside/laguna-xs.2:free"
python3 scripts/openclaude_openrouter.py
```

Do not commit API keys. Keep them in the shell environment or your local shell profile.
