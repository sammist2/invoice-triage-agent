"""Settings for which model to use. This is the "model switcher".

Every provider below speaks the same OpenAI-style chat API, so switching
between a local model and a hosted one is just a different base URL and
model name. Settings come from (highest priority first):
  1. command-line flags  (python -m agent.run --provider ollama --model qwen2.5:7b)
  2. environment variables / the .env file  (see .env.example)
  3. the defaults in PRESETS below
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

try:  # python-dotenv is optional; it just loads the .env file if you made one
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    pass

# Hosted model names change often, so they have no default here.
# Check the provider's docs for current model names and put one in .env.
PRESETS: dict[str, dict] = {
    "baseline": {"base_url": None, "model": "rules"},  # no LLM: keyword rules to beat
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"},
    "vllm": {"base_url": "http://localhost:8000/v1", "model": "Qwen/Qwen2.5-1.5B-Instruct"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": None},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": None},
    "custom": {"base_url": None, "model": None},  # any other OpenAI-compatible server
}


@dataclass
class Settings:
    provider: str
    model: str
    base_url: str | None
    api_key: str
    api_url: str
    input_cost_per_m: float   # USD per 1M input tokens (0 for local models)
    output_cost_per_m: float  # USD per 1M output tokens
    timeout_s: float = 120.0


def _float_env(name: str) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else 0.0
    except ValueError:
        raise SystemExit(f"{name} must be a number, got {raw!r}")


def load_settings(provider: str | None = None, model: str | None = None, base_url: str | None = None) -> Settings:
    provider = (provider or os.environ.get("LLM_PROVIDER") or "baseline").strip().lower()
    if provider not in PRESETS:
        raise SystemExit(f"Unknown provider {provider!r}. Choose one of: {', '.join(PRESETS)}")
    preset = PRESETS[provider]

    model = model or os.environ.get("LLM_MODEL") or preset["model"]
    base_url = base_url or os.environ.get("LLM_BASE_URL") or preset["base_url"]

    if provider != "baseline":
        if not model:
            raise SystemExit(f"Provider {provider!r} needs a model name. Set LLM_MODEL in .env or pass --model.")
        if not base_url:
            raise SystemExit(f"Provider {provider!r} needs a base URL. Set LLM_BASE_URL in .env or pass --base-url.")

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if provider in ("gemini", "groq") and not api_key:
        raise SystemExit(f"Provider {provider!r} needs an API key. Put LLM_API_KEY=... in your .env file.")

    return Settings(
        provider=provider,
        model=model,
        base_url=base_url.rstrip("/") if base_url else None,
        api_key=api_key or "not-needed",  # local servers ignore the key
        api_url=os.environ.get("API_URL", "http://127.0.0.1:8001").rstrip("/"),
        input_cost_per_m=_float_env("LLM_INPUT_COST_PER_M"),
        output_cost_per_m=_float_env("LLM_OUTPUT_COST_PER_M"),
    )
