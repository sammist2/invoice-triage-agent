"""A small client for any OpenAI-compatible chat API (Ollama, vLLM, Gemini, Groq...).

Written with plain `requests` instead of an SDK so you can see exactly what
goes over the wire: one POST to {base_url}/chat/completions with the
conversation and the list of tools the model is allowed to call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import requests

from agent.config import Settings

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4


class LLMError(Exception):
    """The model server could not be reached or returned an error."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as the model wrote it


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        self.url = f"{settings.base_url}/chat/completions"
        self._call_counter = 0

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,  # same input -> (nearly) same output, which makes evals fair
        }
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = self.session.post(self.url, json=payload, headers=headers, timeout=self.settings.timeout_s)
            except requests.RequestException as exc:
                raise LLMError(
                    f"Could not reach the model server at {self.settings.base_url} ({exc.__class__.__name__}). "
                    "Is it running? For Ollama, start the Ollama app or run `ollama serve`."
                ) from exc

            if resp.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS:
                # Free tiers often rate-limit (429). Wait and try again.
                wait = _retry_after_seconds(resp) or 2 ** attempt
                time.sleep(min(wait, 60))
                continue
            if not resp.ok:
                raise LLMError(f"Model server returned HTTP {resp.status_code}: {resp.text[:500]}")
            break

        try:
            data = resp.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response from model server: {resp.text[:500]}") from exc

        tool_calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):  # some servers send a dict instead of JSON text
                args = json.dumps(args if args is not None else {})
            self._call_counter += 1
            tool_calls.append(ToolCall(id=raw.get("id") or f"call_{self._call_counter}",
                                       name=fn.get("name") or "", arguments=args))

        usage = data.get("usage") or {}
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )


def _retry_after_seconds(resp: requests.Response) -> float | None:
    try:
        return float(resp.headers.get("Retry-After", ""))
    except ValueError:
        return None
