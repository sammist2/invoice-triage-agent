"""The agent loop.

For each inbox item:
  1. Send the system prompt + the message (with any PDF text) to the model.
  2. If the model asks to call tools, run them and send back the results.
  3. Repeat until the model calls submit_result, or we hit the step limit.

This is the same loop that agent frameworks (LangGraph, the OpenAI/Anthropic
agent SDKs...) run for you. Writing it by hand once is the best way to
understand what those frameworks do.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from agent.config import Settings
from agent.documents import DATA_DIR, format_item
from agent.llm import LLMClient, LLMError
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOL_SCHEMAS, ApiClient, ToolBox

MAX_STEPS = 10
NUDGE = ("You did not call a tool. Follow the policy: take exactly one action with a tool, "
         "then call submit_result.")
NUDGE_SUBMIT = ("You already took your action, so do not take another one. "
                "Now call submit_result with the category and the extracted fields.")


def build_record(item: dict, toolbox: ToolBox, settings: Settings, *, steps: int, error: str | None,
                 latency_s: float, prompt_tokens: int = 0, completion_tokens: int = 0,
                 server_error: bool = False) -> dict:
    """One line of the results file. The eval reads these."""
    result = toolbox.result.model_dump() if toolbox.result else {}
    cost = (prompt_tokens * settings.input_cost_per_m + completion_tokens * settings.output_cost_per_m) / 1_000_000
    return {
        "id": item["id"],
        "provider": settings.provider,
        "model": settings.model,
        "category": result.get("category"),
        "order_id": result.get("order_id"),
        "invoice_number": result.get("invoice_number"),
        "po_number": result.get("po_number"),
        "amount": result.get("amount"),
        "actions": toolbox.actions,
        "steps": steps,
        "error": error,
        "server_error": server_error,  # True when the model server failed, not the model's reasoning
        "latency_s": round(latency_s, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 6),
        "tool_calls": toolbox.calls,
    }


def _json_from_text(text: str) -> dict | None:
    """Some small models write the final answer as JSON text instead of calling the tool."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "category" in data else None


def process_item(item: dict, llm: LLMClient, api: ApiClient, settings: Settings, run_name: str,
                 data_dir: Path = DATA_DIR, max_steps: int = MAX_STEPS) -> dict:
    toolbox = ToolBox(api, run_name, item)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_item(item, data_dir)},
    ]
    prompt_tokens = completion_tokens = 0
    error: str | None = None
    nudged = False
    server_error = False
    steps = 0
    start = time.perf_counter()

    try:
        while toolbox.result is None:
            if steps >= max_steps:
                error = f"stopped after {max_steps} steps without calling submit_result"
                break
            steps += 1
            response = llm.chat(messages, TOOL_SCHEMAS)
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens

            if not response.tool_calls:
                parsed = _json_from_text(response.content)
                if parsed is not None:
                    toolbox.execute("submit_result", parsed)
                    if toolbox.result is not None:
                        break
                if nudged:
                    error = "model replied with text instead of calling tools"
                    break
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": NUDGE_SUBMIT if toolbox.actions else NUDGE})
                nudged = True
                continue

            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments}}
                               for tc in response.tool_calls],
            })
            for tc in response.tool_calls:
                output = toolbox.execute(tc.name, tc.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
    except LLMError as exc:
        error = str(exc)
        server_error = True

    return build_record(item, toolbox, settings, steps=steps, error=error,
                        latency_s=time.perf_counter() - start,
                        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                        server_error=server_error)
