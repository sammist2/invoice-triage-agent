"""Run the agent (or the baseline) over the whole inbox and save the results.

Examples:
  python -m agent.run --provider baseline
  python -m agent.run --provider ollama --model qwen2.5:7b
  python -m agent.run --provider ollama --limit 5            # quick test on 5 items
  python -m agent.run --provider gemini --name gemini-prompt-v2
  python -m agent.run --provider ollama --resume             # continue a run that was stopped

Results go to results/<name>.jsonl, one line per item. Then score them with:
  python -m evals.evaluate results/<name>.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from agent.agent import MAX_STEPS, process_item
from agent.baseline import process_item_baseline
from agent.config import PROJECT_DIR, load_settings
from agent.documents import DATA_DIR, load_jsonl
from agent.llm import LLMClient
from agent.tools import ApiClient, ApiUnavailable

RESULTS_DIR = PROJECT_DIR / "results"
MAX_SERVER_ERRORS_IN_A_ROW = 3  # stop early instead of failing every item when the model server is down


def default_run_name(provider: str, model: str) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9.]+", "-", model).strip("-").lower()
    return provider if provider == "baseline" else f"{provider}-{safe_model}"


def read_records(path: Path) -> list[dict]:
    """Read a results file, skipping a half-written last line if a run was killed mid-write."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process the inbox with the agent or the baseline.")
    parser.add_argument("--provider", help="baseline, ollama, vllm, gemini, groq or custom")
    parser.add_argument("--model", help="model name (overrides LLM_MODEL)")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL (overrides LLM_BASE_URL)")
    parser.add_argument("--name", help="name for this run; results go to results/<name>.jsonl")
    parser.add_argument("--limit", type=int, help="only process the first N items")
    parser.add_argument("--ids", help="only these item ids, comma separated (e.g. item-001,item-007)")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--items", type=Path, default=DATA_DIR / "items.jsonl")
    parser.add_argument("--out", type=Path, help="results file (default: results/<name>.jsonl)")
    parser.add_argument("--resume", action="store_true",
                        help="keep finished items from an earlier run with the same name and only do the rest "
                             "(items that ended in an error are redone)")
    args = parser.parse_args(argv)

    settings = load_settings(args.provider, args.model, args.base_url)
    run_name = args.name or default_run_name(settings.provider, settings.model)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        print(f"Run name {run_name!r} may only contain letters, numbers, '.', '_' and '-'.")
        return 1

    api = ApiClient(settings.api_url)
    if not api.health():
        print(f"The business API is not running at {settings.api_url}.\n"
              "Open a second terminal and run:  python -m api.app")
        return 1

    if not args.items.exists():
        print(f"{args.items} not found. Generate the data first:  python -m data.generate_data")
        return 1
    items = load_jsonl(args.items)
    data_dir = args.items.resolve().parent
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        items = [it for it in items if it["id"] in wanted]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("No items selected.")
        return 1

    out_path = args.out or RESULTS_DIR / f"{run_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    if args.resume and out_path.exists():
        kept = [r for r in read_records(out_path) if not r.get("error")]
        finished = {r["id"] for r in kept}
        items = [it for it in items if it["id"] not in finished]
        print(f"Resuming: {len(kept)} items already done in {out_path.name}, {len(items)} to go.")
    else:
        api.request("DELETE", f"/runs/{run_name}")  # start clean if this run name was used before

    llm = None if settings.provider == "baseline" else LLMClient(settings)
    print(f"Run '{run_name}': {len(items)} items with {settings.provider} / {settings.model}")
    print(f"Writing results to {out_path}\n")

    started = time.perf_counter()
    errors = 0
    done = 0
    server_errors_in_a_row = 0
    try:
        with open(out_path, "w", encoding="utf-8") as out:
            for record in kept:
                out.write(json.dumps(record) + "\n")
            out.flush()
            for n, item in enumerate(items, start=1):
                if llm is None:
                    record = process_item_baseline(item, api, settings, run_name, data_dir)
                else:
                    record = process_item(item, llm, api, settings, run_name, data_dir, args.max_steps)
                out.write(json.dumps(record) + "\n")
                out.flush()  # save as we go, so a crash or Ctrl+C keeps finished items
                done += 1
                actions = ", ".join(record["actions"]) or "no action"
                line = f"[{n:>3}/{len(items)}] {item['id']}  {record['category'] or '?':<17} {actions:<24} {record['latency_s']:.1f}s"
                if record["error"]:
                    errors += 1
                    line += f"  ERROR: {record['error'][:120]}"
                print(line, flush=True)

                server_errors_in_a_row = server_errors_in_a_row + 1 if record.get("server_error") else 0
                if server_errors_in_a_row >= MAX_SERVER_ERRORS_IN_A_ROW:
                    print(f"\nStopping: the model server failed {server_errors_in_a_row} times in a row.\n"
                          f"Last error: {record['error']}\n"
                          "Fix the problem, then run the same command with --resume to continue.")
                    return 1
    except ApiUnavailable as exc:
        print(f"\n{exc}")
        return 1
    except KeyboardInterrupt:
        print("\nStopped early. Finished items were saved.")

    elapsed = time.perf_counter() - started
    print(f"\nDone: {done} items in {elapsed:.1f}s, {errors} with errors.")
    try:
        shown = out_path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:  # results file is outside the project folder
        shown = str(out_path)
    print(f"Score it with:  python -m evals.evaluate {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
