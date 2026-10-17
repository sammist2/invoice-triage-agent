"""Score agent runs against the answer key (data/labels.jsonl).

Examples:
  python -m evals.evaluate results/baseline.jsonl
  python -m evals.evaluate results/*.jsonl              # compare several runs
  python -m evals.evaluate results/ollama-qwen2.5-7b.jsonl --all-failures

What is measured, per item:
  category  - did it pick the right category?
  action    - did it take exactly one action, and the right one?
              (e.g. a ticket of the right type, or flagging an invoice whose total is wrong)
  fields    - did it extract the order number / invoice number / PO number / amount correctly?
              Only fields that appear in the answer key are checked.
  fully correct - category, action and every field all right.
Plus speed (seconds per item), tokens and cost.

A markdown report is written to results/report.md so you can paste the table into your README.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

from shared.normalize import amounts_equal, normalize_invoice_number, normalize_order_id, normalize_po_number

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = PROJECT_DIR / "data" / "labels.jsonl"
DEFAULT_REPORT = PROJECT_DIR / "results" / "report.md"

FIELD_NORMALIZERS = {
    "order_id": normalize_order_id,
    "invoice_number": normalize_invoice_number,
    "po_number": normalize_po_number,
}
FIELDS = ["order_id", "invoice_number", "po_number", "amount"]
CATEGORY_ORDER = ["refund_request", "shipping_delay", "billing_question", "invoice_received", "spam"]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def field_matches(field: str, expected, predicted) -> bool:
    if field == "amount":
        return amounts_equal(expected, predicted)
    norm = FIELD_NORMALIZERS[field]
    return norm(predicted) is not None and norm(predicted) == norm(expected)


def score_item(label: dict, record: dict) -> dict:
    category_ok = record.get("category") == label["category"]
    actions = record.get("actions") or []
    action_ok = actions == [label["action"]]
    expected_fields = [f for f in FIELDS if label.get(f) is not None]
    wrong_fields = [f for f in expected_fields if not field_matches(f, label[f], record.get(f))]
    return {
        "id": label["id"],
        "category_ok": category_ok,
        "action_ok": action_ok,
        "fields_expected": len(expected_fields),
        "fields_correct": len(expected_fields) - len(wrong_fields),
        "wrong_fields": wrong_fields,
        "fully_ok": category_ok and action_ok and not wrong_fields,
        "error": record.get("error"),
    }


def summarize(records: list[dict], labels: list[dict]) -> dict:
    labels_by_id = {l["id"]: l for l in labels}
    records = [r for r in records if r["id"] in labels_by_id]
    if not records:
        raise SystemExit("None of the results match an id in the labels file.")
    scores = [score_item(labels_by_id[r["id"]], r) for r in records]
    n = len(scores)

    def rate(key: str, subset=scores) -> float:
        return sum(s[key] for s in subset) / len(subset) if subset else 0.0

    fields_expected = sum(s["fields_expected"] for s in scores)
    fields_correct = sum(s["fields_correct"] for s in scores)

    per_category = {}
    by_cat = defaultdict(list)
    for s in scores:
        by_cat[labels_by_id[s["id"]]["category"]].append(s)
    for cat in CATEGORY_ORDER:
        if by_cat[cat]:
            per_category[cat] = {"n": len(by_cat[cat]), "category": rate("category_ok", by_cat[cat]),
                                 "action": rate("action_ok", by_cat[cat]), "fully": rate("fully_ok", by_cat[cat])}

    failures = []
    records_by_id = {r["id"]: r for r in records}
    for s in scores:
        if s["fully_ok"]:
            continue
        label, rec = labels_by_id[s["id"]], records_by_id[s["id"]]
        failures.append({
            "id": s["id"],
            "expected": f"{label['category']} -> {label['action']}",
            "got": f"{rec.get('category')} -> {', '.join(rec.get('actions') or []) or 'no action'}",
            "wrong_fields": {f: {"expected": label[f], "got": rec.get(f)} for f in s["wrong_fields"]},
            "note": label.get("note", ""),
            "error": s["error"],
        })

    latencies = [r.get("latency_s") or 0 for r in records]
    total_cost = sum(r.get("cost_usd") or 0 for r in records)
    return {
        "provider": records[0].get("provider"),
        "model": records[0].get("model"),
        "items": n,
        "errors": sum(1 for s in scores if s["error"]),
        "category_accuracy": rate("category_ok"),
        "action_accuracy": rate("action_ok"),
        "field_accuracy": fields_correct / fields_expected if fields_expected else 1.0,
        "fully_correct": rate("fully_ok"),
        "avg_latency_s": sum(latencies) / n,
        "max_latency_s": max(latencies),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in records),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in records),
        "total_cost_usd": total_cost,
        "cost_per_100_items_usd": total_cost / n * 100,
        "per_category": per_category,
        "failures": failures,
    }


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def comparison_table(summaries: dict[str, dict]) -> str:
    rows = ["| Run | Model | Items | Category | Action | Fields | Fully correct | Avg sec/item | Cost per 100 items |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        rows.append(f"| {name} | {s['model']} | {s['items']} | {pct(s['category_accuracy'])} | "
                    f"{pct(s['action_accuracy'])} | {pct(s['field_accuracy'])} | **{pct(s['fully_correct'])}** | "
                    f"{s['avg_latency_s']:.2f} | ${s['cost_per_100_items_usd']:.4f} |")
    return "\n".join(rows)


def category_table(s: dict) -> str:
    rows = ["| True category | Items | Category right | Action right | Fully correct |",
            "|---|---:|---:|---:|---:|"]
    for cat, c in s["per_category"].items():
        rows.append(f"| {cat} | {c['n']} | {pct(c['category'])} | {pct(c['action'])} | {pct(c['fully'])} |")
    return "\n".join(rows)


def failure_lines(s: dict, limit: int | None) -> list[str]:
    fails = s["failures"] if limit is None else s["failures"][:limit]
    lines = []
    for f in fails:
        line = f"- {f['id']}: expected {f['expected']}, got {f['got']}"
        if f["wrong_fields"]:
            wrong = "; ".join(f"{k} expected {v['expected']!r} got {v['got']!r}" for k, v in f["wrong_fields"].items())
            line += f" | fields: {wrong}"
        if f["note"]:
            line += f" | case: {f['note']}"
        if f["error"]:
            line += f" | ERROR: {f['error'][:150]}"
        lines.append(line)
    if limit is not None and len(s["failures"]) > limit:
        lines.append(f"- ... and {len(s['failures']) - limit} more (use --all-failures to see them)")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score result files against the answer key.")
    parser.add_argument("results", nargs="+", type=Path, help="one or more results/*.jsonl files")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="where to write the markdown report")
    parser.add_argument("--all-failures", action="store_true", help="list every failed item, not just the first 15")
    args = parser.parse_args(argv)

    labels = load_jsonl(args.labels)
    paths: list[Path] = []
    for p in args.results:  # expand patterns ourselves: the Windows shell doesn't expand results/*.jsonl
        matches = sorted(glob.glob(str(p))) if any(ch in str(p) for ch in "*?[") else [str(p)]
        paths += [Path(m) for m in matches] or [p]
    summaries: dict[str, dict] = {}
    for path in dict.fromkeys(paths):  # drop duplicates, keep order
        if not path.exists():
            print(f"Skipping {path}: file not found")
            continue
        records = load_jsonl(path)
        if not records:
            print(f"Skipping {path}: file is empty")
            continue
        summaries[path.stem] = summarize(records, labels)
    if not summaries:
        print("Nothing to evaluate.")
        return 1

    limit = None if args.all_failures else 15
    report = ["# Evaluation report", "", f"Answer key: {len(labels)} labeled items.", "", "## Comparison", "",
              comparison_table(summaries), ""]
    for name, s in summaries.items():
        report += [f"## {name}", "",
                   f"Provider: {s['provider']}, model: {s['model']}, items evaluated: {s['items']}"
                   f" of {len(labels)}, errors: {s['errors']}, tokens: {s['prompt_tokens']:,} in /"
                   f" {s['completion_tokens']:,} out, total cost: ${s['total_cost_usd']:.4f}", "",
                   category_table(s), "", f"Failures ({len(s['failures'])}):", ""]
        report += failure_lines(s, limit) or ["- none"]
        report.append("")
    text = "\n".join(report)
    print(text)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    with open(args.report.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nReport saved to {args.report} (and {args.report.with_suffix('.json').name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
