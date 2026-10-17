import json
import tempfile
import unittest
from pathlib import Path

from evals.evaluate import main, score_item, summarize

LABELS = [
    {"id": "a", "category": "refund_request", "action": "create_ticket:refund", "order_id": "ORD-10001",
     "invoice_number": None, "po_number": None, "amount": None, "note": "valid"},
    {"id": "b", "category": "invoice_received", "action": "record_invoice", "order_id": None,
     "invoice_number": "GRF-1", "po_number": "PO-5001", "amount": 100.5, "note": "match"},
    {"id": "c", "category": "spam", "action": "mark_spam", "order_id": None,
     "invoice_number": None, "po_number": None, "amount": None, "note": "spam"},
    {"id": "d", "category": "shipping_delay", "action": "flag_for_human", "order_id": None,
     "invoice_number": None, "po_number": None, "amount": None, "note": "no order"},
]


def perfect(label: dict) -> dict:
    return {"id": label["id"], "provider": "test", "model": "m", "category": label["category"],
            "order_id": label["order_id"], "invoice_number": label["invoice_number"],
            "po_number": label["po_number"], "amount": label["amount"], "actions": [label["action"]],
            "error": None, "latency_s": 1.0, "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.01}


class EvalTests(unittest.TestCase):
    def test_perfect_run_scores_100(self):
        s = summarize([perfect(l) for l in LABELS], LABELS)
        for key in ("category_accuracy", "action_accuracy", "field_accuracy", "fully_correct"):
            self.assertEqual(s[key], 1.0, key)
        self.assertEqual(s["failures"], [])
        self.assertAlmostEqual(s["cost_per_100_items_usd"], 1.0)
        self.assertEqual(s["prompt_tokens"], 40)

    def test_field_formats_are_forgiven(self):
        rec = perfect(LABELS[0]) | {"order_id": "#10001"}
        self.assertTrue(score_item(LABELS[0], rec)["fully_ok"])
        rec = perfect(LABELS[1]) | {"po_number": "5001", "amount": "$100.50", "invoice_number": "grf-1"}
        self.assertTrue(score_item(LABELS[1], rec)["fully_ok"])

    def test_wrong_things_are_caught(self):
        cases = {
            "wrong category": (LABELS[0], {"category": "billing_question"}),
            "wrong ticket type": (LABELS[0], {"actions": ["create_ticket:billing"]}),
            "two actions": (LABELS[3], {"actions": ["flag_for_human", "flag_for_human"]}),
            "no action": (LABELS[2], {"actions": []}),
            "wrong amount": (LABELS[1], {"amount": 100.51}),
            "missing field": (LABELS[1], {"po_number": None}),
            "wrong order": (LABELS[0], {"order_id": "ORD-10002"}),
        }
        for name, (label, change) in cases.items():
            with self.subTest(name):
                self.assertFalse(score_item(label, perfect(label) | change)["fully_ok"])

    def test_extra_fields_on_flagged_items_are_not_penalised(self):
        """If the label has no order number, whatever the model says there is ignored."""
        rec = perfect(LABELS[3]) | {"order_id": "ORD-12345"}
        self.assertTrue(score_item(LABELS[3], rec)["fully_ok"])

    def test_partial_run_and_metrics(self):
        recs = [perfect(LABELS[0]), perfect(LABELS[1]) | {"amount": 1.0}]
        s = summarize(recs, LABELS)
        self.assertEqual(s["items"], 2)
        self.assertEqual(s["category_accuracy"], 1.0)
        self.assertEqual(s["fully_correct"], 0.5)
        self.assertAlmostEqual(s["field_accuracy"], 3 / 4)  # order_id ok, invoice ok, po ok, amount wrong
        self.assertEqual(len(s["failures"]), 1)
        self.assertEqual(s["failures"][0]["wrong_fields"], {"amount": {"expected": 100.5, "got": 1.0}})

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "labels.jsonl").write_text("".join(json.dumps(l) + "\n" for l in LABELS))
            (d / "run-a.jsonl").write_text("".join(json.dumps(perfect(l)) + "\n" for l in LABELS))
            (d / "run-b.jsonl").write_text(json.dumps(perfect(LABELS[0]) | {"category": "spam"}) + "\n")
            code = main([str(d / "run-a.jsonl"), str(d / "run-b.jsonl"), str(d / "missing.jsonl"),
                         "--labels", str(d / "labels.jsonl"), "--report", str(d / "report.md")])
            self.assertEqual(code, 0)
            report = (d / "report.md").read_text()
            self.assertIn("| run-a | m | 4 | 100.0% | 100.0% | 100.0% | **100.0%**", report)
            self.assertIn("| run-b | m | 1 | 0.0%", report)
            summary = json.loads((d / "report.json").read_text())
            self.assertEqual(set(summary), {"run-a", "run-b"})


if __name__ == "__main__":
    unittest.main()
