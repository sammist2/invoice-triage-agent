"""End-to-end: the real runner + real API + real eval, with fake models.

The key test uses an "oracle" fake LLM that always makes the right tool calls
(it peeks at the answer key). If every part of the pipeline agrees on the
rules, that run must score exactly 100%. Any mismatch between the prompt's
tools, the API, the runner and the eval would show up here.
"""

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from agent.run import main as run_main
from evals.evaluate import summarize
from tests.helpers import ApiServer, FakeLLM, dataset_dir, load_jsonl, text_message, tool_call


def oracle_behaviour(items: list[dict], labels: dict):
    """Fake model that reads the message, finds its label, and makes the right calls."""

    def find_item(content: str) -> dict:
        # Suppliers reuse the same cover email, so the attachment name is part of the match.
        matches = [i for i in items if f"From: {i['from']}\n" in content
                   and f"Subject: {i['subject']}\n" in content and i["body"] in content
                   and all(f"Attachment: {a} " in content for a in i["attachments"])]
        assert len(matches) == 1, f"expected exactly one matching item, got {len(matches)}"
        return matches[0]

    def behaviour(body: dict):
        messages = body["messages"]
        item = find_item(messages[1]["content"])
        label = labels[item["id"]]
        tool_results = sum(1 for m in messages if m["role"] == "tool")

        steps = []
        if label["order_id"] and label["category"] in ("refund_request", "shipping_delay"):
            steps.append(("lookup_order", {"order_id": label["order_id"]}))
        if label["po_number"]:
            steps.append(("lookup_purchase_order", {"po_number": label["po_number"]}))
        action = label["action"]
        if action.startswith("create_ticket:"):
            steps.append(("create_ticket", {"ticket_type": action.split(":")[1], "customer_email": item["from"],
                                            "order_id": label["order_id"], "summary": "test"}))
        elif action == "record_invoice":
            steps.append(("record_invoice", {"invoice_number": label["invoice_number"], "po_number": label["po_number"],
                                             "amount": label["amount"], "supplier": "test"}))
        else:
            steps.append((action, {"reason": label["note"]}))
        steps.append(("submit_result", {k: label[k] for k in
                                        ("category", "order_id", "invoice_number", "po_number", "amount")}))
        name, args = steps[tool_results]
        return tool_call(name, args, f"call_{tool_results}")

    return behaviour


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        src = dataset_dir()
        for name in ("items.jsonl", "labels.jsonl", "business.db"):
            shutil.copy(src / name, self.tmp / name)
        shutil.copytree(src / "invoices", self.tmp / "invoices")
        self.items = load_jsonl(self.tmp / "items.jsonl")
        self.labels = load_jsonl(self.tmp / "labels.jsonl")
        self.api = ApiServer(self.tmp / "business.db").__enter__()
        self.env = mock.patch.dict(os.environ, {"API_URL": self.api.url, "LLM_API_KEY": "",
                                                "LLM_INPUT_COST_PER_M": "", "LLM_OUTPUT_COST_PER_M": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.api.__exit__(None, None, None)
        shutil.rmtree(self.tmp)

    def run_cli(self, *args) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            code = run_main(["--items", str(self.tmp / "items.jsonl"), *args])
        return code, out.getvalue()

    def count(self, kind: str, run_name: str) -> int:
        import requests
        return len(requests.get(f"{self.api.url}/{kind}", params={"run_name": run_name}, timeout=5).json())

    def test_oracle_model_scores_100_percent(self):
        behaviour = oracle_behaviour(self.items, {l["id"]: l for l in self.labels})
        with FakeLLM(behaviour) as fake:
            code, output = self.run_cli("--provider", "custom", "--model", "oracle", "--base-url", fake.base_url,
                                        "--name", "oracle", "--out", str(self.tmp / "oracle.jsonl"))
        self.assertEqual(code, 0, output)
        records = load_jsonl(self.tmp / "oracle.jsonl")
        self.assertEqual(len(records), 105)
        s = summarize(records, self.labels)
        self.assertEqual(s["errors"], 0)
        self.assertEqual(s["fully_correct"], 1.0, json.dumps(s["failures"][:3], indent=1))
        self.assertEqual(self.count("tickets", "oracle"), 19 + 16 + 19)
        self.assertEqual(self.count("invoices", "oracle"), 17)
        self.assertEqual(self.count("flags", "oracle"), 19)
        self.assertEqual(self.count("spam", "oracle"), 15)

    def test_baseline_runs_and_is_imperfect(self):
        code, output = self.run_cli("--provider", "baseline", "--out", str(self.tmp / "baseline.jsonl"))
        self.assertEqual(code, 0, output)
        s = summarize(load_jsonl(self.tmp / "baseline.jsonl"), self.labels)
        self.assertEqual(s["items"], 105)
        self.assertEqual(s["errors"], 0)
        self.assertTrue(0.5 < s["fully_correct"] < 0.95, s["fully_correct"])

    def test_rerunning_a_run_name_starts_clean(self):
        for _ in range(2):
            code, _ = self.run_cli("--provider", "baseline", "--name", "again", "--out", str(self.tmp / "a.jsonl"))
            self.assertEqual(code, 0)
        first_run_tickets = sum(1 for r in load_jsonl(self.tmp / "a.jsonl") for a in r["actions"]
                                if a.startswith("create_ticket"))
        self.assertEqual(self.count("tickets", "again"), first_run_tickets)

    def test_resume_keeps_finished_items_and_redoes_errors(self):
        labels = {l["id"]: l for l in self.labels}
        oracle = oracle_behaviour(self.items, labels)
        out = self.tmp / "resume.jsonl"
        common = ["--provider", "custom", "--model", "oracle", "--name", "resume", "--out", str(out)]

        broken_id = self.items[1]["id"]

        def flaky(body):  # the model server fails on one item during the first run
            if f"Subject: {self.items[1]['subject']}\n" in body["messages"][1]["content"] \
                    and self.items[1]["body"] in body["messages"][1]["content"]:
                return 400, {"error": "temporary problem"}
            return oracle(body)

        with FakeLLM(flaky) as fake:
            code, _ = self.run_cli(*common, "--base-url", fake.base_url, "--limit", "4")
        self.assertEqual(code, 0)
        first = load_jsonl(out)
        self.assertEqual(len(first), 4)
        self.assertEqual([r["id"] for r in first if r["error"]], [broken_id])

        with FakeLLM(oracle) as fake:
            code, output = self.run_cli(*common, "--base-url", fake.base_url, "--limit", "8", "--resume")
            requests_made = len(fake.requests)
        self.assertEqual(code, 0)
        self.assertIn("3 items already done", output)
        records = load_jsonl(out)
        self.assertEqual(sorted(r["id"] for r in records), sorted(i["id"] for i in self.items[:8]))
        self.assertTrue(all(r["error"] is None for r in records))
        self.assertGreater(requests_made, 0)
        s = summarize(records, self.labels)
        self.assertEqual(s["fully_correct"], 1.0)

        # Resuming a finished run does nothing and doesn't lose results.
        with FakeLLM(oracle) as fake:
            code, output = self.run_cli(*common, "--base-url", fake.base_url, "--limit", "8", "--resume")
            self.assertEqual(len(fake.requests), 0)
        self.assertEqual(len(load_jsonl(out)), 8)

    def test_stops_when_model_server_is_down_then_resumes(self):
        out = self.tmp / "down.jsonl"
        common = ["--provider", "custom", "--model", "m", "--name", "down", "--out", str(out), "--limit", "10"]
        code, output = self.run_cli(*common, "--base-url", "http://127.0.0.1:1/v1")
        self.assertEqual(code, 1)
        self.assertIn("failed 3 times in a row", output)
        self.assertIn("--resume", output)
        self.assertEqual(len(load_jsonl(out)), 3)

        oracle = oracle_behaviour(self.items, {l["id"]: l for l in self.labels})
        with FakeLLM(oracle) as fake:
            code, output = self.run_cli(*common, "--base-url", fake.base_url, "--resume")
        self.assertEqual(code, 0)
        self.assertIn("0 items already done", output)
        records = load_jsonl(out)
        self.assertEqual(len(records), 10)
        self.assertEqual(summarize(records, self.labels)["fully_correct"], 1.0)

    def test_limit_and_ids(self):
        code, _ = self.run_cli("--provider", "baseline", "--limit", "5", "--out", str(self.tmp / "l.jsonl"))
        self.assertEqual(code, 0)
        self.assertEqual(len(load_jsonl(self.tmp / "l.jsonl")), 5)
        code, _ = self.run_cli("--provider", "baseline", "--ids", "item-007, item-003",
                               "--out", str(self.tmp / "i.jsonl"))
        self.assertEqual(code, 0)
        self.assertEqual([r["id"] for r in load_jsonl(self.tmp / "i.jsonl")], ["item-003", "item-007"])
        code, output = self.run_cli("--provider", "baseline", "--ids", "item-999")
        self.assertEqual(code, 1)
        self.assertIn("No items selected", output)

    def test_llm_errors_are_saved_not_fatal(self):
        with FakeLLM(lambda body: text_message("no idea")) as fake:
            code, output = self.run_cli("--provider", "custom", "--model", "dumb", "--base-url", fake.base_url,
                                        "--limit", "3", "--out", str(self.tmp / "dumb.jsonl"))
        self.assertEqual(code, 0)
        self.assertIn("3 with errors", output)
        s = summarize(load_jsonl(self.tmp / "dumb.jsonl"), self.labels)
        self.assertEqual((s["items"], s["errors"], s["fully_correct"]), (3, 3, 0.0))

    def test_helpful_messages(self):
        with mock.patch.dict(os.environ, {"API_URL": "http://127.0.0.1:1"}):
            code, output = self.run_cli("--provider", "baseline")
        self.assertEqual(code, 1)
        self.assertIn("python -m api.app", output)

        code, output = self.run_cli("--provider", "baseline", "--name", "bad name!")
        self.assertEqual(code, 1)

        code, output = run_main_capture(["--provider", "baseline", "--items", str(self.tmp / "nope.jsonl")])
        self.assertEqual(code, 1)
        self.assertIn("python -m data.generate_data", output)

        for argv, expected in [(["--provider", "nope"], "Unknown provider"),
                               (["--provider", "gemini", "--model", "x"], "needs an API key"),
                               (["--provider", "groq"], "needs a model name"),
                               (["--provider", "custom", "--model", "x"], "needs a base URL")]:
            with mock.patch.dict(os.environ, {"LLM_MODEL": "", "LLM_BASE_URL": ""}):
                with self.assertRaises(SystemExit) as ctx:
                    run_main(argv)
            self.assertIn(expected, str(ctx.exception.code))


def run_main_capture(argv):
    out = io.StringIO()
    with redirect_stdout(out):
        code = run_main(argv)
    return code, out.getvalue()


if __name__ == "__main__":
    unittest.main()
