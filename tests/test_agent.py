"""Tests for the agent loop, using a fake LLM server so no real model is needed."""

import json
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.agent import NUDGE, NUDGE_SUBMIT, process_item
from agent.config import Settings
from agent.llm import LLMClient
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOL_SCHEMAS, ApiClient, ApiUnavailable
from tests.helpers import ApiServer, FakeLLM, dataset_dir, load_jsonl, text_message, tool_call


def scripted(replies: list):
    """A fake-LLM behaviour that returns the given replies in order."""
    replies = iter(replies)
    return lambda body: next(replies)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class AgentLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        shutil.copy(dataset_dir() / "business.db", cls.tmp / "business.db")
        cls.api_server = ApiServer(cls.tmp / "business.db").__enter__()
        cls.api = ApiClient(cls.api_server.url)
        items = load_jsonl(dataset_dir() / "items.jsonl")
        labels = load_jsonl(dataset_dir() / "labels.jsonl")
        cls.items = {i["id"]: i for i in items}
        cls.refund_label = next(l for l in labels if l["action"] == "create_ticket:refund")
        cls.refund_item = cls.items[cls.refund_label["id"]]
        cls.invoice_label = next(l for l in labels if l["action"] == "record_invoice")
        cls.invoice_item = cls.items[cls.invoice_label["id"]]

    @classmethod
    def tearDownClass(cls):
        cls.api_server.__exit__(None, None, None)
        shutil.rmtree(cls.tmp)

    def settings(self, base_url: str) -> Settings:
        return Settings(provider="custom", model="fake-model", base_url=base_url, api_key="test-key",
                        api_url=self.api_server.url, input_cost_per_m=1.0, output_cost_per_m=2.0)

    def run_agent(self, behaviour, item=None, max_steps=10):
        item = item or self.refund_item
        with FakeLLM(behaviour) as fake:
            settings = self.settings(fake.base_url)
            record = process_item(item, LLMClient(settings), self.api, settings, self._testMethodName,
                                  dataset_dir(), max_steps=max_steps)
        return record, fake.requests

    def test_full_refund_flow(self):
        label = self.refund_label
        script = [
            tool_call("lookup_order", {"order_id": label["order_id"]}, "call_a"),
            tool_call("create_ticket", {"ticket_type": "refund", "customer_email": self.refund_item["from"],
                                        "order_id": label["order_id"], "summary": "wants a refund"}, "call_b"),
            tool_call("submit_result", {"category": "refund_request", "order_id": label["order_id"][4:]}, "call_c"),
        ]
        record, reqs = self.run_agent(scripted(script))
        self.assertIsNone(record["error"])
        self.assertEqual(record["actions"], ["create_ticket:refund"])
        self.assertEqual(record["category"], "refund_request")
        self.assertEqual(record["order_id"], label["order_id"])  # normalized from bare digits
        self.assertEqual(record["steps"], 3)
        self.assertEqual((record["prompt_tokens"], record["completion_tokens"]), (300, 60))
        self.assertAlmostEqual(record["cost_usd"], (300 * 1.0 + 60 * 2.0) / 1e6)
        self.assertEqual([c["name"] for c in record["tool_calls"]], ["lookup_order", "create_ticket", "submit_result"])

        first = reqs[0]
        self.assertEqual(first["path"], "/v1/chat/completions")
        self.assertEqual(first["auth"], "Bearer test-key")
        self.assertEqual(first["body"]["model"], "fake-model")
        self.assertEqual(first["body"]["temperature"], 0)
        self.assertEqual(first["body"]["tools"], TOOL_SCHEMAS)
        self.assertEqual(first["body"]["messages"][0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertIn(self.refund_item["body"], first["body"]["messages"][1]["content"])

        second = reqs[1]["body"]["messages"]
        self.assertEqual(second[2]["role"], "assistant")
        self.assertEqual(second[2]["tool_calls"][0]["id"], "call_a")
        self.assertEqual(second[3]["role"], "tool")
        self.assertEqual(second[3]["tool_call_id"], "call_a")
        self.assertEqual(json.loads(second[3]["content"])["customer_email"], self.refund_item["from"])

        tickets = self.api.request("GET", "/tickets", params={"run_name": self._testMethodName})[1]
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets[0]["item_id"], self.refund_item["id"])

    def test_invoice_pdf_text_reaches_the_model(self):
        _, reqs = self.run_agent(lambda body: tool_call("submit_result", {"category": "invoice_received"}),
                                 item=self.invoice_item)
        user_msg = reqs[0]["body"]["messages"][1]["content"]
        self.assertIn("text extracted from PDF", user_msg)
        self.assertIn(self.invoice_label["invoice_number"], user_msg)
        self.assertIn(f"${self.invoice_label['amount']:,.2f}", user_msg)

    def test_failed_action_is_not_counted(self):
        script = [
            tool_call("record_invoice", {"invoice_number": "X-1", "po_number": "PO-9999", "amount": 10,
                                         "supplier": "s"}),
            tool_call("flag_for_human", {"reason": "PO does not exist"}),
            tool_call("submit_result", {"category": "invoice_received", "po_number": "PO-9999"}),
        ]
        record, _ = self.run_agent(scripted(script), item=self.invoice_item)
        self.assertEqual(record["actions"], ["flag_for_human"])
        self.assertEqual(record["tool_calls"][0]["status"], 404)

    def test_bad_json_arguments_are_reported_back(self):
        bad = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "lookup_order", "arguments": "{bad json"}}]}
        script = [bad, tool_call("submit_result", {"category": "refund_request"})]
        record, reqs = self.run_agent(scripted(script))
        self.assertIsNone(record["error"])
        self.assertEqual(record["tool_calls"][0]["response"]["error"], "invalid_json_arguments")
        self.assertIn("invalid_json_arguments", reqs[1]["body"]["messages"][3]["content"])

    def test_unknown_tool(self):
        script = [tool_call("delete_database", {}), tool_call("submit_result", {"category": "spam"})]
        record, _ = self.run_agent(scripted(script))
        self.assertEqual(record["tool_calls"][0]["response"]["error"], "unknown_tool")
        self.assertEqual(record["category"], "spam")

    def test_invalid_result_lets_model_retry(self):
        script = [tool_call("submit_result", {"category": "complaint"}),
                  tool_call("submit_result", {"category": "refund_request", "amount": "null"})]
        record, _ = self.run_agent(scripted(script))
        self.assertEqual(record["tool_calls"][0]["response"]["error"], "invalid_result")
        self.assertEqual(record["category"], "refund_request")
        self.assertIsNone(record["amount"])
        self.assertEqual(record["steps"], 2)

    def test_dict_arguments_and_missing_ids_are_handled(self):
        """Some servers send arguments as an object and leave out the call id."""
        msg = {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function", "function": {"name": "submit_result", "arguments": {"category": "spam"}}}]}
        record, _ = self.run_agent(lambda body: msg)
        self.assertEqual(record["category"], "spam")
        self.assertIsNone(record["error"])

    def test_json_in_plain_text_is_accepted(self):
        record, reqs = self.run_agent(lambda body: text_message('Result: {"category": "spam", "order_id": null}'))
        self.assertEqual(record["category"], "spam")
        self.assertEqual(len(reqs), 1)

    def test_text_only_replies_get_one_nudge_then_error(self):
        record, reqs = self.run_agent(lambda body: text_message("I think this is a refund."))
        self.assertEqual(len(reqs), 2)
        self.assertEqual(reqs[1]["body"]["messages"][-1], {"role": "user", "content": NUDGE})
        self.assertIn("text instead of calling tools", record["error"])
        self.assertIsNone(record["category"])

    def test_nudge_after_action_asks_only_for_the_result(self):
        script = [tool_call("mark_spam", {"reason": "scam"}),
                  text_message("Done, I marked it as spam."),
                  tool_call("submit_result", {"category": "spam"})]
        record, reqs = self.run_agent(scripted(script))
        self.assertEqual(reqs[2]["body"]["messages"][-1], {"role": "user", "content": NUDGE_SUBMIT})
        self.assertEqual(record["actions"], ["mark_spam"])
        self.assertEqual(record["category"], "spam")
        self.assertIsNone(record["error"])

    def test_step_limit(self):
        record, reqs = self.run_agent(lambda body: tool_call("lookup_customer", {"email": "a@b.c"}), max_steps=3)
        self.assertEqual(len(reqs), 3)
        self.assertIn("stopped after 3 steps", record["error"])

    def test_rate_limit_is_retried(self):
        calls = {"n": 0}

        def behaviour(body):
            calls["n"] += 1
            if calls["n"] == 1:
                return 429, {"error": {"message": "slow down"}}
            return tool_call("submit_result", {"category": "spam"})

        with mock.patch("agent.llm.time.sleep") as sleep:
            record, reqs = self.run_agent(behaviour)
        self.assertEqual(len(reqs), 2)
        self.assertEqual(record["category"], "spam")
        sleep.assert_called_once()

    def test_server_error_becomes_record_error(self):
        with mock.patch("agent.llm.time.sleep"):
            record, reqs = self.run_agent(lambda body: (500, {"error": "boom"}))
        self.assertEqual(len(reqs), 4)  # 1 try + 3 retries
        self.assertIn("HTTP 500", record["error"])

    def test_bad_request_is_not_retried(self):
        record, reqs = self.run_agent(lambda body: (400, {"error": "model not found"}))
        self.assertEqual(len(reqs), 1)
        self.assertIn("HTTP 400", record["error"])
        self.assertIn("model not found", record["error"])

    def test_model_server_down(self):
        settings = self.settings(f"http://127.0.0.1:{free_port()}/v1")
        record = process_item(self.refund_item, LLMClient(settings), self.api, settings, "down", dataset_dir())
        self.assertIn("Could not reach the model server", record["error"])

    def test_business_api_down_raises(self):
        dead_api = ApiClient(f"http://127.0.0.1:{free_port()}")
        self.assertFalse(dead_api.health())
        with FakeLLM(lambda body: tool_call("lookup_order", {"order_id": "ORD-10001"})) as fake:
            settings = self.settings(fake.base_url)
            with self.assertRaises(ApiUnavailable):
                process_item(self.refund_item, LLMClient(settings), dead_api, settings, "x", dataset_dir())


if __name__ == "__main__":
    unittest.main()
