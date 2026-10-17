"""Checks that the generated data and the answer key are correct and consistent.

The most important test here is test_every_label_is_correct: it re-checks each
label independently, using the API and the PDF text, so a bug in the generator
can't silently produce wrong answers.
"""

import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from agent.documents import extract_pdf_text
from api.app import create_app
from data.generate_data import SUPPLIERS, generate
from tests.helpers import dataset_dir, load_jsonl

MONEY = r"\$([\d,]+\.\d{2})"


def money_to_float(s: str) -> float:
    return float(s.replace(",", ""))


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = dataset_dir()
        cls.items = load_jsonl(cls.dir / "items.jsonl")
        cls.labels = load_jsonl(cls.dir / "labels.jsonl")
        cls.items_by_id = {i["id"]: i for i in cls.items}
        cls.client = create_app(cls.dir / "business.db").test_client()

    def test_counts(self):
        self.assertEqual(len(self.items), 105)
        self.assertEqual([i["id"] for i in self.items], [l["id"] for l in self.labels])
        self.assertEqual(len({i["id"] for i in self.items}), 105)
        self.assertEqual(Counter(l["category"] for l in self.labels), {
            "refund_request": 25, "invoice_received": 25, "shipping_delay": 21, "billing_question": 19, "spam": 15})
        self.assertEqual(Counter(l["action"] for l in self.labels), {
            "create_ticket:refund": 19, "flag_for_human": 19, "create_ticket:billing": 19,
            "record_invoice": 17, "create_ticket:shipping": 16, "mark_spam": 15})

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate(Path(a))
            generate(Path(b))
            for name in ("items.jsonl", "labels.jsonl"):
                self.assertEqual((Path(a) / name).read_text(), (Path(b) / name).read_text(), name)
            pdfs = sorted(p.name for p in (Path(a) / "invoices").glob("*.pdf"))
            self.assertEqual(len(pdfs), 25)
            for name in pdfs:
                self.assertEqual(extract_pdf_text(str(Path(a) / "invoices" / name)),
                                 extract_pdf_text(str(Path(b) / "invoices" / name)), name)

    def test_invoice_arithmetic(self):
        """Line items add up to the subtotal, and subtotal + GST = total, on every PDF."""
        for pdf in sorted((self.dir / "invoices").glob("*.pdf")):
            text = extract_pdf_text(str(pdf))
            lines = text.splitlines()
            start = next(i for i, l in enumerate(lines) if l.startswith("Description"))
            end = next(i for i, l in enumerate(lines) if l.startswith("Subtotal"))
            line_total = 0.0
            for row in lines[start + 1:end]:
                qty, unit, amount = re.search(r"\s(\d+)\s" + MONEY + r"\s" + MONEY + r"$", row).groups()
                self.assertAlmostEqual(int(qty) * money_to_float(unit), money_to_float(amount), places=2, msg=row)
                line_total += money_to_float(amount)
            subtotal = money_to_float(re.search(r"^Subtotal " + MONEY, text, re.M).group(1))
            gst = money_to_float(re.search(r"^GST \(5%\) " + MONEY, text, re.M).group(1))
            total = money_to_float(re.findall(MONEY, lines[end + 2])[-1])
            self.assertAlmostEqual(line_total, subtotal, places=2, msg=pdf.name)
            self.assertAlmostEqual(subtotal + gst, total, places=2, msg=pdf.name)
            self.assertAlmostEqual(gst, subtotal * 0.05, delta=0.01, msg=pdf.name)

    def test_every_label_is_correct(self):
        supplier_domains = {s["domain"] for s in SUPPLIERS}
        for label in self.labels:
            item = self.items_by_id[label["id"]]
            with self.subTest(item=label["id"], note=label["note"]):
                self._check_label(item, label, supplier_domains)

    def _check_label(self, item, label, supplier_domains):
        c = self.client
        text = f"{item['subject']}\n{item['body']}"
        sender = item["from"]
        cat, action = label["category"], label["action"]
        post = lambda path, payload: c.post(path, json={"run_name": "label-check", "item_id": item["id"], **payload})

        if cat == "spam":
            self.assertEqual(action, "mark_spam")
            self.assertEqual(c.get(f"/customers?email={sender}").status_code, 404, "spam sender is a customer")
            self.assertNotIn(sender.split("@")[1], supplier_domains)
            self.assertEqual(item["attachments"], [])
            return

        if cat == "invoice_received":
            self.assertEqual(len(item["attachments"]), 1)
            self.assertIn(sender.split("@")[1], supplier_domains)
            pdf = extract_pdf_text(str(self.dir / item["attachments"][0]))
            self.assertIn(label["invoice_number"], pdf)
            self.assertIn(label["po_number"].split("-")[1], pdf)
            self.assertIn(f"${label['amount']:,.2f}", pdf)
            self.assertIsNone(label["order_id"])
            payload = {"invoice_number": label["invoice_number"], "po_number": label["po_number"],
                       "amount": label["amount"], "supplier": "test"}
            status = post("/invoices", payload).status_code
            if action == "record_invoice":
                self.assertEqual(status, 201)
            else:
                self.assertEqual(action, "flag_for_human")
                self.assertIn(status, (404, 422))
                self.assertEqual(status == 404, "does not exist" in label["note"])
            return

        # Customer messages
        self.assertEqual(item["attachments"], [])
        self.assertIsNone(label["invoice_number"])
        self.assertIsNone(label["po_number"])
        self.assertIsNone(label["amount"])
        digits_in_text = re.findall(r"\b\d{5}\b", text)
        if label["order_id"]:
            self.assertIn(label["order_id"][4:], digits_in_text, "order number not in message")
        else:
            self.assertEqual(digits_in_text, [], "message has an order number but label says none")

        if cat == "billing_question":
            self.assertEqual(action, "create_ticket:billing")
            self.assertEqual(post("/tickets", {"ticket_type": "billing", "customer_email": sender,
                                               "order_id": label["order_id"], "summary": "x"}).status_code, 201)
            return

        expected_type = {"refund_request": "refund", "shipping_delay": "shipping"}[cat]
        order = c.get(f"/orders/{label['order_id']}") if label["order_id"] else None
        reasons = []
        if not label["order_id"]:
            reasons.append("no order")
        elif order.status_code == 404:
            reasons.append("not found")
        elif order.json["customer_email"].lower() != sender.lower():
            reasons.append("different customer")

        if action == f"create_ticket:{expected_type}":
            self.assertEqual(reasons, [])
            self.assertEqual(post("/tickets", {"ticket_type": expected_type, "customer_email": sender,
                                               "order_id": label["order_id"], "summary": "x"}).status_code, 201)
            if cat == "refund_request":
                self.assertEqual(order.json["status"], "delivered")
            else:
                self.assertIn(order.json["status"], ("shipped", "processing"))
        else:
            self.assertEqual(action, "flag_for_human")
            self.assertEqual(len(reasons), 1, "a flagged item should break exactly one rule")


if __name__ == "__main__":
    unittest.main()
