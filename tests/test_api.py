import shutil
import tempfile
import unittest
from pathlib import Path

from api.app import create_app
from tests.helpers import dataset_dir


class ApiTests(unittest.TestCase):
    def setUp(self):
        # Each test gets its own copy of the database so tests don't affect each other.
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copy(dataset_dir() / "business.db", self.tmp / "business.db")
        self.client = create_app(self.tmp / "business.db").test_client()
        con = __import__("sqlite3").connect(self.tmp / "business.db")
        self.order = con.execute(
            "SELECT o.id, c.email FROM orders o JOIN customers c ON c.id = o.customer_id LIMIT 1").fetchone()
        self.po = con.execute("SELECT po_number, amount FROM purchase_orders LIMIT 1").fetchone()
        con.close()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def ticket(self, **overrides):
        payload = {"run_name": "t", "item_id": "item-001", "ticket_type": "refund",
                   "customer_email": self.order[1], "order_id": self.order[0], "summary": "broken"}
        payload.update(overrides)
        return self.client.post("/tickets", json=payload)

    def invoice(self, **overrides):
        payload = {"run_name": "t", "item_id": "item-002", "invoice_number": "INV-1", "po_number": self.po[0],
                   "amount": self.po[1], "supplier": "Acme"}
        payload.update(overrides)
        return self.client.post("/invoices", json=payload)

    def test_missing_database_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            create_app(self.tmp / "nope.db")

    def test_health(self):
        self.assertEqual(self.client.get("/health").json, {"status": "ok"})

    def test_order_lookup_accepts_many_formats(self):
        digits = self.order[0][4:]
        for ref in [self.order[0], digits, f"ord-{digits}", f"%23{digits}"]:
            r = self.client.get(f"/orders/{ref}")
            self.assertEqual(r.status_code, 200, ref)
            self.assertEqual(r.json["id"], self.order[0])
            self.assertEqual(r.json["customer_email"], self.order[1])
        r = self.client.get("/orders/ORD-99999")
        self.assertEqual((r.status_code, r.json["error"]), (404, "order_not_found"))

    def test_customer_lookup(self):
        r = self.client.get(f"/customers?email={self.order[1].upper()}")
        self.assertEqual(r.status_code, 200)
        self.assertIn(self.order[0], [o["id"] for o in r.json["orders"]])
        self.assertEqual(self.client.get("/customers?email=nobody@example.com").status_code, 404)
        self.assertEqual(self.client.get("/customers").status_code, 400)

    def test_purchase_order_lookup(self):
        r = self.client.get(f"/purchase-orders/{self.po[0][3:]}")
        self.assertEqual((r.status_code, r.json["amount"]), (200, self.po[1]))
        self.assertEqual(self.client.get("/purchase-orders/PO-9999").status_code, 404)

    def test_create_ticket(self):
        r = self.ticket(order_id=f"#{self.order[0][4:]}", ticket_type="Refund")
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json["order_id"], self.order[0])
        self.assertEqual(r.json["ticket_type"], "refund")
        self.assertEqual(self.ticket(order_id=None, ticket_type="billing").status_code, 201)

    def test_ticket_validation(self):
        self.assertEqual(self.ticket(ticket_type="complaint").json["error"], "invalid_ticket_type")
        self.assertEqual(self.ticket(summary="").json["error"], "missing_fields")
        self.assertEqual(self.ticket(customer_email="x@y.example").status_code, 404)
        self.assertEqual(self.ticket(order_id="ORD-99999").json["error"], "order_not_found")
        self.assertEqual(self.client.post("/tickets", data="not json").status_code, 400)

    def test_record_invoice_rules(self):
        self.assertEqual(self.invoice(amount=f"${self.po[1]:,.2f}").status_code, 201)
        self.assertEqual(self.invoice().status_code, 409)  # same invoice twice in one run
        self.assertEqual(self.invoice(run_name="other").status_code, 201)  # fine in another run
        r = self.invoice(invoice_number="INV-2", amount=self.po[1] + 0.01)
        self.assertEqual((r.status_code, r.json["error"]), (422, "amount_mismatch"))
        self.assertEqual(self.invoice(invoice_number="INV-3", po_number="PO-9999").status_code, 404)
        self.assertEqual(self.invoice(invoice_number="INV-4", amount="lots").json["error"], "invalid_amount")
        self.assertEqual(self.invoice(supplier=None).status_code, 400)

    def test_flags_spam_listing_and_reset(self):
        self.assertEqual(self.client.post("/flags", json={"run_name": "t", "item_id": "a", "reason": "r"}).status_code, 201)
        self.assertEqual(self.client.post("/flags", json={"run_name": "t", "item_id": "a"}).status_code, 400)
        self.assertEqual(self.client.post("/spam", json={"run_name": "t", "item_id": "b"}).status_code, 201)
        self.ticket()
        self.invoice()
        self.ticket(run_name="keep")
        for kind in ("tickets", "invoices", "flags", "spam"):
            self.assertEqual(len(self.client.get(f"/{kind}?run_name=t").json), 1, kind)
        r = self.client.delete("/runs/t")
        self.assertEqual(r.json["deleted"], {"tickets": 1, "invoices": 1, "flags": 1, "spam": 1})
        self.assertEqual(self.client.get("/tickets?run_name=t").json, [])
        self.assertEqual(len(self.client.get("/tickets").json), 1)  # other run untouched

    def test_unknown_routes_return_json(self):
        r = self.client.get("/nope")
        self.assertEqual((r.status_code, r.json["error"]), (404, "not_found"))
        self.assertEqual(self.client.put("/tickets").status_code, 405)

    def test_regenerating_data_while_running_still_works(self):
        """The API creates its action tables on every connection, not just at startup."""
        import sqlite3
        self.assertEqual(self.ticket().status_code, 201)  # tables exist now
        con = sqlite3.connect(self.tmp / "business.db")
        con.executescript("DROP TABLE tickets; DROP TABLE flags;")  # like a fresh database
        con.close()
        self.assertEqual(self.ticket().status_code, 201)
        self.assertEqual(len(self.client.get("/tickets").json), 1)


if __name__ == "__main__":
    unittest.main()
