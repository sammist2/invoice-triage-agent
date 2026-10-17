"""Mock business REST API for Cedar Lane Home Goods.

This stands in for the real systems a business would have (order database,
ticketing system, accounts payable). The agent talks to it over HTTP, exactly
like it would talk to a real company's internal API.

Read endpoints:
  GET  /health
  GET  /customers?email=...            customer + their orders
  GET  /orders/<order_id>              accepts 'ORD-10023', '10023', '#10023'
  GET  /purchase-orders/<po_number>

Action endpoints (every action is tagged with run_name and item_id):
  POST /tickets    {run_name, item_id, ticket_type, customer_email, order_id?, summary}
  POST /invoices   {run_name, item_id, invoice_number, po_number, amount, supplier}
  POST /flags      {run_name, item_id, reason}
  POST /spam       {run_name, item_id, reason?}

Listing / housekeeping (used by the dashboard and the runner):
  GET    /tickets?run_name=...   (also /invoices, /flags, /spam)
  DELETE /runs/<run_name>        clear all actions from one run

Run:  python -m api.app            (serves on http://127.0.0.1:8001)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request

from shared.normalize import amounts_equal, normalize_invoice_number, normalize_order_id, normalize_po_number, parse_amount

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_DIR / "data" / "business.db"
DEFAULT_PORT = 8001  # not 5000: macOS uses port 5000 for AirPlay
TICKET_TYPES = {"refund", "billing", "shipping"}

ACTION_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_name TEXT NOT NULL, item_id TEXT NOT NULL,
    ticket_type TEXT NOT NULL, customer_email TEXT NOT NULL, order_id TEXT, summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recorded_invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_name TEXT NOT NULL, item_id TEXT NOT NULL,
    invoice_number TEXT NOT NULL, po_number TEXT NOT NULL, supplier TEXT NOT NULL, amount REAL NOT NULL,
    created_at TEXT NOT NULL, UNIQUE (run_name, invoice_number)
);
CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_name TEXT NOT NULL, item_id TEXT NOT NULL,
    reason TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spam_marks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_name TEXT NOT NULL, item_id TEXT NOT NULL,
    reason TEXT, created_at TEXT NOT NULL
);
"""

LIST_TABLES = {"tickets": "tickets", "invoices": "recorded_invoices", "flags": "flags", "spam": "spam_marks"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def error(code: int, message: str, **extra):
    return jsonify({"error": message, **extra}), code


def create_app(db_path: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    path = Path(db_path or os.environ.get("DB_PATH") or DEFAULT_DB)
    if not path.exists():
        raise FileNotFoundError(f"Database not found at {path}. Run: python -m data.generate_data")

    @contextmanager
    def db():
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        try:
            # Cheap no-op when the tables exist. Doing it on every connection means
            # regenerating the data while the API is running still works.
            con.executescript(ACTION_TABLES_SQL)
            yield con
            con.commit()
        finally:
            con.close()

    def body() -> dict:
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}

    def required(data: dict, *fields: str) -> str | None:
        missing = [f for f in fields if data.get(f) in (None, "")]
        return ", ".join(missing) if missing else None

    # ---------------- read endpoints ----------------

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/customers")
    def get_customer():
        email = (request.args.get("email") or "").strip().lower()
        if not email:
            return error(400, "missing_parameter", detail="email is required")
        with db() as con:
            row = con.execute("SELECT * FROM customers WHERE lower(email) = ?", (email,)).fetchone()
            if row is None:
                return error(404, "customer_not_found", email=email)
            orders = con.execute(
                "SELECT id, product, status, order_date FROM orders WHERE customer_id = ? ORDER BY order_date",
                (row["id"],)).fetchall()
        return jsonify({**dict(row), "orders": [dict(o) for o in orders]})

    @app.get("/orders/<path:order_id>")
    def get_order(order_id: str):
        oid = normalize_order_id(order_id)
        with db() as con:
            row = con.execute(
                """SELECT o.*, c.name AS customer_name, c.email AS customer_email
                   FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = ?""",
                (oid,)).fetchone()
        if row is None:
            return error(404, "order_not_found", order_id=oid)
        return jsonify(dict(row))

    @app.get("/purchase-orders/<path:po_number>")
    def get_purchase_order(po_number: str):
        po = normalize_po_number(po_number)
        with db() as con:
            row = con.execute("SELECT * FROM purchase_orders WHERE po_number = ?", (po,)).fetchone()
        if row is None:
            return error(404, "purchase_order_not_found", po_number=po)
        return jsonify(dict(row))

    # ---------------- action endpoints ----------------

    @app.post("/tickets")
    def create_ticket():
        data = body()
        missing = required(data, "item_id", "ticket_type", "customer_email", "summary")
        if missing:
            return error(400, "missing_fields", detail=missing)
        ticket_type = str(data["ticket_type"]).strip().lower()
        if ticket_type not in TICKET_TYPES:
            return error(400, "invalid_ticket_type", allowed=sorted(TICKET_TYPES))
        email = str(data["customer_email"]).strip().lower()
        order_id = normalize_order_id(data.get("order_id"))
        with db() as con:
            if con.execute("SELECT 1 FROM customers WHERE lower(email) = ?", (email,)).fetchone() is None:
                return error(404, "customer_not_found", email=email)
            if order_id and con.execute("SELECT 1 FROM orders WHERE id = ?", (order_id,)).fetchone() is None:
                return error(404, "order_not_found", order_id=order_id)
            cur = con.execute(
                "INSERT INTO tickets (run_name, item_id, ticket_type, customer_email, order_id, summary, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(data.get("run_name") or "default"), str(data["item_id"]), ticket_type, email, order_id,
                 str(data["summary"]), now()))
            ticket_id = cur.lastrowid
        return jsonify({"ticket_id": ticket_id, "ticket_type": ticket_type, "order_id": order_id}), 201

    @app.post("/invoices")
    def record_invoice():
        data = body()
        missing = required(data, "item_id", "invoice_number", "po_number", "amount", "supplier")
        if missing:
            return error(400, "missing_fields", detail=missing)
        amount = parse_amount(data["amount"])
        if amount is None:
            return error(400, "invalid_amount", detail=str(data["amount"]))
        po_number = normalize_po_number(data["po_number"])
        invoice_number = normalize_invoice_number(data["invoice_number"])
        run_name = str(data.get("run_name") or "default")
        with db() as con:
            po = con.execute("SELECT * FROM purchase_orders WHERE po_number = ?", (po_number,)).fetchone()
            if po is None:
                return error(404, "purchase_order_not_found", po_number=po_number)
            if not amounts_equal(amount, po["amount"]):
                return error(422, "amount_mismatch", po_amount=po["amount"], invoice_amount=amount)
            try:
                cur = con.execute(
                    "INSERT INTO recorded_invoices (run_name, item_id, invoice_number, po_number, supplier, amount,"
                    " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_name, str(data["item_id"]), invoice_number, po_number, str(data["supplier"]), amount, now()))
            except sqlite3.IntegrityError:
                return error(409, "invoice_already_recorded", invoice_number=invoice_number)
            record_id = cur.lastrowid
        return jsonify({"record_id": record_id, "invoice_number": invoice_number, "po_number": po_number}), 201

    @app.post("/flags")
    def flag_for_human():
        data = body()
        missing = required(data, "item_id", "reason")
        if missing:
            return error(400, "missing_fields", detail=missing)
        with db() as con:
            cur = con.execute("INSERT INTO flags (run_name, item_id, reason, created_at) VALUES (?, ?, ?, ?)",
                              (str(data.get("run_name") or "default"), str(data["item_id"]), str(data["reason"]), now()))
            flag_id = cur.lastrowid
        return jsonify({"flag_id": flag_id}), 201

    @app.post("/spam")
    def mark_spam():
        data = body()
        missing = required(data, "item_id")
        if missing:
            return error(400, "missing_fields", detail=missing)
        with db() as con:
            cur = con.execute("INSERT INTO spam_marks (run_name, item_id, reason, created_at) VALUES (?, ?, ?, ?)",
                              (str(data.get("run_name") or "default"), str(data["item_id"]),
                               str(data.get("reason") or ""), now()))
            spam_id = cur.lastrowid
        return jsonify({"spam_id": spam_id}), 201

    # ---------------- listing / housekeeping ----------------

    @app.get("/<any(tickets, invoices, flags, spam):kind>")
    def list_actions(kind: str):
        table = LIST_TABLES[kind]
        run_name = request.args.get("run_name")
        with db() as con:
            if run_name:
                rows = con.execute(f"SELECT * FROM {table} WHERE run_name = ? ORDER BY id", (run_name,)).fetchall()
            else:
                rows = con.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
        return jsonify([dict(r) for r in rows])

    @app.delete("/runs/<run_name>")
    def clear_run(run_name: str):
        deleted = {}
        with db() as con:
            for kind, table in LIST_TABLES.items():
                deleted[kind] = con.execute(f"DELETE FROM {table} WHERE run_name = ?", (run_name,)).rowcount
        return jsonify({"run_name": run_name, "deleted": deleted})

    @app.errorhandler(404)
    def not_found(_e):
        return error(404, "not_found", detail="unknown endpoint")

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return error(405, "method_not_allowed")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock business API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("API_PORT", DEFAULT_PORT)))
    args = parser.parse_args()
    app = create_app()
    print(f"Mock API running on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
