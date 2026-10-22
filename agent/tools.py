"""The tools the agent can use, and the code that runs them.

TOOL_SCHEMAS is what the model sees: a name, a description and the
arguments for each tool (OpenAI "function calling" format).

ToolBox runs the tools the model asks for by calling the mock business API,
and keeps a record of everything that happened for the eval.
"""

from __future__ import annotations

import json
from typing import Literal
from urllib.parse import quote

import requests
from pydantic import BaseModel, ValidationError, field_validator

from shared.normalize import normalize_invoice_number, normalize_order_id, normalize_po_number, parse_amount

CATEGORIES = ("refund_request", "shipping_delay", "billing_question", "invoice_received", "spam")
ACTION_TOOLS = {"create_ticket", "record_invoice", "flag_for_human", "mark_spam"}


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


# Plain "string" (not ["string", "null"]) because some providers, like Gemini,
# reject union types. The model can still send null; SubmitResult accepts it.
NULLABLE_STRING = {"type": "string"}

TOOL_SCHEMAS = [
    _fn("lookup_order", "Look up an order by its order number. Returns the product, status and the customer it belongs to.",
        {"order_id": {"type": "string", "description": "Order number, e.g. ORD-10023"}}, ["order_id"]),
    _fn("lookup_customer", "Look up a customer by email address. Returns their details and their orders.",
        {"email": {"type": "string"}}, ["email"]),
    _fn("lookup_purchase_order", "Look up one of our purchase orders (PO) to a supplier. Returns the supplier and the approved amount.",
        {"po_number": {"type": "string", "description": "PO number, e.g. PO-5013"}}, ["po_number"]),
    _fn("create_ticket", "ACTION: open a support ticket for a customer.",
        {"ticket_type": {"type": "string", "enum": ["refund", "shipping", "billing"]},
         "customer_email": {"type": "string", "description": "The sender's email address"},
         "order_id": {**NULLABLE_STRING, "description": "Order number if there is one, otherwise null"},
         "summary": {"type": "string", "description": "One sentence describing the customer's problem"}},
        ["ticket_type", "customer_email", "summary"]),
    _fn("record_invoice", "ACTION: record a supplier invoice for payment. Only for invoices whose total matches the PO amount.",
        {"invoice_number": {"type": "string"}, "po_number": {"type": "string"},
         "amount": {"type": "number", "description": "Invoice total, e.g. 1746.28"},
         "supplier": {"type": "string"}},
        ["invoice_number", "po_number", "amount", "supplier"]),
    _fn("flag_for_human", "ACTION: send this message to a human because it breaks a policy rule or information is missing.",
        {"reason": {"type": "string", "description": "Why a human needs to look at it"}}, ["reason"]),
    _fn("mark_spam", "ACTION: mark the message as spam, a scam or phishing.",
        {"reason": {"type": "string"}}, ["reason"]),
    _fn("submit_result", "FINAL STEP: report the category and extracted fields. Call this once, after your action.",
        {"category": {"type": "string", "enum": list(CATEGORIES)},
         "order_id": {**NULLABLE_STRING, "description": "ORD-12345 or null"},
         "invoice_number": {**NULLABLE_STRING, "description": "Invoice number exactly as printed, or null"},
         "po_number": {**NULLABLE_STRING, "description": "PO-1234 or null"},
         "amount": {"type": "number", "description": "Invoice total (invoices only), or null"}},
        ["category"]),
]


class SubmitResult(BaseModel):
    """The final answer. Pydantic checks the types and cleans up the values."""
    category: Literal["refund_request", "shipping_delay", "billing_question", "invoice_received", "spam"]
    order_id: str | None = None
    invoice_number: str | None = None
    po_number: str | None = None
    amount: float | None = None

    @field_validator("category", mode="before")
    @classmethod
    def _clean_category(cls, v):
        return str(v).strip().lower() if v is not None else v

    @field_validator("order_id", mode="before")
    @classmethod
    def _clean_order(cls, v):
        return normalize_order_id(v)

    @field_validator("po_number", mode="before")
    @classmethod
    def _clean_po(cls, v):
        return normalize_po_number(v)

    @field_validator("invoice_number", mode="before")
    @classmethod
    def _clean_invoice(cls, v):
        return normalize_invoice_number(v)

    @field_validator("amount", mode="before")
    @classmethod
    def _clean_amount(cls, v):
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none")):
            return None
        parsed = parse_amount(v)
        if parsed is None:
            raise ValueError(f"amount must be a number, got {v!r}")
        return parsed


class ApiUnavailable(Exception):
    """The mock business API is not running."""


class ApiClient:
    def __init__(self, base_url: str, session: requests.Session | None = None, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout_s = timeout_s

    def request(self, method: str, path: str, **kwargs) -> tuple[int, dict | list]:
        try:
            resp = self.session.request(method, f"{self.base_url}{path}", timeout=self.timeout_s, **kwargs)
        except requests.RequestException as exc:
            raise ApiUnavailable(
                f"Could not reach the business API at {self.base_url}. Start it with: python -m api.app"
            ) from exc
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"error": "non_json_response", "detail": resp.text[:200]}

    def health(self) -> bool:
        try:
            status, _ = self.request("GET", "/health")
            return status == 200
        except ApiUnavailable:
            return False


class ToolBox:
    """Runs tool calls for ONE inbox item and records what happened."""

    def __init__(self, api: ApiClient, run_name: str, item: dict):
        self.api = api
        self.run_name = run_name
        self.item = item
        self.calls: list[dict] = []    # every tool call, for debugging
        self.actions: list[str] = []   # successful actions, e.g. "create_ticket:refund"
        self.result: SubmitResult | None = None

    def execute(self, name: str, raw_args) -> str:
        """Run one tool. Returns a JSON string to send back to the model."""
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                return self._log(name, raw_args, None, {"error": "invalid_json_arguments",
                                                        "detail": "Arguments must be a JSON object."})
            if not isinstance(args, dict):
                return self._log(name, raw_args, None, {"error": "invalid_json_arguments",
                                                        "detail": "Arguments must be a JSON object."})

        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return self._log(name, args, None, {"error": "unknown_tool", "detail": f"No tool named {name!r}."})
        status, response = handler(args)

        if name in ACTION_TOOLS and status is not None and 200 <= status < 300:
            action = f"create_ticket:{response.get('ticket_type')}" if name == "create_ticket" else name
            self.actions.append(action)
        return self._log(name, args, status, response)

    def _log(self, name: str, args, status: int | None, response) -> str:
        self.calls.append({"name": name, "args": args, "status": status, "response": response})
        return json.dumps(response)

    def _missing(self, args: dict, *fields: str) -> tuple[None, dict] | None:
        missing = [f for f in fields if args.get(f) in (None, "")]
        if missing:
            return None, {"error": "missing_arguments", "detail": ", ".join(missing)}
        return None

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        return self.api.request("POST", path, json={"run_name": self.run_name, "item_id": self.item["id"], **payload})

    # ---- read tools ----
    def _tool_lookup_order(self, args):
        return self._missing(args, "order_id") or self.api.request("GET", f"/orders/{quote(str(args['order_id']), safe='')}")

    def _tool_lookup_customer(self, args):
        return self._missing(args, "email") or self.api.request("GET", "/customers", params={"email": args["email"]})

    def _tool_lookup_purchase_order(self, args):
        return (self._missing(args, "po_number")
                or self.api.request("GET", f"/purchase-orders/{quote(str(args['po_number']), safe='')}"))

    # ---- action tools ----
    def _tool_create_ticket(self, args):
        return self._post("/tickets", {
            "ticket_type": args.get("ticket_type"),
            "customer_email": args.get("customer_email") or self.item.get("from"),
            "order_id": args.get("order_id"),
            "summary": args.get("summary"),
        })

    def _tool_record_invoice(self, args):
        return self._post("/invoices", {k: args.get(k) for k in ("invoice_number", "po_number", "amount", "supplier")})

    def _tool_flag_for_human(self, args):
        return self._post("/flags", {"reason": args.get("reason")})

    def _tool_mark_spam(self, args):
        return self._post("/spam", {"reason": args.get("reason") or ""})

    # ---- final answer ----
    def _tool_submit_result(self, args):
        try:
            self.result = SubmitResult(**args)
        except (ValidationError, TypeError) as exc:
            return None, {"error": "invalid_result", "detail": str(exc)[:500]}
        return None, {"status": "recorded", "result": self.result.model_dump()}
