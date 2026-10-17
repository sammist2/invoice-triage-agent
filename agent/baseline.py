"""A rule-based baseline: keywords and regular expressions, no AI.

Every AI project needs something simple to beat. If your LLM agent can't
clearly outperform these ~100 lines of rules, the LLM isn't earning its cost.
It uses the same ToolBox and follows the same policy as the agent, so the
eval can compare them directly.

The keyword lists are deliberately generic (what you'd write in 20 minutes
without studying the data). Tuning them to the test set would be cheating:
the rules would look great here and fail on next week's emails.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from agent.agent import build_record
from agent.config import Settings
from agent.documents import DATA_DIR, attachment_texts
from agent.tools import ApiClient, ToolBox
from shared.normalize import amounts_equal, normalize_invoice_number, parse_amount

SPAM_WORDS = ["unsubscribe", "click here", "you have won", "lottery", "prize", "limited time", "act now",
              "crypto", "loan", "verify your account", "banking details", "password", "guarantee", "free"]
REFUND_WORDS = ["refund", "money back", "return"]
SHIPPING_WORDS = ["shipping", "delivery", "delivered", "tracking", "arrive", "where is my order", "late"]

ORDER_RE = re.compile(r"(?:ORD-|order\s*(?:#|number)?\s*|#)(\d{5})\b", re.IGNORECASE)
INVOICE_RE = re.compile(r"Invoice\s*(?:No\.?|Number|#)\s*:\s*(\S+)", re.IGNORECASE)
PO_RE = re.compile(r"\bPO-(\d{4})\b")
TOTAL_RE = re.compile(r"^(?:Total|Total Due|Amount Due)\s+\$([\d,]+\.\d{2})", re.MULTILINE | re.IGNORECASE)


def classify(text: str, has_pdf: bool) -> str:
    lower = text.lower()
    if any(w in lower for w in SPAM_WORDS):
        return "spam"
    if has_pdf:
        return "invoice_received"
    if any(w in lower for w in REFUND_WORDS):
        return "refund_request"
    if any(w in lower for w in SHIPPING_WORDS):
        return "shipping_delay"
    return "billing_question"


def process_item_baseline(item: dict, api: ApiClient, settings: Settings, run_name: str,
                          data_dir: Path = DATA_DIR) -> dict:
    start = time.perf_counter()
    tb = ToolBox(api, run_name, item)
    attachments = attachment_texts(item, data_dir)
    email_text = f"{item['subject']}\n{item['body']}"
    category = classify(email_text, has_pdf=bool(attachments))
    result = {"category": category}
    sender = item["from"]

    if category == "spam":
        tb.execute("mark_spam", {"reason": "spam keywords"})

    elif category == "invoice_received":
        pdf = attachments[0][1]
        inv = INVOICE_RE.search(pdf)
        po = PO_RE.search(pdf)
        total = TOTAL_RE.findall(pdf)
        result.update(invoice_number=normalize_invoice_number(inv.group(1)) if inv else None,
                      po_number=f"PO-{po.group(1)}" if po else None,
                      amount=parse_amount(total[-1]) if total else None)
        supplier = pdf.splitlines()[1] if len(pdf.splitlines()) > 1 else "unknown"
        if not (result["po_number"] and result["amount"] is not None and result["invoice_number"]):
            tb.execute("flag_for_human", {"reason": "could not read invoice details"})
        else:
            tb.execute("lookup_purchase_order", {"po_number": result["po_number"]})
            status, po_data = tb.calls[-1]["status"], tb.calls[-1]["response"]
            if status == 200 and amounts_equal(po_data.get("amount"), result["amount"]):
                tb.execute("record_invoice", {"invoice_number": result["invoice_number"],
                                              "po_number": result["po_number"],
                                              "amount": result["amount"], "supplier": supplier})
            else:
                tb.execute("flag_for_human", {"reason": "PO not found or amount does not match"})

    else:
        match = ORDER_RE.search(email_text)
        order_id = f"ORD-{match.group(1)}" if match else None
        result["order_id"] = order_id
        if category == "billing_question":
            tb.execute("create_ticket", {"ticket_type": "billing", "customer_email": sender,
                                         "order_id": order_id, "summary": item["subject"]})
        elif not order_id:
            tb.execute("flag_for_human", {"reason": "no order number"})
        else:
            tb.execute("lookup_order", {"order_id": order_id})
            status, order = tb.calls[-1]["status"], tb.calls[-1]["response"]
            if status != 200 or order.get("customer_email", "").lower() != sender.lower():
                tb.execute("flag_for_human", {"reason": "order not found or belongs to someone else"})
            else:
                ticket_type = "refund" if category == "refund_request" else "shipping"
                tb.execute("create_ticket", {"ticket_type": ticket_type, "customer_email": sender,
                                             "order_id": order_id, "summary": item["subject"]})

    tb.execute("submit_result", result)
    return build_record(item, tb, settings, steps=len(tb.calls), error=None,
                        latency_s=time.perf_counter() - start)
