"""Helpers to clean up IDs and amounts so different spellings compare equal.

People (and LLMs) write the same order number in many ways: "ORD-10023",
"order #10023", "ord-10023". These functions turn them into one canonical form.
They are used by the API, the agent and the eval, so all three agree.
"""

from __future__ import annotations

import re


def normalize_order_id(value) -> str | None:
    """'order #10023' -> 'ORD-10023'. Returns None for empty values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", "n/a"):
        return None
    match = re.search(r"\d{5,}", text)
    return f"ORD-{match.group(0)}" if match else text.upper()


def normalize_po_number(value) -> str | None:
    """'po 5013' -> 'PO-5013'. Returns None for empty values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("null", "none", "n/a"):
        return None
    match = re.search(r"\d{4,}", text)
    return f"PO-{match.group(0)}" if match else text.upper()


def normalize_invoice_number(value) -> str | None:
    """' grf-14930 ' -> 'GRF-14930'. Returns None for empty values."""
    if value is None:
        return None
    text = re.sub(r"\s+", "", str(value)).upper()
    if not text or text in ("NULL", "NONE", "N/A"):
        return None
    return text


def parse_amount(value) -> float | None:
    """'$1,234.50' -> 1234.5. Returns None if it can't be read as a number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = re.sub(r"[,$\s]|CAD|USD", "", str(value), flags=re.IGNORECASE)
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def amounts_equal(a, b, tolerance: float = 0.005) -> bool:
    """True if two amounts are the same to the cent."""
    a, b = parse_amount(a), parse_amount(b)
    return a is not None and b is not None and abs(a - b) < tolerance
