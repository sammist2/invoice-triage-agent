"""Loading inbox items and turning them (plus any PDF) into text for the model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pdfplumber

from agent.config import PROJECT_DIR

DATA_DIR = PROJECT_DIR / "data"


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=None)
def extract_pdf_text(path: str) -> str:
    """Pull the text out of every page of a PDF."""
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip()


def attachment_texts(item: dict, data_dir: Path = DATA_DIR) -> list[tuple[str, str]]:
    """[(filename, text), ...] for every attachment on the item."""
    out = []
    for rel in item.get("attachments", []):
        path = data_dir / rel
        if path.suffix.lower() == ".pdf" and path.exists():
            out.append((rel, extract_pdf_text(str(path))))
        else:
            out.append((rel, "(attachment could not be read)"))
    return out


def format_item(item: dict, data_dir: Path = DATA_DIR) -> str:
    """The message exactly as the model will see it."""
    parts = [
        f"From: {item['from']}",
        f"To: {item['to']}",
        f"Received: {item['received_at']}",
        f"Subject: {item['subject']}",
        "",
        item["body"],
    ]
    for name, text in attachment_texts(item, data_dir):
        parts += ["", f"--- Attachment: {name} (text extracted from PDF) ---", text, "--- End of attachment ---"]
    return "\n".join(parts)
