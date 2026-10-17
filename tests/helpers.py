"""Shared test helpers: temporary data, a live API server, and a fake LLM server."""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from werkzeug.serving import make_server

from api.app import create_app
from data.generate_data import generate

logging.getLogger("werkzeug").setLevel(logging.ERROR)  # hide per-request log lines in test output

_DATASET_DIR: Path | None = None


def dataset_dir() -> Path:
    """Generate the dataset once into a temp folder and reuse it across tests."""
    global _DATASET_DIR
    if _DATASET_DIR is None:
        _DATASET_DIR = Path(tempfile.mkdtemp(prefix="triage-data-"))
        generate(_DATASET_DIR)
    return _DATASET_DIR


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ApiServer:
    """Runs the real Flask API on a free port in a background thread."""

    def __init__(self, db_path: Path):
        self.server = make_server("127.0.0.1", 0, create_app(db_path))
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.thread.join()


class FakeLLM:
    """A tiny OpenAI-compatible server. `behaviour(request_json)` returns
    either a message dict, or (status_code, body) to simulate an error."""

    def __init__(self, behaviour):
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                fake.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                out = behaviour(body)
                if isinstance(out, tuple):
                    status, payload = out
                else:
                    status = 200
                    payload = {"choices": [{"index": 0, "message": out, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # keep test output clean
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    """A model message that calls one tool."""
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def text_message(text: str) -> dict:
    return {"role": "assistant", "content": text}
