#!/usr/bin/env python3
"""Tiny round-robin proxy for OpenAI-compatible local model servers."""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Upstreams:
    def __init__(self, urls: list[str]) -> None:
        self.urls = [url.rstrip("/") for url in urls]
        self.index = 0
        self.lock = threading.Lock()

    def next(self) -> str:
        with self.lock:
            url = self.urls[self.index % len(self.urls)]
            self.index += 1
            return url


class Handler(BaseHTTPRequestHandler):
    upstreams: Upstreams
    timeout: float

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[openai_lb_proxy] {self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forward(self, method: str, body: bytes = b"") -> None:
        last_error = ""
        for _ in range(len(self.upstreams.urls)):
            upstream = self.upstreams.next()
            target = upstream + self.path
            req = urllib.request.Request(target, data=body if method == "POST" else None, method=method)
            req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))
            auth = self.headers.get("Authorization")
            if auth:
                req.add_header("Authorization", auth)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = resp.read()
                    self.send_response(resp.status)
                    self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("X-Upstream", upstream)
                    self.end_headers()
                    self.wfile.write(payload)
                    return
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                self.send_response(exc.code)
                self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Upstream", upstream)
                self.end_headers()
                self.wfile.write(payload)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"[openai_lb_proxy] upstream failed {upstream}: {last_error}", flush=True)
                time.sleep(0.1)
        self._send_json(502, {"error": {"message": f"all upstreams failed; last={last_error}"}})

    def do_GET(self) -> None:  # noqa: N802
        self._forward("GET")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self._forward("POST", self.rfile.read(length) if length else b"{}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18060)
    ap.add_argument("--upstream", action="append", required=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    Handler.upstreams = Upstreams(args.upstream)
    Handler.timeout = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[openai_lb_proxy] serving on {args.host}:{args.port} "
        f"upstreams={Handler.upstreams.urls}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
