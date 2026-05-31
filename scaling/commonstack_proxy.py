#!/usr/bin/env python3
"""Tiny OpenAI-compatible HTTP forward proxy for restricted GPU hosts.

The GPU box can reach localhost but currently cannot open outbound HTTPS to
CommonStack. Run this on a machine with internet, then expose it to the GPU
with an SSH reverse tunnel:

    python scaling/commonstack_proxy.py --port 18082
    ssh -N -R 18082:127.0.0.1:18082 gpu-host

Point the GPU client at `http://127.0.0.1:18082/v1`. Authorization headers are
forwarded from the client; this proxy does not store or print API keys.
"""
from __future__ import annotations

import argparse
import http.client
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    target_host = "api.commonstack.ai"
    target_port = 443

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"[commonstack-proxy] {self.command} {self.path} " + fmt % args + "\n")

    def _proxy(self) -> None:
        body_len = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(body_len) if body_len else None
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower()
            not in {"host", "connection", "proxy-connection", "keep-alive", "transfer-encoding"}
        }
        headers["Host"] = self.target_host

        conn = http.client.HTTPSConnection(self.target_host, self.target_port, timeout=180)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            self.send_response(resp.status, resp.reason)
            for key, value in resp.getheaders():
                if key.lower() in {"connection", "transfer-encoding", "content-encoding"}:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            conn.close()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._proxy()

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._proxy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(
        f"[commonstack-proxy] listening on {args.host}:{args.port} -> "
        f"https://{ProxyHandler.target_host}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
