#!/usr/bin/env python3
"""Simple HTTP proxy that injects OpenRouter provider preferences into requests."""
from __future__ import annotations

import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class OpenRouterProxyHandler(BaseHTTPRequestHandler):
    target_base = "https://openrouter.ai"
    provider_ignore: list[str] = []

    def log_message(self, format, *args):
        print(f"[PROXY] {format % args}", file=sys.stderr)

    def _forward(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"

        # Inject provider preferences
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if isinstance(payload, dict):
            provider_cfg = payload.get("provider", {})
            if not isinstance(provider_cfg, dict):
                provider_cfg = {}
            if self.provider_ignore:
                existing_ignore = set(provider_cfg.get("ignore", []))
                existing_ignore.update(self.provider_ignore)
                provider_cfg["ignore"] = sorted(existing_ignore)
                payload["provider"] = provider_cfg
                print(f"[PROXY] Injected provider.ignore: {provider_cfg['ignore']}", file=sys.stderr)
            body = json.dumps(payload).encode("utf-8")

        target_url = self.target_base + self.path
        req = urllib.request.Request(
            target_url,
            data=body,
            headers={key: val for key, val in self.headers.items() if key.lower() not in ("host", "content-length")},
            method=self.command,
        )

        try:
            with urllib.request.urlopen(req) as resp:
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    if key.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(key, val)
                self.send_header("Content-Length", str(len(resp.read())))
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()


def start_proxy(port: int, provider_ignore: list[str]) -> HTTPServer:
    OpenRouterProxyHandler.provider_ignore = provider_ignore
    server = HTTPServer(("127.0.0.1", port), OpenRouterProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[PROXY] OpenRouter proxy running on http://127.0.0.1:{port}", file=sys.stderr)
    print(f"[PROXY] Ignoring providers: {provider_ignore}", file=sys.stderr)
    return server


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--ignore", nargs="+", default=["Io Net", "Chutes", "Ambient", "SiliconFlow"])
    args = parser.parse_args()

    server = start_proxy(args.port, args.ignore)
    try:
        while True:
            pass
    except KeyboardInterrupt:
        server.shutdown()
