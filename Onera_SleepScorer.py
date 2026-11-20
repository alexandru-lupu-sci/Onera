"""Standalone launcher for Onera SleepScorer (FastAPI + Uvicorn + vanilla HTML5/JS)."""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time
import webbrowser

os.environ.setdefault("MNE_USE_NUMBA", "false")

import uvicorn

from onera_scorer.backend import app


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _open_browser_when_ready(host: str, port: int, server: uvicorn.Server) -> None:
    url = f"http://{host}:{port}"
    for _ in range(100):
        if getattr(server, "started", False):
            webbrowser.open(url)
            return
        time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Onera SleepScorer local web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="TCP port to bind (default: 0 = automatic free port)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the system browser automatically")
    args = parser.parse_args()

    port = int(args.port) if int(args.port) > 0 else _pick_free_port(args.host)
    url = f"http://{args.host}:{port}"
    print(f"Onera SleepScorer running at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    config = uvicorn.Config(app, host=args.host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(args.host, port, server),
            daemon=True,
        ).start()

    server.run()


if __name__ == "__main__":
    main()
