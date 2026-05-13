"""Batched seed-satisfaction check.

Apps' /api/state returns 404 until the BROWSER has pushed its initial seed
state via PUT (see apps/<app>/js/state.js _pushStateToServer). So we boot
the server once, open the app in headless Chromium once to trigger the
seed push, then run each verifier via plain HTTP with /api/reset between
calls. Tells us which atomic tasks are degenerate (already pass at seed)
so Step 1 can exclude them.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import requests

ROOT = Path(__file__).resolve().parents[2]
# evaluation/tasks.py uses a top-level `from agents import AgentResult`, which
# only resolves when evaluation/ is on sys.path (not just the repo root).
for _p in (str(ROOT), str(ROOT / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.server import start_server, stop_server, wait_for_server  # noqa: E402
from evaluation.tasks import load_verifier  # noqa: E402

from .schema import AtomicTemplate

VerifierFn = Callable[[str], tuple[bool, str]]

DEFAULT_PORT = 8765
SERVER_BOOT_TIMEOUT = 120
RESET_TIMEOUT = 30
SEED_PUSH_TIMEOUT = 30


def _bootstrap_seed_via_browser(server_url: str) -> None:
    """Open the app in headless Chromium once; wait for the initial PUT.

    The browser's app.js fires _pushStateToServer() on load, which gives the
    server its `_seed_state` snapshot. After this the /api/reset endpoint
    can restore state without needing the browser again.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            # `networkidle` never settles because the app keeps an SSE
            # connection open. `domcontentloaded` is enough; the seed PUT
            # fires shortly after the JS bundle runs.
            page.goto(server_url, wait_until="domcontentloaded", timeout=15000)
            deadline = time.time() + SEED_PUSH_TIMEOUT
            while time.time() < deadline:
                try:
                    r = requests.get(f"{server_url}/api/state", timeout=5)
                    if r.status_code == 200:
                        return
                except requests.RequestException:
                    pass
                time.sleep(0.25)
            raise RuntimeError(
                "Browser opened the app but /api/state never returned 200 — "
                "seed push likely failed"
            )
        finally:
            browser.close()


def safe_verify(verify_fn: VerifierFn, server_url: str) -> tuple[bool, str]:
    try:
        ok, msg = verify_fn(server_url)
        return bool(ok), str(msg)
    except Exception as e:
        return False, f"Verifier exception: {type(e).__name__}: {e}"


def check_seeds_for_app(
    app_dir: Path,
    templates: list[AtomicTemplate],
    *,
    port: int = DEFAULT_PORT,
    batch_timeout: int = 7200,
) -> dict[str, dict]:
    """Return {task_id: {'passed': bool, 'message': str}} for every template.

    Boots the server once, resets state between templates. Bails out and
    marks remaining tasks as `('Verifier batch timeout', False)` if the wall
    clock exceeds `batch_timeout`.
    """
    app_dir = Path(app_dir)
    server_url = f"http://localhost:{port}"
    proc = start_server(str(app_dir), port)
    deadline = time.time() + batch_timeout
    out: dict[str, dict] = {}
    try:
        if not wait_for_server(port, timeout=SERVER_BOOT_TIMEOUT):
            for t in templates:
                out[t.task_id] = {"passed": False,
                                  "message": "server failed to start"}
            return out

        try:
            _bootstrap_seed_via_browser(server_url)
        except Exception as e:
            for t in templates:
                out[t.task_id] = {"passed": False,
                                  "message": f"seed bootstrap failed: {e}"}
            return out

        for t in templates:
            if time.time() > deadline:
                out[t.task_id] = {"passed": False,
                                  "message": "Verifier batch timeout"}
                continue

            # /api/reset must precede every call — `/api/state` returns 404
            # until the server has been seeded at least once, and we want a
            # clean seed between templates anyway.
            try:
                r = requests.post(
                    f"{server_url}/api/reset", timeout=RESET_TIMEOUT
                )
                r.raise_for_status()
                time.sleep(0.5)
            except requests.RequestException as e:
                out[t.task_id] = {"passed": False,
                                  "message": f"reset failed: {e}"}
                continue

            try:
                verify_fn = load_verifier(str(app_dir), t.verifier_path)
            except Exception as e:
                out[t.task_id] = {"passed": False,
                                  "message": f"verifier load failed: {e}"}
                continue

            ok, msg = safe_verify(verify_fn, server_url)
            out[t.task_id] = {"passed": ok, "message": msg}
    finally:
        stop_server(proc)
    return out
