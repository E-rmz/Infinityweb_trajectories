"""State restoration: ``POST /api/reset`` + replay an action prefix.

This is the single most important primitive in the search loop. The
production server has no checkpoint API — the only sanctioned way to
restore an internal MCTS node is:

  1. ``POST /api/reset`` (server restores ``_app_state`` to the immutable
     seed; an SSE ``reset`` event is broadcast to the browser).
  2. The browser's pre-existing SSE handler clears localStorage,
     reloads seed data in-memory, and ``PUT``s it back to the server.
  3. We replay the recorded action prefix one step at a time.

The replay is correct as long as actions use the stable selectors from
``browser.py`` (id / data-action / data-testid / role+name).
"""

from __future__ import annotations

import asyncio
import json
import time

import requests
from playwright.async_api import Page

from .action import Action
from .browser import exec_action


def _seed_snapshot(server_url: str) -> dict:
    """Read the current /api/state — used as a seed reference."""
    r = requests.get(f"{server_url}/api/state", timeout=3)
    r.raise_for_status()
    return r.json()


async def _wait_for_seed_repush(server_url: str, expected_seed: dict, timeout_s: float = 3.0) -> None:
    """Wait until the browser has re-PUT seed state after a reset.

    The server's deep-copy on reset means /api/state will already match
    the seed, but the browser's later PUT (triggered by its SSE reset
    handler) must also land before we run any new actions — otherwise the
    browser's stale, in-memory pre-reset state can overwrite the server
    on the next user action.

    We approximate "browser re-PUT done" by polling until the JSON
    response equals the expected seed for two consecutive reads spaced
    ~150 ms apart. Falls back to a timeout-bounded sleep.
    """
    deadline = time.time() + timeout_s
    last_match = False
    while time.time() < deadline:
        try:
            r = requests.get(f"{server_url}/api/state", timeout=2)
            ok = r.status_code == 200 and r.json() == expected_seed
        except (requests.RequestException, ValueError):
            ok = False
        if ok and last_match:
            return
        last_match = ok
        await asyncio.sleep(0.15)
    # Best-effort — proceed even if we never got two matching reads.


async def reset_to_seed(page: Page, server_url: str, expected_seed: dict) -> None:
    """Issue POST /api/reset and wait for the browser to settle."""
    requests.post(f"{server_url}/api/reset", timeout=5).raise_for_status()
    await _wait_for_seed_repush(server_url, expected_seed)
    # Best-effort UI settle: wait for the inbox view to be present.
    try:
        await page.wait_for_load_state("networkidle", timeout=1500)
    except Exception:
        await asyncio.sleep(0.2)


async def replay(page: Page, actions: list[Action]) -> int:
    """Replay actions sequentially. Returns the index of the first failing
    action (or ``len(actions)`` if all succeeded). Caller decides whether
    a partial replay is fatal."""
    for i, a in enumerate(actions):
        try:
            await exec_action(page, a)
        except Exception:
            return i
    return len(actions)


async def reset_and_replay(
    page: Page,
    server_url: str,
    expected_seed: dict,
    actions: list[Action],
) -> bool:
    """Convenience wrapper: reset, then replay. Returns True on full success."""
    await reset_to_seed(page, server_url, expected_seed)
    n = await replay(page, actions)
    return n == len(actions)


def state_hash(state: dict) -> str:
    """Stable hash of /api/state for de-duplicating MCTS leaves."""
    import hashlib

    blob = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()
