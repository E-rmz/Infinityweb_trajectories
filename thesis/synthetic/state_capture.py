"""Capture seed and terminal ``/api/state`` for a passing trajectory.

Two paths exist:

* **Cached read** — for trajectories that already have ``states/step_NNN.json``
  files (the existing MCTS runner saves these by default), simply read the
  files. No browser, no replay.
* **Live replay** — when state files are missing or you want to re-verify, drive
  Playwright through the action sequence using the existing
  ``mcts.browser.exec_action`` and ``mcts.replay.reset_to_seed`` primitives.

The pipeline prefers the cached path; live replay is reserved for the
validator's replay-pass arm.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import requests
from playwright.async_api import Page

from ..mcts.action import Action
from ..mcts.browser import exec_action
from ..mcts.replay import reset_to_seed


# ---------------------------------------------------------------------------
# Cached state read
# ---------------------------------------------------------------------------


def load_cached_states(path_dir: Path) -> tuple[dict, dict, list[dict]] | None:
    """Return ``(seed, terminal, per_step)`` for a path's saved states.

    Returns ``None`` when the ``states/`` subdirectory is missing or empty.
    """
    states_dir = path_dir / "states"
    if not states_dir.exists():
        return None
    files = sorted(states_dir.glob("step_*.json"))
    if not files:
        return None
    per_step = [json.loads(f.read_text()) for f in files]
    return per_step[0], per_step[-1], per_step


def load_actions(path_dir: Path) -> list[Action]:
    """Reconstruct ``Action`` objects from a path's ``actions.json``."""
    raw = json.loads((path_dir / "actions.json").read_text())
    return [
        Action(
            kind=a["kind"],
            selector=a["selector"],
            value=a.get("value"),
            label=a.get("label", ""),
        )
        for a in raw
    ]


# ---------------------------------------------------------------------------
# Live replay
# ---------------------------------------------------------------------------


async def replay_capture(
    *,
    page: Page,
    server_url: str,
    seed_state: dict,
    actions: list[Action],
    capture_per_step: bool = False,
) -> tuple[dict, list[dict]]:
    """Reset, replay actions, and return ``(terminal_state, per_step_states)``.

    ``per_step_states`` is empty unless ``capture_per_step`` is True.
    """
    await reset_to_seed(page, server_url, seed_state)
    per_step: list[dict] = []
    if capture_per_step:
        per_step.append(_fetch_state(server_url))
    for a in actions:
        await exec_action(page, a)
        if capture_per_step:
            per_step.append(_fetch_state(server_url))
    terminal = _fetch_state(server_url)
    return terminal, per_step


def _fetch_state(server_url: str) -> dict:
    r = requests.get(f"{server_url}/api/state", timeout=3)
    r.raise_for_status()
    return r.json()
