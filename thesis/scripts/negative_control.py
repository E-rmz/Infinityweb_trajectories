#!/usr/bin/env python3
"""Negative control: verify the validator catches solver/verifier drift.

Loads a generated task.py, replaces its solver with one that omits email 5's
star mutation, and runs the validator. The golden-path arm MUST fail —
otherwise the validator is not actually checking what we think it is.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from thesis.synthetic.state_capture import load_actions  # noqa: E402
from thesis.synthetic.validator import Validator  # noqa: E402


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("synth_neg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _corrupted_solver(state):
    """Stars only email 1, OMITS email 5 — should make golden-path fail."""
    for e in state.get("emails", []):
        if e.get("id") == 1:
            e["isStarred"] = True
            e["starType"] = "yellow-star"


async def main() -> int:
    task_dir = REPO_ROOT / "thesis" / "synthetic-tasks" / "gmail" / "task_e1.bak" / "path_004"
    src_dir = REPO_ROOT / "thesis" / "traces" / "gmail" / "task_e1.bak" / "trajectories" / "path_004"
    web_app_dir = REPO_ROOT / "apps" / "gmail"

    mod = _load_module(task_dir / "task.py")
    actions = load_actions(src_dir)

    print("=== negative control: corrupted solver omits email 5 ===")
    async with Validator(web_app_dir=web_app_dir, port=8771, headless=True) as v:
        result = await v.validate(
            solver_fn=_corrupted_solver,
            verify_fn=mod.verify,
            actions=actions,
        )
    out = result.to_dict()
    print(json.dumps(out, indent=2))

    expected = (not out["golden_pass"]) and out["seed_fail"] and out["replay_pass"]
    print()
    if expected:
        print("PASS: golden_pass is False (drift caught), seed_fail and replay_pass still True.")
        return 0
    print("FAIL: validator did NOT catch the corruption.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
