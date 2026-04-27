#!/usr/bin/env python3
"""Run MCTS on a single task (debugging entry point).

Usage:
    cd webarena-infinity
    python -m thesis.scripts.run_one_task \
        --web-app apps/gmail \
        --task-id task_e6 \
        --budget 30 --max-depth 8 --top-k 5 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from thesis.mcts.runner import run_task  # noqa: E402
from thesis.mcts.search import MCTSConfig  # noqa: E402


def _load_tasks(web_app_dir: Path, suite: str) -> list[dict]:
    return json.loads((web_app_dir / f"{suite}.json").read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--web-app", required=True, help="path to apps/<name>")
    p.add_argument("--task-id", required=True, help="e.g. task_e6")
    p.add_argument("--task-suite", default="real-tasks")
    p.add_argument("--output-dir", default=None,
                   help="default: thesis/traces/<app-name>/<task-id>")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--sim-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-screenshots", action="store_true")
    p.add_argument("--no-states", action="store_true")
    args = p.parse_args()

    web_app_dir = (REPO_ROOT / args.web_app).resolve() if not Path(args.web_app).is_absolute() \
        else Path(args.web_app)
    if not web_app_dir.exists():
        print(f"web-app dir not found: {web_app_dir}", file=sys.stderr)
        return 2

    tasks = _load_tasks(web_app_dir, args.task_suite)
    task = next((t for t in tasks if t["id"] == args.task_id), None)
    if task is None:
        print(f"task id {args.task_id!r} not found in {args.task_suite}.json", file=sys.stderr)
        return 2

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = REPO_ROOT / "thesis" / "traces" / web_app_dir.name / task["id"]

    cfg = MCTSConfig(
        budget=args.budget,
        max_depth=args.max_depth,
        top_k=args.top_k,
        sim_steps=args.sim_steps,
        seed=args.seed,
    )
    summary = asyncio.run(run_task(
        task=task,
        web_app_dir=web_app_dir,
        output_dir=output_dir,
        port=args.port,
        config=cfg,
        capture_screenshots=not args.no_screenshots,
        save_states=not args.no_states,
    ))
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("error") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
