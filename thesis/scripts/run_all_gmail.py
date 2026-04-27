#!/usr/bin/env python3
"""Run MCTS sequentially across every task in apps/gmail/real-tasks.json.

The script is resume-aware: any task that already has ``summary.json``
under ``thesis/traces/gmail/<task_id>/`` is skipped.

Usage:
    cd webarena-infinity
    python -m thesis.scripts.run_all_gmail
    # …or restrict to one difficulty / suite:
    python -m thesis.scripts.run_all_gmail --difficulty easy
    python -m thesis.scripts.run_all_gmail --task-suite function-tasks
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from thesis.mcts.runner import run_task  # noqa: E402
from thesis.mcts.search import MCTSConfig  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--web-app", default="apps/gmail")
    p.add_argument("--task-suite", default="real-tasks")
    p.add_argument("--output-root", default="thesis/traces")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--sim-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-screenshots", action="store_true")
    p.add_argument("--no-states", action="store_true")
    p.add_argument("--no-resume", action="store_true",
                   help="re-run tasks even if summary.json already exists")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after N tasks (0 = no limit; useful for smoke tests)")
    args = p.parse_args()

    web_app_dir = (REPO_ROOT / args.web_app).resolve() if not Path(args.web_app).is_absolute() \
        else Path(args.web_app)
    if not web_app_dir.exists():
        print(f"web-app dir not found: {web_app_dir}", file=sys.stderr)
        return 2

    tasks_path = web_app_dir / f"{args.task_suite}.json"
    tasks = json.loads(tasks_path.read_text())
    if args.difficulty:
        tasks = [t for t in tasks if t.get("difficulty") == args.difficulty]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    output_root = (REPO_ROOT / args.output_root).resolve() / web_app_dir.name
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = MCTSConfig(
        budget=args.budget,
        max_depth=args.max_depth,
        top_k=args.top_k,
        sim_steps=args.sim_steps,
        seed=args.seed,
    )

    print(f"\n=== MCTS trajectory generation ===")
    print(f"  app          : {web_app_dir.name}")
    print(f"  suite        : {args.task_suite}  ({len(tasks)} tasks)")
    print(f"  budget       : {cfg.budget}")
    print(f"  max_depth    : {cfg.max_depth}")
    print(f"  top_k        : {cfg.top_k}")
    print(f"  sim_steps    : {cfg.sim_steps}")
    print(f"  output       : {output_root}")
    print()

    n_done = n_skipped = n_error = 0
    n_with_paths = 0
    t0 = time.time()
    for i, task in enumerate(tasks, start=1):
        task_dir = output_root / task["id"]
        summary_path = task_dir / "summary.json"
        if not args.no_resume and summary_path.exists():
            print(f"[{i}/{len(tasks)}] skip {task['id']} (already done)")
            n_skipped += 1
            continue

        print(f"[{i}/{len(tasks)}] {task['id']}  ({task.get('difficulty', '?')})")
        try:
            summary = asyncio.run(run_task(
                task=task,
                web_app_dir=web_app_dir,
                output_dir=task_dir,
                port=args.port,
                config=cfg,
                capture_screenshots=not args.no_screenshots,
                save_states=not args.no_states,
                quiet=True,
            ))
        except Exception as e:
            n_error += 1
            print(f"    -> ERROR: {type(e).__name__}: {e}")
            continue
        n_done += 1
        if summary.get("n_passing_paths", 0) > 0:
            n_with_paths += 1
            print(
                f"    -> {summary['n_passing_paths']} path(s)  "
                f"min_steps={summary['min_steps']}  states={summary['n_unique_states']}  "
                f"{summary['elapsed_s']:.1f}s"
            )
        elif summary.get("error"):
            print(f"    -> error: {summary['error']}")
        else:
            print(f"    -> 0 paths within budget  ({summary['elapsed_s']:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed/60:.1f} min ===")
    print(f"  ran:        {n_done}")
    print(f"  skipped:    {n_skipped}")
    print(f"  errored:    {n_error}")
    print(f"  with paths: {n_with_paths}/{n_done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
