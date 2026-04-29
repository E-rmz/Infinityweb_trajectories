#!/usr/bin/env python3
"""Build synthetic tasks for one (env, task_id).

Usage::

    cd webarena-infinity
    python -m thesis.scripts.build_synthetic \
        --env gmail --task-id task_e1 \
        [--only-path path_004] [--port 8765] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from thesis.synthetic.builder import build_for_task  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="environment name (e.g. gmail)")
    p.add_argument("--task-id", required=True, help="source task id (e.g. task_e1)")
    p.add_argument("--traces-dir", default=None,
                   help="default: thesis/traces/<env>")
    p.add_argument("--out-dir", default=None,
                   help="default: thesis/synthetic-tasks/<env>/<task_id>")
    p.add_argument("--web-app-dir", default=None,
                   help="default: apps/<env>")
    p.add_argument("--only-path", default=None,
                   help="restrict to a single path_NNN (for debugging)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--force", action="store_true",
                   help="rebuild even if validation.json already exists")
    p.add_argument("--no-headless", action="store_true",
                   help="run browser with a visible window (debugging)")
    args = p.parse_args()

    traces_dir = Path(args.traces_dir) if args.traces_dir else REPO_ROOT / "thesis" / "traces" / args.env
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "thesis" / "synthetic-tasks" / args.env / args.task_id
    web_app_dir = Path(args.web_app_dir) if args.web_app_dir else REPO_ROOT / "apps" / args.env

    if not traces_dir.exists():
        print(f"traces dir not found: {traces_dir}", file=sys.stderr)
        return 2
    if not web_app_dir.exists():
        print(f"web-app dir not found: {web_app_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== build_synthetic ===")
    print(f"  env       : {args.env}")
    print(f"  task_id   : {args.task_id}")
    print(f"  traces    : {traces_dir}")
    print(f"  output    : {out_dir}")
    print(f"  web-app   : {web_app_dir}")
    print(f"  only_path : {args.only_path or '<all>'}")
    print()

    summary = asyncio.run(build_for_task(
        env=args.env,
        task_id=args.task_id,
        traces_dir=traces_dir,
        out_dir=out_dir,
        web_app_dir=web_app_dir,
        port=args.port,
        only_path=args.only_path,
        headless=not args.no_headless,
        force=args.force,
    ))

    print()
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
