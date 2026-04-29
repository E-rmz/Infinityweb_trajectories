#!/usr/bin/env python3
"""Rebuild a per-task index.json by re-reading every path's validation.json.

Useful when --only-path runs overwrite the index with a single-path summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task-dir", required=True, help="thesis/synthetic-tasks/<env>/<task_id>")
    args = p.parse_args()

    task_dir = Path(args.task_dir).resolve()
    if not task_dir.exists():
        print(f"not found: {task_dir}", file=sys.stderr)
        return 2

    results = []
    for path_dir in sorted(task_dir.glob("path_*")):
        vj = path_dir / "validation.json"
        if not vj.exists():
            continue
        results.append({"path_id": path_dir.name, **json.loads(vj.read_text())})

    summary = {
        "env": task_dir.parent.name,
        "task_id": task_dir.name,
        "n_built": len(results),
        "n_valid": sum(1 for r in results if r.get("valid")),
        "results": results,
    }
    (task_dir / "index.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {task_dir/'index.json'}: n_built={summary['n_built']} n_valid={summary['n_valid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
