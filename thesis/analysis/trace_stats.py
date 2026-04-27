#!/usr/bin/env python3
"""Aggregate statistics over a directory of MCTS traces.

Reads every ``thesis/traces/<app>/<task_id>/summary.json`` and produces:

  * a per-task table (CSV-ish, also written as JSON);
  * per-difficulty rollups (success-rate, min/median/max steps);
  * a global "diversity" score per task — the unique action key
    fraction, computed by reading each trajectory's ``actions.json``.

Usage:
    python -m thesis.analysis.trace_stats --traces-dir thesis/traces/gmail
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from collections import defaultdict


def load_task_summary(task_dir: Path) -> dict | None:
    summary_path = task_dir / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def load_path_actions(task_dir: Path) -> list[list[dict]]:
    out: list[list[dict]] = []
    traj_root = task_dir / "trajectories"
    if not traj_root.exists():
        return out
    for sub in sorted(traj_root.iterdir()):
        af = sub / "actions.json"
        if af.exists():
            try:
                out.append(json.loads(af.read_text()))
            except json.JSONDecodeError:
                pass
    return out


def action_key(a: dict) -> str:
    return f"{a.get('kind', '')}|{a.get('selector', '')}|{a.get('value') or ''}"


def levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance over action-key sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[n]


def compute_task_stats(task_dir: Path) -> dict:
    summary = load_task_summary(task_dir)
    if summary is None:
        return {"task_id": task_dir.name, "missing_summary": True}

    paths = load_path_actions(task_dir)
    n_paths = len(paths)

    out = {
        "task_id": summary["task_id"],
        "difficulty": summary.get("difficulty", ""),
        "n_passing_paths": summary.get("n_passing_paths", 0),
        "min_steps": summary.get("min_steps"),
        "max_steps": summary.get("max_steps"),
        "median_steps": summary.get("median_steps"),
        "n_unique_states": summary.get("n_unique_states", 0),
        "n_unique_action_keys": summary.get("n_unique_action_keys", 0),
        "elapsed_s": summary.get("elapsed_s", 0.0),
        "error": summary.get("error"),
    }

    # Edit-distance-based diversity: average pairwise Levenshtein over action keys.
    if n_paths >= 2:
        seqs = [[action_key(a) for a in p] for p in paths]
        dists = []
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                dists.append(levenshtein(seqs[i], seqs[j]))
        out["pairwise_edit_distance_mean"] = round(statistics.mean(dists), 2)
        out["pairwise_edit_distance_max"] = max(dists)
    else:
        out["pairwise_edit_distance_mean"] = None
        out["pairwise_edit_distance_max"] = None

    return out


def aggregate(traces_dir: Path) -> dict:
    per_task = []
    for sub in sorted(traces_dir.iterdir()):
        if not sub.is_dir():
            continue
        per_task.append(compute_task_stats(sub))

    by_diff: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "with_paths": 0,
        "min_steps": [],
        "median_steps": [],
        "n_paths": [],
        "edit_distance_mean": [],
    })

    for row in per_task:
        d = row.get("difficulty") or "unknown"
        bucket = by_diff[d]
        bucket["total"] += 1
        n_p = row.get("n_passing_paths", 0)
        if n_p > 0:
            bucket["with_paths"] += 1
            if row.get("min_steps") is not None:
                bucket["min_steps"].append(row["min_steps"])
            if row.get("median_steps") is not None:
                bucket["median_steps"].append(row["median_steps"])
            bucket["n_paths"].append(n_p)
            if row.get("pairwise_edit_distance_mean") is not None:
                bucket["edit_distance_mean"].append(row["pairwise_edit_distance_mean"])

    rollups = {}
    for d, b in by_diff.items():
        rollups[d] = {
            "total": b["total"],
            "with_paths": b["with_paths"],
            "success_rate": round(b["with_paths"] / b["total"], 3) if b["total"] else 0.0,
            "min_steps_avg":   round(statistics.mean(b["min_steps"]), 2)   if b["min_steps"]   else None,
            "median_steps_avg": round(statistics.mean(b["median_steps"]), 2) if b["median_steps"] else None,
            "n_paths_avg":     round(statistics.mean(b["n_paths"]), 2)     if b["n_paths"]     else None,
            "edit_distance_avg": round(statistics.mean(b["edit_distance_mean"]), 2) if b["edit_distance_mean"] else None,
        }

    return {"per_task": per_task, "by_difficulty": rollups}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir", required=True,
                   help="e.g. thesis/traces/gmail")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: <traces-dir>/_stats.json)")
    args = p.parse_args()
    traces_dir = Path(args.traces_dir)
    if not traces_dir.exists():
        print(f"traces dir not found: {traces_dir}", file=sys.stderr)
        return 2
    out_path = Path(args.out) if args.out else traces_dir / "_stats.json"

    agg = aggregate(traces_dir)
    out_path.write_text(json.dumps(agg, indent=2))

    # Pretty console summary.
    print(f"\nWrote {out_path}")
    print(f"  tasks scanned: {len(agg['per_task'])}")
    print(f"  per difficulty:")
    for d, r in agg["by_difficulty"].items():
        print(f"    {d:<8s}  total={r['total']:<3d}  "
              f"with_paths={r['with_paths']:<3d}  "
              f"sr={r['success_rate']:.0%}  "
              f"min_steps_avg={r['min_steps_avg']}  "
              f"n_paths_avg={r['n_paths_avg']}  "
              f"edit_dist_avg={r['edit_distance_avg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
