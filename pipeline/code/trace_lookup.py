"""Per-task `estimated_min_actions` source.

Priority: existing MCTS trace `summary.min_steps` if usable, else admissible
per-difficulty default. Defaults intentionally low to keep A* heuristic admissible.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "thesis" / "traces"

DIFFICULTY_DEFAULT: dict[str, int] = {"easy": 1, "medium": 2, "hard": 3}


def load_min_steps(app: str, task_id: str) -> int | None:
    """Return `summary.min_steps` from the MCTS trace if available, else None."""
    summary_path = TRACES_DIR / app / task_id / "summary.json"
    if not summary_path.exists():
        return None
    try:
        s = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None
    if s.get("error") is not None:
        return None
    n = s.get("min_steps")
    if isinstance(n, int) and n > 0:
        return n
    return None


def estimated_min_actions(app: str, task_id: str, difficulty: str) -> int:
    n = load_min_steps(app, task_id)
    if n is not None:
        return n
    return DIFFICULTY_DEFAULT.get(difficulty, 1)


if __name__ == "__main__":
    print("task_e1 (trace exists):", estimated_min_actions("gmail", "task_e1", "easy"))
    print("task_e2 (no trace, easy):", estimated_min_actions("gmail", "task_e2", "easy"))
    print("task_h1 (no trace, hard):", estimated_min_actions("gmail", "task_h1", "hard"))
