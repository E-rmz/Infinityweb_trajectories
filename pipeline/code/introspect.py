"""Pass A (real-tasks.json parse) and Pass B (verifier AST walk).

Top-level state keys only — see step_0_deep_breakdown §4.2.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


def parse_task_json(app_dir: Path) -> list[dict]:
    """Pass A: read `<app_dir>/real-tasks.json` as-is."""
    app_dir = Path(app_dir)
    return json.loads((app_dir / "real-tasks.json").read_text())


class StateReadVisitor(ast.NodeVisitor):
    """Capture the top-level state keys a verifier reads.

    Detects:
      - `state["emails"]` and `state["emails"][...]`-style subscript chains
      - `state.get("emails")` and `state.get("emails", default)`

    Returns only the **top-level** key (e.g. "emails", not "emails.isStarred")
    because verifiers typically bind a local from `state[K]` then index it,
    and we don't follow bindings. Field-level granularity lives in Pass D's
    `affects_paths` field.
    """

    def __init__(self) -> None:
        self.reads: list[str] = []

    def visit_Subscript(self, node: ast.Subscript) -> None:
        chain: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Subscript):
            if isinstance(cur.slice, ast.Constant):
                chain.insert(0, str(cur.slice.value))
            else:
                chain.insert(0, "*")
            cur = cur.value
        if isinstance(cur, ast.Name) and cur.id == "state" and chain:
            self.reads.append(chain[0])
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "state"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            self.reads.append(str(node.args[0].value))
        self.generic_visit(node)


def extract_state_reads(verifier_path: Path) -> list[str]:
    """Pass B: return sorted unique top-level state keys the verifier reads."""
    tree = ast.parse(Path(verifier_path).read_text())
    visitor = StateReadVisitor()
    visitor.visit(tree)
    return sorted(set(visitor.reads))


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[2]
    gmail_dir = here / "apps" / "gmail"
    for tid in ("task_e1", "task_e6", "task_h1"):
        path = gmail_dir / "real-tasks" / f"{tid}.py"
        if path.exists():
            print(f"{tid}: {extract_state_reads(path)}")
        else:
            print(f"{tid}: <missing>")
