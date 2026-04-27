"""Verifier loading + invocation.

The existing repo ships standalone verifiers under
``apps/<app>/real-tasks/task_*.py`` that read ``GET /api/state`` and
return ``(passed: bool, message: str)``. We reuse them verbatim — the
only thing this module does is dynamically import a verifier file and
invoke its ``verify(server_url)`` function safely.

The function never raises: any verifier exception is converted to
``(False, "Verifier exception: ...")`` so the MCTS loop keeps running.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable


VerifierFn = Callable[[str], tuple[bool, str]]


def load_verifier(web_app_dir: str | Path, verify_path: str) -> VerifierFn:
    """Load the ``verify`` function from ``<web_app_dir>/<verify_path>``.

    This mirrors ``evaluation/tasks.py::load_verifier`` so behaviour is
    identical to the existing harness.
    """
    full = Path(web_app_dir) / verify_path
    spec = importlib.util.spec_from_file_location(f"verifier_{full.stem}", str(full))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load verifier: {full}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "verify"):
        raise AttributeError(f"Verifier {full} has no `verify` function")
    return mod.verify


def safe_verify(verify_fn: VerifierFn, server_url: str) -> tuple[bool, str]:
    """Run a verifier, catching every exception."""
    try:
        passed, message = verify_fn(server_url)
        return bool(passed), str(message)
    except Exception as e:
        return False, f"Verifier exception: {type(e).__name__}: {e}"
