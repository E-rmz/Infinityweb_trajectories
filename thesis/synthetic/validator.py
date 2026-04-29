"""Three-arm validator for generated synthetic tasks.

Mirrors the "golden path" pattern from ``apps/gmail/sanity_check_real.py``:
the verifier is only trusted if a hand-shaped end-state passes, the
unmodified seed fails, and a real browser replay also passes.

* **golden_pass** — apply ``solver`` to a fresh copy of the seed, PUT it,
  run ``verify`` → must pass. Catches solver/verifier drift.
* **seed_fail**  — reset, do nothing, run ``verify`` → must fail. Catches
  tautological verifiers that return True regardless of state.
* **replay_pass** — reset, drive Playwright through ``actions``, run
  ``verify`` → must pass. The exact path the agent will face at eval time.

A task is ``valid`` only when all three arms succeed.
"""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

from ..mcts.action import Action
from ..mcts.browser import BrowserSession, exec_action
from ..mcts.replay import reset_to_seed

# Reuse the existing server lifecycle helpers (read-only import).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "evaluation"))
from server import start_server, stop_server, wait_for_server  # noqa: E402


VerifierFn = Callable[[str], tuple[bool, str]]
SolverFn = Callable[[dict], None]


@dataclass
class ArmResult:
    passed: bool
    message: str

    def to_dict(self) -> dict:
        return {"passed": self.passed, "message": self.message}


@dataclass
class ValidationResult:
    golden: ArmResult
    seed: ArmResult
    replay: ArmResult

    @property
    def valid(self) -> bool:
        return self.golden.passed and self.seed.passed and self.replay.passed

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "golden_pass": self.golden.passed,
            "seed_fail": self.seed.passed,
            "replay_pass": self.replay.passed,
            "messages": {
                "golden": self.golden.message,
                "seed": self.seed.message,
                "replay": self.replay.message,
            },
        }


# ---------------------------------------------------------------------------
# Validator session
# ---------------------------------------------------------------------------


class Validator:
    """Owns one server + one browser for the lifetime of many validations.

    Use as an async context manager::

        async with Validator(web_app_dir="apps/gmail", port=8765) as v:
            result = await v.validate(
                solver_fn=mod.solver,
                verify_fn=mod.verify,
                actions=actions,
            )
    """

    def __init__(self, *, web_app_dir: str | Path, port: int = 8765, headless: bool = True):
        self.web_app_dir = str(Path(web_app_dir).resolve())
        self.port = port
        self.server_url = f"http://localhost:{port}"
        self._headless = headless
        self._proc = None
        self._session: BrowserSession | None = None
        self.seed_state: dict | None = None

    async def __aenter__(self) -> "Validator":
        self._proc = start_server(self.web_app_dir, self.port)
        if not wait_for_server(self.port):
            raise RuntimeError(f"server failed to come up on :{self.port}")
        self._session = BrowserSession(headless=self._headless)
        await self._session.start(self.server_url)
        # The browser's initial PUT has now landed; capture seed.
        self.seed_state = self._fetch_state()
        return self

    async def __aexit__(self, *_exc) -> None:
        try:
            if self._session is not None:
                await self._session.stop()
        finally:
            if self._proc is not None:
                stop_server(self._proc)

    # ----- arms -----

    async def validate(
        self,
        *,
        solver_fn: SolverFn,
        verify_fn: VerifierFn,
        actions: list[Action],
    ) -> ValidationResult:
        return ValidationResult(
            golden=await self._golden(solver_fn, verify_fn),
            seed=await self._seed_fail(verify_fn),
            replay=await self._replay(actions, verify_fn),
        )

    async def _golden(self, solver_fn: SolverFn, verify_fn: VerifierFn) -> ArmResult:
        try:
            assert self._session is not None and self.seed_state is not None
            await reset_to_seed(self._session.page, self.server_url, self.seed_state)
            modified = copy.deepcopy(self.seed_state)
            solver_fn(modified)
            r = requests.put(f"{self.server_url}/api/state", json=modified, timeout=5)
            r.raise_for_status()
            ok, msg = _safe_verify(verify_fn, self.server_url)
            if ok:
                return ArmResult(True, msg)
            return ArmResult(False, f"verifier rejected solved state: {msg}")
        except Exception as e:
            return ArmResult(False, f"{type(e).__name__}: {e}")

    async def _seed_fail(self, verify_fn: VerifierFn) -> ArmResult:
        try:
            assert self._session is not None and self.seed_state is not None
            await reset_to_seed(self._session.page, self.server_url, self.seed_state)
            ok, msg = _safe_verify(verify_fn, self.server_url)
            if not ok:
                return ArmResult(True, f"verifier correctly rejected seed: {msg}")
            return ArmResult(False, f"verifier passed on seed (tautology): {msg}")
        except Exception as e:
            return ArmResult(False, f"{type(e).__name__}: {e}")

    async def _replay(self, actions: list[Action], verify_fn: VerifierFn) -> ArmResult:
        try:
            assert self._session is not None and self.seed_state is not None
            await reset_to_seed(self._session.page, self.server_url, self.seed_state)
            for a in actions:
                await exec_action(self._session.page, a)
            ok, msg = _safe_verify(verify_fn, self.server_url)
            if ok:
                return ArmResult(True, msg)
            return ArmResult(False, f"verifier rejected replay terminal: {msg}")
        except Exception as e:
            return ArmResult(False, f"{type(e).__name__}: {e}")

    # ----- helpers -----

    def _fetch_state(self) -> dict:
        # Server returns 404 until the browser's first PUT lands; retry briefly.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            r = requests.get(f"{self.server_url}/api/state", timeout=2)
            if r.status_code == 200:
                return r.json()
            time.sleep(0.15)
        r.raise_for_status()
        return r.json()


def _safe_verify(verify_fn: VerifierFn, server_url: str) -> tuple[bool, str]:
    try:
        passed, message = verify_fn(server_url)
        return bool(passed), str(message)
    except Exception as e:
        return False, f"verifier exception: {type(e).__name__}: {e}"
