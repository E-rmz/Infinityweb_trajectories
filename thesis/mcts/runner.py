"""Per-task orchestrator: spawn server, run MCTS, persist trajectories.

The runner ties the rest of the package together for a single task.
It is responsible for:

  * spawning ``apps/<app>/server.py`` (we reuse the helpers from
    ``evaluation/server.py`` — no duplication);
  * launching one Playwright Chromium tab and navigating to the app;
  * loading the task's standalone verifier;
  * running ``run_mcts`` from ``search.py``;
  * for each passing path, replaying it once more to capture per-step
    state snapshots and screenshots;
  * writing ``tree.json``, ``summary.json``, ``log.txt`` and the
    ``trajectories/path_NNN/`` directories.

It does not modify any file in the existing repo. The only thing it
imports from the existing codebase is the read-only server-lifecycle
helpers in ``evaluation/server.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Reuse the existing server lifecycle helpers (read-only import).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "evaluation"))
from server import start_server, stop_server, wait_for_server  # noqa: E402

from .browser import BrowserSession, exec_action  # noqa: E402
from .replay import reset_to_seed, state_hash  # noqa: E402
from .reward import load_verifier, safe_verify  # noqa: E402
from .search import (  # noqa: E402
    MCTSConfig,
    MCTSResult,
    PassingPath,
    run_mcts,
    serialize_tree,
)
from .action import Action  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_state(server_url: str) -> dict:
    r = requests.get(f"{server_url}/api/state", timeout=3)
    r.raise_for_status()
    return r.json()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_task(
    *,
    task: dict,
    web_app_dir: str | Path,
    output_dir: str | Path,
    port: int = 8765,
    config: MCTSConfig | None = None,
    capture_screenshots: bool = True,
    save_states: bool = True,
    quiet: bool = False,
) -> dict:
    """Run MCTS on one task and persist all artifacts under ``output_dir``.

    Returns a small summary dict (also written to ``summary.json``).
    """
    web_app_dir = str(Path(web_app_dir).resolve())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        log_lines.append(line)
        if not quiet:
            print(line, flush=True)

    config = config or MCTSConfig()
    log(f"task={task['id']} difficulty={task.get('difficulty', '?')} "
        f"budget={config.budget} max_depth={config.max_depth} top_k={config.top_k}")
    log(f"instruction: {task['instruction']}")

    server_url = f"http://localhost:{port}"
    server_proc = None
    session = BrowserSession(headless=True)
    summary: dict = {
        "task_id": task["id"],
        "difficulty": task.get("difficulty", ""),
        "instruction": task["instruction"],
        "config": {
            "budget": config.budget,
            "max_depth": config.max_depth,
            "top_k": config.top_k,
            "sim_steps": config.sim_steps,
            "ucb_c": config.ucb_c,
            "seed": config.seed,
        },
        "n_passing_paths": 0,
        "min_steps": None,
        "max_steps": None,
        "median_steps": None,
        "n_unique_states": 0,
        "n_unique_action_keys": 0,
        "elapsed_s": 0.0,
        "error": None,
    }

    try:
        # ----- start server -----
        log(f"starting server on port {port} for {web_app_dir}")
        server_proc = start_server(web_app_dir, port)
        if not wait_for_server(port):
            raise RuntimeError(f"server failed to come up on :{port}")

        # ----- start browser -----
        log("launching browser")
        await session.start(server_url)
        page = session.page
        assert page is not None

        # ----- capture seed snapshot -----
        seed_state = _fetch_state(server_url)
        log(f"seed state captured (hash={state_hash(seed_state)[:8]})")

        # ----- load verifier -----
        verify_fn = load_verifier(web_app_dir, task["verify"])

        def _run_verifier() -> tuple[bool, str]:
            return safe_verify(verify_fn, server_url)

        # ----- run MCTS -----
        log("running MCTS...")
        t0 = time.time()
        result: MCTSResult = await run_mcts(
            instruction=task["instruction"],
            page=page,
            server_url=server_url,
            expected_seed=seed_state,
            get_state=lambda: _fetch_state(server_url),
            run_verifier=_run_verifier,
            config=config,
            log=log,
        )
        log(f"MCTS done in {time.time() - t0:.1f}s — paths={len(result.passing_paths)}")

        # ----- persist tree.json -----
        _write_json(output_dir / "tree.json", serialize_tree(result.root))

        # ----- replay each passing path to capture states/screenshots -----
        if result.passing_paths:
            log(f"persisting {len(result.passing_paths)} passing trajector(y/ies)")
            await _persist_trajectories(
                paths=result.passing_paths,
                page=page,
                session=session,
                server_url=server_url,
                expected_seed=seed_state,
                output_dir=output_dir,
                save_states=save_states,
                capture_screenshots=capture_screenshots,
                log=log,
            )
        else:
            log("no passing trajectories within budget")

        # ----- summary -----
        depths = [p.depth for p in result.passing_paths]
        all_action_keys: set[str] = set()
        for p in result.passing_paths:
            for a in p.actions:
                all_action_keys.add(a.key())
        summary.update(
            n_passing_paths=len(result.passing_paths),
            min_steps=min(depths) if depths else None,
            max_steps=max(depths) if depths else None,
            median_steps=(sorted(depths)[len(depths) // 2] if depths else None),
            n_unique_states=len(result.visited_state_hashes),
            n_unique_action_keys=len(all_action_keys),
            elapsed_s=round(result.elapsed, 2),
        )
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        log(f"ERROR: {summary['error']}")
    finally:
        try:
            await session.stop()
        except Exception:
            pass
        if server_proc is not None:
            try:
                stop_server(server_proc)
            except Exception:
                pass
        log_path.write_text("\n".join(log_lines) + "\n")
        _write_json(output_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Trajectory persistence
# ---------------------------------------------------------------------------


async def _persist_trajectories(
    *,
    paths: list[PassingPath],
    page,
    session: BrowserSession,
    server_url: str,
    expected_seed: dict,
    output_dir: Path,
    save_states: bool,
    capture_screenshots: bool,
    log,
) -> None:
    traj_root = output_dir / "trajectories"
    traj_root.mkdir(parents=True, exist_ok=True)

    # Sort by length, then by rollout discovered, so path_001 is the shortest.
    paths_sorted = sorted(paths, key=lambda p: (p.depth, p.found_in_rollout))

    for i, p in enumerate(paths_sorted, start=1):
        sub = traj_root / f"path_{i:03d}"
        sub.mkdir(exist_ok=True)
        (sub / "actions.json").write_text(
            json.dumps([a.to_dict() for a in p.actions], indent=2)
        )
        (sub / "verify.json").write_text(json.dumps({
            "passed": True,
            "message": p.verifier_msg,
            "terminal_depth": p.depth,
            "state_hash": p.state_hash,
            "found_in_rollout": p.found_in_rollout,
        }, indent=2))

        if not (save_states or capture_screenshots):
            continue

        # Replay this path to capture per-step artifacts.
        try:
            await reset_to_seed(page, server_url, expected_seed)
        except Exception as e:
            log(f"  path {i:03d}: reset failed before replay: {e}")
            continue
        if save_states:
            states_dir = sub / "states"
            states_dir.mkdir(exist_ok=True)
            try:
                _dump_state(states_dir / "step_000.json", _fetch_state(server_url))
            except Exception:
                pass
        if capture_screenshots:
            shots_dir = sub / "screenshots"
            shots_dir.mkdir(exist_ok=True)
            await session.screenshot(shots_dir / "step_000.png")

        for step_idx, action in enumerate(p.actions, start=1):
            try:
                await exec_action(page, action)
            except Exception as e:
                log(f"  path {i:03d} step {step_idx}: replay failed ({e}) — stopping replay")
                break
            if save_states:
                try:
                    _dump_state(sub / "states" / f"step_{step_idx:03d}.json",
                                _fetch_state(server_url))
                except Exception:
                    pass
            if capture_screenshots:
                await session.screenshot(sub / "screenshots" / f"step_{step_idx:03d}.png")


def _dump_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True))
