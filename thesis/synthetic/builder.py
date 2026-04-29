"""Pipeline orchestrator: trajectories → synthetic tasks (with validation).

For one ``(env, task_id)`` pair:

1. Walk ``thesis/traces/<env>/<task_id>/trajectories/path_*``
2. Read each path's cached ``states/`` to compute a ``StateDiff``
3. Skip empty diffs and signature-duplicates
4. Render a ``task.py`` (instruction + verify + solver + provenance)
5. Run the 3-arm ``Validator`` on each generated task
6. Write artifacts to ``thesis/synthetic-tasks/<env>/<task_id>/path_NNN/``
7. Emit a per-task ``index.json`` with the validation summary
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from .codegen import render_task_module
from .diff import StateDiff, diff_states
from .instruction import template_from_actions
from .state_capture import load_actions, load_cached_states
from .validator import Validator


@dataclass
class Candidate:
    path_dir: Path
    diff: StateDiff
    seed: dict
    instruction: str
    source: dict


# ---------------------------------------------------------------------------
# Phase 1 — collect & dedup candidates (no server needed)
# ---------------------------------------------------------------------------


def collect_candidates(
    *,
    env: str,
    task_id: str,
    traces_dir: Path,
    only_path: str | None = None,
    log=print,
) -> list[Candidate]:
    """Scan a task's trajectories dir, return dedup'd candidates."""
    paths_dir = traces_dir / task_id / "trajectories"
    if not paths_dir.exists():
        log(f"  no trajectories dir at {paths_dir}")
        return []

    path_dirs = sorted(p for p in paths_dir.glob("path_*") if p.is_dir())
    if only_path:
        path_dirs = [p for p in path_dirs if p.name == only_path]

    seen: set[str] = set()
    out: list[Candidate] = []
    for pd in path_dirs:
        cached = load_cached_states(pd)
        if cached is None:
            log(f"  {pd.name}: no cached states/, skipping")
            continue
        seed, terminal, _ = cached
        d = diff_states(seed, terminal)
        if d.is_empty():
            log(f"  {pd.name}: empty diff (no-op), skipping")
            continue
        sig = d.signature()
        if sig in seen:
            log(f"  {pd.name}: duplicate signature, skipping")
            continue
        seen.add(sig)
        actions_raw = json.loads((pd / "actions.json").read_text())
        instruction = template_from_actions(actions_raw)
        verify_meta_path = pd / "verify.json"
        verify_meta = (
            json.loads(verify_meta_path.read_text())
            if verify_meta_path.exists()
            else {}
        )
        source = {
            "env": env,
            "task_id": task_id,
            "path_id": pd.name,
            "state_hash": verify_meta.get("state_hash"),
        }
        out.append(
            Candidate(path_dir=pd, diff=d, seed=seed, instruction=instruction, source=source)
        )
    return out


# ---------------------------------------------------------------------------
# Phase 2 — render + validate
# ---------------------------------------------------------------------------


def write_artifacts(candidate: Candidate, out_dir: Path) -> Path:
    """Write task.py, actions.json, source.json, diff.json. Returns the dir."""
    target = out_dir / candidate.source["path_id"]
    target.mkdir(parents=True, exist_ok=True)
    src = render_task_module(
        diff=candidate.diff,
        seed_state=candidate.seed,
        instruction=candidate.instruction,
        source=candidate.source,
    )
    (target / "task.py").write_text(src)
    (target / "actions.json").write_text(
        (candidate.path_dir / "actions.json").read_text()
    )
    (target / "source.json").write_text(json.dumps(candidate.source, indent=2))
    (target / "diff.json").write_text(
        json.dumps(candidate.diff.to_dict(), indent=2, default=str)
    )
    return target


def _load_generated_module(target: Path):
    spec = importlib.util.spec_from_file_location(
        f"synth_{target.parent.name}_{target.name}", target / "task.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load task module at {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Phase 3 — top-level orchestrator
# ---------------------------------------------------------------------------


async def build_for_task(
    *,
    env: str,
    task_id: str,
    traces_dir: Path,
    out_dir: Path,
    web_app_dir: Path,
    port: int = 8765,
    only_path: str | None = None,
    headless: bool = True,
    force: bool = False,
    log=print,
) -> dict:
    candidates = collect_candidates(
        env=env, task_id=task_id, traces_dir=traces_dir, only_path=only_path, log=log
    )
    log(f"  {len(candidates)} candidate(s) after dedup")
    if not candidates:
        return {"env": env, "task_id": task_id, "n_candidates": 0, "n_valid": 0, "results": []}

    # Filter out ones already built (unless force).
    if not force:
        kept: list[Candidate] = []
        for c in candidates:
            target = out_dir / c.source["path_id"]
            if (target / "validation.json").exists():
                log(f"  {c.source['path_id']}: already built, skipping (use --force to redo)")
                continue
            kept.append(c)
        candidates = kept
        if not candidates:
            return {"env": env, "task_id": task_id, "n_candidates": 0, "n_valid": 0, "results": []}

    results: list[dict] = []
    async with Validator(web_app_dir=web_app_dir, port=port, headless=headless) as v:
        for c in candidates:
            target = write_artifacts(c, out_dir)
            mod = _load_generated_module(target)
            actions = load_actions(c.path_dir)
            try:
                vr = await v.validate(
                    solver_fn=mod.solver, verify_fn=mod.verify, actions=actions
                )
            except Exception as e:
                log(f"  {c.source['path_id']}: validator crashed: {e}")
                vr_dict = {
                    "valid": False,
                    "golden_pass": False,
                    "seed_fail": False,
                    "replay_pass": False,
                    "messages": {"error": f"{type(e).__name__}: {e}"},
                }
                (target / "validation.json").write_text(json.dumps(vr_dict, indent=2))
                results.append({"path_id": c.source["path_id"], **vr_dict})
                continue
            (target / "validation.json").write_text(json.dumps(vr.to_dict(), indent=2))
            log(
                f"  {c.source['path_id']}: golden={vr.golden.passed} "
                f"seed_fail={vr.seed.passed} replay={vr.replay.passed} → "
                f"{'VALID' if vr.valid else 'INVALID'}"
            )
            results.append({"path_id": c.source["path_id"], **vr.to_dict()})

    summary = {
        "env": env,
        "task_id": task_id,
        "n_candidates": len(candidates) + (
            len(collect_candidates(env=env, task_id=task_id, traces_dir=traces_dir, log=lambda *a, **k: None))
            - len(candidates)
        ),
        "n_built": len(candidates),
        "n_valid": sum(1 for r in results if r.get("valid")),
        "results": results,
    }
    (out_dir / "index.json").write_text(json.dumps(summary, indent=2))
    return summary
