"""Standalone seed-check entry point.

Re-reads pipeline/data/templates/<app>.json, boots the app server once,
runs every verifier, and rewrites the envelope with updated `seed_satisfies`
fields. Lets us refresh seed-satisfaction without paying for LLM extraction.

Usage:
    python -m pipeline.code.check_seed --app apps/gmail
    python -m pipeline.code.check_seed --all-apps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .schema import load_templates, wrap_envelope
from .seed_check import check_seeds_for_app
from .utils import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = ROOT / "pipeline" / "data" / "templates"


def _run_for_app(app_dir: Path, *, batch_timeout: int) -> dict:
    app = app_dir.name
    out_json = TEMPLATES_DIR / f"{app}.json"
    if not out_json.exists():
        raise FileNotFoundError(
            f"{out_json} not found — run build_templates.py first"
        )
    _, templates = load_templates(out_json)
    t0 = time.time()
    seed_out = check_seeds_for_app(
        app_dir, templates, batch_timeout=batch_timeout
    )
    for t in templates:
        row = seed_out.get(t.task_id)
        if row is not None:
            t.seed_satisfies = bool(row["passed"])
    envelope = wrap_envelope(app, templates)
    atomic_write_json(out_json, envelope)
    degen = [t.task_id for t in templates if t.seed_satisfies]
    print(f"[{app}] wrote {out_json.relative_to(ROOT)}  "
          f"degenerate={len(degen)}  elapsed={time.time()-t0:.1f}s")
    return {"app": app, "n_degenerate": len(degen)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path)
    parser.add_argument("--all-apps", action="store_true")
    parser.add_argument("--batch-timeout", type=int, default=7200)
    args = parser.parse_args(argv)

    if args.all_apps:
        apps_root = ROOT / "apps"
        targets = sorted(
            p for p in apps_root.iterdir()
            if p.is_dir() and (TEMPLATES_DIR / f"{p.name}.json").exists()
        )
    elif args.app:
        targets = [args.app]
    else:
        parser.error("--app PATH or --all-apps is required")

    for app_dir in targets:
        _run_for_app(app_dir, batch_timeout=args.batch_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
