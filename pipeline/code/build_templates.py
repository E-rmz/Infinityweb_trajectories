"""Entry point: build pipeline/data/templates/<app>.json end-to-end.

Pipeline per app:
    Pass A  parse real-tasks.json
    Pass B  AST walk → state_reads
    Pass C  Stage A LLM → intent_skeleton + entities
    Pass D  Stage B LLM → effect_summary, affects_paths, effect_kind,
                          bound_entities, semantic_actions
    merge  → AtomicTemplate
    validators → <app>.validation.json
    seed-check (unless --skip-seed-check) → seed_satisfies in place
    write   → templates/<app>.json + vocabulary/<app>.unbound.json

CLI flags:
    --app apps/gmail
    --all-apps
    --ast-only         skip LLM, just sanity-check AST walker
    --no-llm           cache-only mode (fail loudly on miss)
    --skip-seed-check
    --only task_e1,task_e6
    --force-refresh    bust the LLM content-hash cache
    --batch-timeout 7200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from .introspect import extract_state_reads, parse_task_json
from .llm_extract import stage_a, stage_b
from .normalize import normalize_stage_b
from .schema import AtomicTemplate, wrap_envelope
from .seed_check import check_seeds_for_app
from .trace_lookup import estimated_min_actions
from .utils import atomic_write_json
from .validate import run_validators, write_validation_report

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DATA = ROOT / "pipeline" / "data"
TEMPLATES_DIR = PIPELINE_DATA / "templates"
VOCAB_DIR = PIPELINE_DATA / "vocabulary"
SCHEMA_MAP_PATH = PIPELINE_DATA / "state_schema_map.json"


def _load_schema_map() -> dict:
    return json.loads(SCHEMA_MAP_PATH.read_text())["apps"]


def _all_app_dirs() -> list[Path]:
    apps_root = ROOT / "apps"
    return sorted(
        p for p in apps_root.iterdir()
        if p.is_dir() and (p / "real-tasks.json").exists()
    )


def _ast_only(app_dirs: list[Path]) -> int:
    schema_map = _load_schema_map()
    print(f"AST pre-flight across {len(app_dirs)} apps\n")
    total = 0
    empty: list[tuple[str, str]] = []
    unknown: list[tuple[str, str, str]] = []
    exceptions: list[tuple[str, str, str]] = []
    for app_dir in app_dirs:
        name = app_dir.name
        if name not in schema_map:
            print(f"  {name:40s} <missing from state_schema_map>")
            continue
        known = set(schema_map[name]["state_keys"])
        tasks = parse_task_json(app_dir)
        n = len(tasks)
        total += n
        n_empty = 0
        n_unknown = 0
        for t in tasks:
            vpath = app_dir / t["verify"]
            try:
                reads = extract_state_reads(vpath)
            except Exception as e:
                exceptions.append((name, t["id"], repr(e)))
                continue
            if not reads:
                n_empty += 1
                empty.append((name, t["id"]))
            for k in reads:
                if k not in known:
                    n_unknown += 1
                    unknown.append((name, t["id"], k))
        print(
            f"  {name:40s} tasks={n:4d}  empty={n_empty:3d}  unknown_keys={n_unknown:3d}"
        )
    print()
    print(f"TOTAL tasks: {total}")
    print(f"empty state_reads: {len(empty)}")
    if empty[:5]:
        print(f"  first 5: {empty[:5]}")
    print(f"reads outside schema_map: {len(unknown)}")
    if unknown[:5]:
        print(f"  first 5: {unknown[:5]}")
    print(f"AST exceptions: {len(exceptions)}")
    if exceptions[:5]:
        print(f"  first 5: {exceptions[:5]}")
    return 1 if exceptions else 0


def _build_one_template(
    app: str,
    app_schema: dict,
    app_dir: Path,
    entry: dict,
    *,
    no_llm: bool,
    force_refresh: bool,
) -> AtomicTemplate:
    vpath = app_dir / entry["verify"]
    verifier_src = vpath.read_text()
    reads = extract_state_reads(vpath)

    paraphrase = stage_a(
        entry["instruction"], no_llm=no_llm, force_refresh=force_refresh
    )
    raw_analysis = stage_b(
        entry["instruction"], verifier_src, app_schema,
        no_llm=no_llm, force_refresh=force_refresh,
    )
    analysis, _norm_report = normalize_stage_b(raw_analysis)

    return AtomicTemplate(
        task_id=entry["id"],
        app=app,
        difficulty=entry["difficulty"],
        intent_raw=entry["instruction"],
        intent_skeleton=paraphrase.get("intent_skeleton", ""),
        entities=paraphrase.get("entities", {}),
        verifier_path=entry["verify"],
        state_reads=reads,
        effect_summary=analysis.get("effect_summary", ""),
        affects_paths=analysis.get("affects_paths", []),
        effect_kind=analysis.get("effect_kind", ""),
        bound_entities=analysis.get("bound_entities", {}),
        semantic_actions=analysis.get("semantic_actions", []),
        estimated_min_actions=estimated_min_actions(
            app, entry["id"], entry["difficulty"]
        ),
    )


def _write_vocab_scaffold(app: str, templates: list[AtomicTemplate]) -> Path:
    used_in: dict[str, list[str]] = defaultdict(list)
    for t in templates:
        for a in t.semantic_actions:
            used_in[a].append(t.task_id)
    scaffold = {
        a: {"used_in": sorted(used_in[a]), "selectors": []}
        for a in sorted(used_in)
    }
    out = VOCAB_DIR / f"{app}.unbound.json"
    atomic_write_json(out, scaffold)
    return out


def _build_for_app(
    app_dir: Path,
    schema_map: dict,
    *,
    only_ids: set[str] | None,
    no_llm: bool,
    force_refresh: bool,
    skip_seed_check: bool,
    batch_timeout: int,
) -> dict:
    app = app_dir.name
    if app not in schema_map:
        raise KeyError(f"{app}: missing from pipeline/data/state_schema_map.json")
    app_schema = schema_map[app]
    raw = parse_task_json(app_dir)
    if only_ids is not None:
        raw = [r for r in raw if r["id"] in only_ids]
    print(f"[{app}] building {len(raw)} templates", flush=True)
    t0 = time.time()
    templates: list[AtomicTemplate] = []
    for i, entry in enumerate(raw, 1):
        templates.append(
            _build_one_template(
                app, app_schema, app_dir, entry,
                no_llm=no_llm, force_refresh=force_refresh,
            )
        )
        if i % 10 == 0 or i == len(raw):
            print(f"  [{app}] {i}/{len(raw)}  elapsed={time.time()-t0:.1f}s",
                  flush=True)

    report = run_validators(templates, app_schema)
    report_path = write_validation_report(TEMPLATES_DIR, app, report)
    print(f"  [{app}] validation: {report['summary']}", flush=True)

    if not skip_seed_check:
        print(f"  [{app}] seed check (one server boot + headless chrome)…",
              flush=True)
        seed_out = check_seeds_for_app(
            app_dir, templates, batch_timeout=batch_timeout
        )
        for t in templates:
            row = seed_out.get(t.task_id)
            if row is not None:
                t.seed_satisfies = bool(row["passed"])
        degen = [t.task_id for t in templates if t.seed_satisfies]
        if degen:
            print(f"  [{app}] WARNING: {len(degen)} degenerate (seed-pass) "
                  f"tasks: {degen[:8]}{'…' if len(degen)>8 else ''}",
                  flush=True)

    envelope = wrap_envelope(app, templates)
    out_json = TEMPLATES_DIR / f"{app}.json"
    atomic_write_json(out_json, envelope)
    vocab_path = _write_vocab_scaffold(app, templates)

    elapsed = time.time() - t0
    print(f"  [{app}] wrote {out_json.relative_to(ROOT)}", flush=True)
    print(f"  [{app}] wrote {vocab_path.relative_to(ROOT)}", flush=True)
    print(f"  [{app}] wrote {report_path.relative_to(ROOT)}", flush=True)
    print(f"  [{app}] done in {elapsed:.1f}s", flush=True)
    return {
        "app": app,
        "n_templates": len(templates),
        "n_failing_templates": report["summary"]["n_failing_templates"],
        "elapsed_s": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path,
        help="Path to a single app dir (e.g. apps/gmail)")
    parser.add_argument("--all-apps", action="store_true")
    parser.add_argument("--ast-only", action="store_true")
    parser.add_argument("--no-llm", action="store_true",
        help="Use cached LLM responses only — fail on cache miss")
    parser.add_argument("--skip-seed-check", action="store_true")
    parser.add_argument("--only", default="",
        help="Comma-separated task ids to restrict to")
    parser.add_argument("--force-refresh", action="store_true",
        help="Bust the content-hash LLM cache")
    parser.add_argument("--batch-timeout", type=int, default=7200,
        help="Seed-check wall-clock ceiling, seconds (default 2 hr)")
    args = parser.parse_args(argv)

    if args.ast_only:
        targets = _all_app_dirs() if args.all_apps else (
            [args.app] if args.app else _all_app_dirs())
        return _ast_only(targets)

    if not args.app and not args.all_apps:
        parser.error("--app PATH or --all-apps is required")

    targets = _all_app_dirs() if args.all_apps else [args.app]
    only_ids = (
        {s.strip() for s in args.only.split(",") if s.strip()}
        if args.only else None
    )
    schema_map = _load_schema_map()
    summary: list[dict] = []
    for app_dir in targets:
        summary.append(
            _build_for_app(
                app_dir, schema_map,
                only_ids=only_ids,
                no_llm=args.no_llm,
                force_refresh=args.force_refresh,
                skip_seed_check=args.skip_seed_check,
                batch_timeout=args.batch_timeout,
            )
        )
    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['app']:40s} templates={s['n_templates']:4d}  "
              f"failing={s['n_failing_templates']:3d}  "
              f"elapsed={s['elapsed_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
