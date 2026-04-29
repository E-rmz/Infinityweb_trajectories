# Synthetic Task Generator — Roadmap & How-To Guide

This guide explains the synthetic-task pipeline that turns MCTS trajectories
into self-contained synthetic tasks (instruction + verifier + solver), and
how to apply it to **new tasks**, **new MCTS runs**, and **new
environments**.

The pipeline is deterministic, LLM-free, and resume-aware. Generated tasks
are validated by a 3-arm test that mirrors the original repo's golden-path
sanity check.

---

## 1. Pipeline at a Glance

```
                MCTS run (existing)                synthetic pipeline (new)
                ────────────────────               ────────────────────────
   apps/<env>     thesis/scripts/run_one_task.py     thesis/scripts/build_synthetic.py
   real-tasks/    └── runner.py + search.py          └── builder.py
       │                  │                                │
       ▼                  ▼                                ▼
   verifiers       thesis/traces/<env>/<task_id>/     thesis/synthetic-tasks/<env>/<task_id>/
                   ├── tree.json                      └── path_NNN/
                   ├── summary.json                       ├── task.py        ← generated
                   └── trajectories/path_NNN/             ├── actions.json   ← copied
                       ├── actions.json                   ├── source.json    ← provenance
                       ├── verify.json                    ├── diff.json      ← human-readable
                       └── states/step_*.json             └── validation.json ← 3-arm result
```

**Inputs the pipeline needs from MCTS:**
- `trajectories/path_NNN/actions.json` — action sequence (always saved)
- `trajectories/path_NNN/states/step_*.json` — per-step `/api/state` snapshots

The state snapshots are produced when MCTS runs with `save_states=True`
(the default in `runner.py`). If a path lacks a `states/` dir, it is
skipped during candidate collection.

---

## 2. Current Verified Coverage (gmail)

| Source task | Paths | Built | Valid | Notes |
|---|---|---|---|---|
| `task_e1.bak` | 16 | 16 | **16/16** | Star variants — see `index.json` |
| `task_e6` | 1 | 1 | **1/1** | Dark-mode toggle |

Both negative-control and positive-control checks pass. The pipeline is
**verified for production use on gmail** at the structural-state level.

---

## 3. How to Generate Synthetic Tasks for a New `(env, task_id)`

### 3.1 Prerequisites

1. The environment must be in `apps/<env>/` and run via
   `python3 server.py --port <port>`.
2. The original task verifier must live in
   `apps/<env>/real-tasks/<task_id>.py` and pass on its expected solution.
3. MCTS must have been run for that task: `thesis/traces/<env>/<task_id>/`
   exists with `summary.json` and at least one `trajectories/path_NNN/`.

### 3.2 Single-task command

```bash
cd /Users/ehsan/Desktop/Thesis/webinfinity/webarena-infinity

python -m thesis.scripts.build_synthetic \
    --env <env> \
    --task-id <task_id> \
    --port 8765
```

**Flags:**

| Flag | Purpose |
|---|---|
| `--env` | Sub-dir under `thesis/traces/` and `apps/` (e.g. `gmail`) |
| `--task-id` | Sub-dir under `traces/<env>/` (e.g. `task_e1`) |
| `--only-path` | Build a single `path_NNN` (debug only) |
| `--port` | Server port; pick something free if running concurrently |
| `--force` | Re-validate even if `validation.json` exists |
| `--no-headless` | Show the Playwright browser (for debugging) |
| `--traces-dir` | Override input dir (default `thesis/traces/<env>`) |
| `--out-dir` | Override output dir |
| `--web-app-dir` | Override app dir (default `apps/<env>`) |

### 3.3 Outputs

For each kept candidate `(env, task_id, path_NNN)`:

```
thesis/synthetic-tasks/<env>/<task_id>/path_NNN/
├── task.py          # INSTRUCTION + verify(server_url) + solver(state)
├── actions.json     # exact replay of the source path
├── source.json      # {env, task_id, path_id, state_hash}
├── diff.json        # the StateDiff that produced verify+solver
└── validation.json  # {valid, golden_pass, seed_fail, replay_pass, messages}
```

Plus a per-task aggregate `index.json` summarising n_built / n_valid.

### 3.4 Reading `validation.json`

A task is `valid: true` only when **all three arms** pass:

| Arm | What it proves |
|---|---|
| `golden_pass` | Solver mutates seed in a way the verifier accepts. Catches solver/verifier *drift*. |
| `seed_fail` | Verifier rejects the unmodified seed. Catches *tautological* verifiers (always-true). |
| `replay_pass` | Re-running the original action sequence in a real browser still satisfies the verifier. Catches *non-replayable* tasks. |

If any arm fails, the artifact is still written and the failure is captured
in `messages` for inspection.

---

## 4. How to Generate Trajectories for a New Task First

Use this when there is no MCTS run yet for `(env, task_id)`.

```bash
python -m thesis.scripts.run_one_task \
    --web-app apps/<env> \
    --task-id <task_id> \
    --budget 40 --max-depth 8 --top-k 6 --sim-steps 4
```

**Practical defaults from `CLAUDE.md`:**

| Difficulty | `--budget` | `--max-depth` | `--top-k` |
|---|---|---|---|
| Easy | 30 | 4 | 6 |
| Medium / Hard | 40 | 8 | 5 |

**Diversity mode (more long-horizon trajectories):**
`--budget 80 --max-depth 10 --top-k 6 --sim-steps 6`. Costs ~30 min/task,
known to produce ~25–30 unique paths on hard gmail tasks. **Caveat:** in
testing, one diversity run hung mid-rollout; if you go this route, consider
adding a per-rollout watchdog (see §7 "Open hardening tasks").

After MCTS finishes, run §3.2.

### 4.1 Resume rules

- `run_one_task` re-runs a task even if `summary.json` exists — move the
  old dir aside (e.g. `task_e1 → task_e1.bak/`) before re-running to
  preserve provenance.
- `run_all_gmail` is resume-aware and **skips** any task that already has
  `summary.json`.
- `build_synthetic` skips any `path_NNN` that already has
  `validation.json`. Pass `--force` to override.

---

## 5. How to Onboard a New Environment

The pipeline is largely environment-agnostic, but two things need tuning
when you go beyond gmail.

### 5.1 Required (no code changes)

- `apps/<env>/server.py` exposes the canonical HTTP API:
  `GET /api/state`, `PUT /api/state`, `POST /api/reset`, `GET /api/events`.
  All gmail/linear/gitlab apps in this repo already do this.
- `apps/<env>/real-tasks/<task_id>.py` exports a
  `verify(server_url) -> (bool, str)` function. Use existing tasks as a
  template.

### 5.2 Likely tuning per environment (`thesis/synthetic/diff.py`)

`diff.py` carries three configuration constants. **These were tuned for
gmail. New environments will need updates.**

```python
LIST_FIELDS = {        # top-level keys whose value is a list-of-objects
    "emails", "labels", "filters", "contacts", "blockedSenders",
}
SINGLETON_FIELDS = {   # top-level keys whose value is a dict to diff in place
    "settings", "currentUser",
}
TOP_DENY = {           # top-level keys excluded from diffs (server-internal)
    "_seedVersion", "_seedDataVersion",
    "_nextEmailId", "_nextLabelId", "_nextFilterId",
}
ENTITY_DENY = {        # per-entity-type field denylist
    "labels": {"messageCount", "unreadCount"},   # derived counters
}
```

**For each new environment, before running the pipeline:**

1. **Inventory the seed state.** Run the app, hit `GET /api/state`, and
   list every top-level key. Classify each as a list, a singleton, or
   internal/derived.
2. **Update `LIST_FIELDS` and `SINGLETON_FIELDS`** to match. Anything
   absent from both is treated as a free field at the top level.
3. **Find the denyable fields** by inspecting the app's server code for
   `_nextSomethingId` counters and any `set_*` mutators that recompute
   derived fields (e.g. `messageCount`). Add them to `TOP_DENY` or
   `ENTITY_DENY`.
4. **Confirm stable keys for created entities** in `_stable_key()`. The
   current logic uses `name` for labels, `criteria` for filters, etc. If
   your environment creates entities of new types, extend `_stable_key`.

> **Note:** `_stable_key` is what the verifier uses to *find* a created
> entity in the live state, since the seed counter (`_nextEmailId`) may
> have been bumped by a different number of actions. Get this wrong and
> "EntityCreated" assertions will fail intermittently.

### 5.3 Codegen tuning (`thesis/synthetic/codegen.py`)

Codegen does not depend on environment-specific knowledge — it walks the
diff and emits assertions/mutations field by field. You should **not**
need to edit it for a new env.

The only env-specific helper it emits is `_find_filter_by_criteria`,
which is gmail-specific. If a new environment also creates entities by
nested-criteria, generalise that helper; otherwise leave it.

### 5.4 Validator (no changes expected)

`thesis/synthetic/validator.py` is environment-agnostic. It uses
`evaluation/server.py`'s lifecycle helpers and the same Playwright
`BrowserSession` the MCTS runner uses.

### 5.5 Sanity checklist for a new environment

```bash
# 1. Pick a known-passing task and run MCTS
python -m thesis.scripts.run_one_task \
    --web-app apps/<env> --task-id <known_task> \
    --budget 30 --max-depth 4 --top-k 6

# 2. Generate synthetic tasks from its trajectories
python -m thesis.scripts.build_synthetic \
    --env <env> --task-id <known_task> --port 8780

# 3. Inspect output
cat thesis/synthetic-tasks/<env>/<known_task>/index.json

# 4. Manually read one task.py and confirm verify() looks right
cat thesis/synthetic-tasks/<env>/<known_task>/path_001/task.py
```

If any path's `validation.json` shows
`golden_pass=False` and the message blames a derived/internal field, that
field needs to be added to `TOP_DENY` or `ENTITY_DENY` in `diff.py`.

If `replay_pass=False`, the action selectors in `actions.json` are
unstable — fix the app's `id`/`data-action` attributes (see "Selector
Stability" in `CLAUDE.md`); do not patch the pipeline.

---

## 6. Roadmap (in priority order)

### Tier 1 — finish the gmail pilot

1. **Diversity rerun** of `task_e1, task_e6, task_m1, task_h1, task_h2`
   with `--budget 80 --max-depth 10 --top-k 6 --sim-steps 6`. Move
   existing trace dirs to `<task>.bak/` first to preserve provenance.
2. **Pipeline run** on each. Aggregate validation rate.
3. **Target:** ≥80% valid synthetic tasks across the pilot. (`task_e1.bak`
   alone already hits 100%.)

### Tier 2 — instruction polishing

The auto-generated instructions are mechanical action transcripts:
`"Perform: click(email-star-1: ☆) → click(email-star-5: ☆)"`. To make
synthetic tasks usable as eval prompts:

1. Add `thesis/synthetic/instruction_polish.py` that takes the diff +
   action labels and queries an open-source LLM (Llama / Qwen) for a
   natural-language paraphrase. Cache results in
   `task.py` as a separate `INSTRUCTION_POLISHED` constant.
2. Keep the mechanical version as `INSTRUCTION_RAW` for ablation.
3. Re-run the validator after polishing — only the `INSTRUCTION` field
   changes, the verify/solver are untouched, so all three arms must
   still pass.

### Tier 3 — scale across environments

For each environment in `apps/` (linear-account-settings, gitlab-plan-and-track, …):

1. Tune `LIST_FIELDS`, `SINGLETON_FIELDS`, `TOP_DENY`, `ENTITY_DENY`
   (see §5.2).
2. Sanity-check on one known-passing task (§5.5).
3. Run the full pipeline.
4. Track aggregate valid-rate per environment in a single
   `thesis/synthetic-tasks/_summary.json`.

### Tier 4 — productionising

1. **Per-rollout watchdog** in MCTS runner (one task hung last time —
   needs root-cause + retry/skip logic before mass runs).
2. **CI gate**: cron job re-validates every existing synthetic task
   weekly to catch regressions in apps or evaluation.
3. **Difficulty filter**: classify generated tasks by their action-count
   and entity-touch count, so downstream eval sets can balance difficulty.
4. **De-duplication across source tasks**: currently dedup is
   per-`task_id`. A path from `task_e1` and a path from `task_m1` may
   produce the same diff — collapse those.

---

## 7. Known Risks / Open Hardening Tasks

| Risk | Trigger | Mitigation |
|---|---|---|
| MCTS hang | Long diversity runs sat at 0% CPU mid-rollout | Add per-rollout watchdog (kill+continue) |
| Denylist gap | New env writes a derived counter we don't know about | Diff.json will show non-action mutations; add to `TOP_DENY` |
| Created-entity drift | Stable key for a new entity type not in `_stable_key` | Extend `_stable_key`; "EntityCreated" assertions will be flaky otherwise |
| Browser race after reset | Skipping `_wait_for_seed_repush` in `replay.py` | Already guarded; **do not remove** the two-consecutive-reads check |
| Selector instability | App uses generated CSS selectors | Fix the app's `id`/`data-action`; do not patch generic selectors |

---

## 8. Files You Should Know About

```
thesis/synthetic/
├── diff.py             # state diff — env-specific tuning lives here
├── codegen.py          # emits verify(server_url) + solver(state)
├── instruction.py      # mechanical instruction templater (Tier 2 will replace)
├── state_capture.py    # cached state read + live replay
├── validator.py        # 3-arm validator (golden / seed-fail / replay)
└── builder.py          # orchestrator

thesis/scripts/
├── build_synthetic.py  # CLI entry point
├── negative_control.py # validator drift-detection sanity check
└── rebuild_index.py    # rebuild a per-task index.json from disk

thesis/synthetic-tasks/ # output root (gitignored)
└── <env>/<task_id>/
    ├── index.json      # aggregate validation summary
    └── path_NNN/...    # one self-contained task per path
```

---

## 9. Quickref: typical workflow for a teammate

```bash
# A. Run MCTS on a new task
python -m thesis.scripts.run_one_task \
    --web-app apps/gmail --task-id task_m2 \
    --budget 40 --max-depth 8 --top-k 6

# B. Convert trajectories to synthetic tasks
python -m thesis.scripts.build_synthetic \
    --env gmail --task-id task_m2 --port 8765

# C. Look at results
cat thesis/synthetic-tasks/gmail/task_m2/index.json
ls  thesis/synthetic-tasks/gmail/task_m2/

# D. Sanity-check the validator behaves correctly
python -m thesis.scripts.negative_control     # must say "PASS"

# E. Rebuild aggregate index after edits / partial runs
python thesis/scripts/rebuild_index.py \
    --task-dir thesis/synthetic-tasks/gmail/task_m2
```
