# MCTS Trajectory Generator

## Start here (minimal concepts)

If you only need the essentials for running experiments, use this section.

### Core terms

- **Task**: one verifier-backed objective (for example `task_e1`).
- **Trajectory**: one action sequence that ends in verifier pass.
- **Rollout**: one MCTS iteration (select -> reset+replay -> expand -> simulate -> backprop).
- **State**: JSON returned by `GET /api/state` on the app server.

### Why there is both reset and PUT

- `POST /api/reset` (server endpoint): restores app to seed state.
- `PUT /api/state` (browser -> server sync): browser sends full current state after mutations.
- Verifier reads state through `GET /api/state`.

So practically: each rollout resets server state, browser re-syncs, then MCTS tries actions and checks verifier.

### How to think about `states/step_XXX.json`

- `step_000.json`: seed state before actions in that saved path.
- `step_001.json`: state after first action.
- `step_002.json`: state after second action.
- ...and so on until the trajectory terminal step.

These are not "tree-internal hidden states"; they are concrete server states captured during replay of each saved path.

### Why some paths include useless actions

MCTS stops a **rollout** when it hits pass, but the whole run continues across many rollouts until budget is used.
Also, verifier may not penalize side actions, so a path can include irrelevant clicks and still pass.
That is expected and useful for measuring efficiency gaps between agents.

### Most important parameters

- `--budget`: number of rollouts (primary diversity knob).
- `--max-depth`: max action length of a path.
- `--top-k`: branching width per expansion.
- `--sim-steps`: extra simulation depth after expansion.

### Practical defaults (single MacBook)

- Easy tasks: `--budget 30 --max-depth 4 --top-k 6`
- Medium/hard tasks: `--budget 40 --max-depth 8 --top-k 5`

### Small run (single task)

```bash
python -m thesis.scripts.run_one_task \
  --web-app apps/gmail \
  --task-id task_e1 \
  --budget 40 --max-depth 4 --top-k 6
```

### Long run (all gmail tasks)

```bash
nohup python -m thesis.scripts.run_all_gmail \
  --budget 30 --max-depth 8 --top-k 5 \
  > /tmp/mcts-gmail.log 2>&1 &
```

---

A self-contained pipeline that uses **Monte-Carlo Tree Search** to discover
multiple distinct successful execution paths for every task in a
WebArena-Infinity environment, **without modifying the original repo**.
All new code lives under this `thesis/` directory; outputs land under
`thesis/traces/` (which is git-ignored).

The first target environment is `apps/gmail` (60 verifiable tasks across
easy / medium / hard). The runner is fully parameterised — pointing it at
`apps/linear-account-settings`, `apps/handshake-career-exploration`, etc.
is a one-line change.

---

## 1. What this generates and why

For each task you get a directory like:

```
thesis/traces/gmail/task_e6/
├── tree.json                 # full MCTS tree (visits, wins, action edges)
├── summary.json              # n_paths, min/median/max steps, ...
├── log.txt                   # human-readable run log
└── trajectories/
    ├── path_001/             # shortest passing path
    │   ├── actions.json      # ordered Action dicts
    │   ├── verify.json       # {passed, message, terminal_depth, state_hash}
    │   ├── states/
    │   │   ├── step_000.json # /api/state at each step
    │   │   └── step_001.json
    │   └── screenshots/
    │       ├── step_000.png
    │       └── step_001.png
    ├── path_002/             # next-shortest distinct path (different state hash)
    └── ...
```

The intended use for your thesis benchmark:

- For each task you have **multiple different successful paths**, sorted
  by length. Use `min_steps` / `median_steps` / `max_steps` plus
  `pairwise_edit_distance_mean` (from `thesis/analysis/trace_stats.py`)
  as your "long-horizon efficiency" labels.
- The MCTS process also records the **set of unique server-state hashes
  visited** — this gives you a denominator for "exploration coverage".
- Failed paths are not separately persisted, but the full tree
  (`tree.json`) records every action edge that was tried and never paid
  off. That's enough to reconstruct unsuccessful explorations later.

---

## 2. How it works

```
                         POST /api/reset
                       (server restores seed,
                        SSE tells browser to
                        re-PUT seed state)
                         │
            ┌────────────┴────────────┐
            │                         │
runner.py spawns apps/gmail/server.py │  Playwright Chromium tab
            │                         │
            └─── PUT /api/state ──────┘
            ▲     (browser → server,
            │      after every mutation)
            │
            │
        run_mcts() in search.py:
          select  → reset+replay → expand → execute → simulate → backprop
          (verifier at every step → shortest passing path)
          (proposer.py scores DOM candidates with BM25 + heuristics)
```

**The key trick**: the only state-restoration primitive provided by the
existing servers is `POST /api/reset` (seed only). MCTS therefore
treats every node by its action-prefix from seed, restores via
reset+replay, and relies on **stable selectors** (id / data-action /
data-testid / role+name) so replays are deterministic.

---

## 3. Running

### 3.1 Prerequisites

This module reuses only `requests` and `playwright` — both come in via
the existing `browser-use` dependency. If `browser-use` isn't installed
in the active Python env yet, run the repo's setup script first:

```bash
cd webarena-infinity
bash setup.sh
```

(That installs `browser-use`, which pulls Playwright, which downloads
the headless Chromium binary.)

### 3.2 One task (debug)

```bash
cd webarena-infinity
python -m thesis.scripts.run_one_task \
    --web-app apps/gmail \
    --task-id task_e6 \
    --budget 30 --max-depth 8 --top-k 5
```

Output → `thesis/traces/gmail/task_e6/`.

### 3.3 All gmail tasks (overnight run)

```bash
cd webarena-infinity
python -m thesis.scripts.run_all_gmail
```

Resume-aware: tasks with an existing `summary.json` are skipped. To
re-run them anyway, pass `--no-resume`. Restrict scope with
`--difficulty easy` or `--limit 5`.

### 3.4 Aggregate stats

```bash
python -m thesis.analysis.trace_stats --traces-dir thesis/traces/gmail
# → writes thesis/traces/gmail/_stats.json + prints a per-difficulty rollup
```

### 3.5 Different environment

Every script accepts `--web-app apps/<other>`. The runner does not assume
gmail anywhere — it loads tasks from `<web-app>/<suite>.json` and the
verifiers from `<web-app>/<verify_path>` exactly the way the existing
harness in `evaluation/tasks.py` does.

---

## 4. Output schema (field-by-field)

### `summary.json`

| field                  | type            | meaning                                                |
|------------------------|-----------------|--------------------------------------------------------|
| `task_id`              | str             | e.g. `task_e6`                                         |
| `difficulty`           | str             | `easy` / `medium` / `hard`                             |
| `instruction`          | str             | natural-language task                                  |
| `config.budget`        | int             | rollouts requested                                     |
| `config.max_depth`     | int             | max actions from seed                                  |
| `config.top_k`         | int             | candidates per expansion                               |
| `config.sim_steps`     | int             | sim depth beyond expanded leaf                         |
| `config.ucb_c`         | float           | UCB1 exploration constant                              |
| `n_passing_paths`      | int             | distinct successful paths (deduped by terminal state)  |
| `min_steps`            | int \| null     | shortest path length                                   |
| `max_steps`, `median_steps` | int \| null| —                                                      |
| `n_unique_states`      | int             | distinct `/api/state` hashes visited during search     |
| `n_unique_action_keys` | int             | distinct `(kind, selector, value)` triples used        |
| `elapsed_s`            | float           | wall-clock for `run_mcts`                              |
| `error`                | str \| null     | populated if the runner aborted                        |

### `actions.json` (per trajectory)

Ordered list of `Action` dicts:

```json
[
  {"kind": "click", "selector": "id=settingsBtn", "value": null,
   "label": "Settings"},
  {"kind": "click", "selector": "data-action=switch-theme",
   "value": null, "label": "Dark"}
]
```

`kind` is one of `click`, `fill`, `press`, `navigate`. `selector` uses
the small mini-syntax described in `thesis/mcts/action.py` —
`id=…`, `data-action=…`, `data-testid=…`, `role=…[name=…]`, `text=…`,
or `css=…` as a last resort.

### `tree.json`

Recursive JSON tree:

```json
{
  "depth": 0,
  "visits": 30, "wins": 4.0, "passed": false, "verifier_msg": "",
  "state_hash": "abc...",
  "action_from_parent": null,
  "children": [
    {"depth": 1, "visits": 12, "wins": 4.0, "passed": false,
     "action_from_parent": {"kind": "click", "selector": "id=settingsBtn",
                            "value": null, "label": "Settings"},
     "children": [...]}
  ]
}
```

### `states/step_NNN.json`

The server's full `/api/state` JSON (sorted keys) right after step
`NNN`. `step_000.json` is the seed snapshot.

### `verify.json`

```json
{
  "passed": true,
  "message": "Theme is set to dark mode (theme='dark').",
  "terminal_depth": 2,
  "state_hash": "...",
  "found_in_rollout": 4
}
```

---

## 5. Important tips & considerations

These are the things that matter most when running, extending, or
analysing this pipeline. Read them once before you start an overnight
run.

### 5.1 Determinism is selector hygiene

The whole approach relies on **action replay being deterministic**. We
guarantee this by ranking selectors as:

1. `id=…` — only used when the id matches `[A-Za-z][A-Za-z0-9_-]*` so we
   never escape weird characters.
2. `data-action=…` — gmail and the other apps use this attribute as the
   primary "user intent" hook (see `CLAUDE.md` "Lessons Learned").
3. `data-testid=…` — fallback hook authored by the apps' generators.
4. `role=…[name=…]` — accessibility-tree based, robust to refactors.
5. `text=…` — last-resort visible-text match.
6. `css=…` — explicit nth-of-type path (avoid; only used for nameless
   non-semantic divs).

If you find replay flaking on a task, look at `actions.json` for that
path: if you see `css=…` selectors, the upstream app is missing
semantic anchors — file it as an environment bug, don't fight the
search.

### 5.2 Reset latency: don't trust an immediate GET

After `POST /api/reset` the server's `_app_state` already equals seed
(deep-copy on reset). But the **browser** still holds its post-action
in-memory state. If you don't wait for the browser's SSE handler to
re-PUT seed, the next user action you take will overwrite the server
with stale state.

`thesis/mcts/replay.py` waits for two consecutive `/api/state` reads to
match the seed snapshot before proceeding. If you swap in a different
app whose JS doesn't follow the canonical SSE pattern, you may need to
extend `_wait_for_seed_repush` — never just remove it.

### 5.3 Verifier-at-every-step is essentially free

Every verifier in `apps/<app>/real-tasks/task_*.py` is a single GET +
Python check. Calling it after each step means MCTS automatically
discovers the **shortest** passing path per branch — and lets us bail
out of useless deeper explorations.

### 5.4 State equivalence: hash JSON to dedupe

Two action sequences that produce the same `/api/state` aren't really
two different paths. We hash JSON state at every node and dedupe
passing paths by terminal state hash. When you analyse trajectories,
prefer `len(set(state_hashes))` over `len(actions)` — that's the
honest "exploration breadth" metric.

### 5.5 Replay cost dominates — keep depth shallow

Each rollout pays:

```
~1 reset (≈ 0.4 s)  +  N×replay actions (≈ 0.3 s each)  +  K×expand+verify
```

So total time per task ≈ `budget × max_depth × 0.3-0.5s`.

Defaults that work on a 16 GB MacBook (Air or Pro):

| param        | default | sane range  | notes                                |
|--------------|--------:|-------------|--------------------------------------|
| `budget`     |     30  | 20 – 60     | more rollouts → higher diversity, linear cost |
| `max_depth`  |      8  |  5 – 12     | most real-tasks need ≤6 ground-truth steps    |
| `top_k`      |      5  |  3 – 8      | lower = greedier, higher = wider but slower   |
| `sim_steps`  |      4  |  2 – 6      | stochastic random rollout depth past the leaf |

### 5.6 Verifier semantics: only `/api/state` matters

Every verifier reads `GET /api/state` and ignores UI focus, scroll
position, modal state, etc. So a "passing" trajectory may stop
mid-modal — that's fine for benchmark purposes. But it means you
shouldn't take screenshots as ground-truth task completion; trust
`verify.json` only.

### 5.7 Native UI elements are absent

Per `CLAUDE.md`, generated apps avoid `<select>`, `alert()`, `confirm()`,
file pickers. So Playwright's normal click/fill works without
`page.expect_dialog()` shims. Don't add them — you'll just slow things
down.

### 5.8 Cross-module bugs in apps

Also from `CLAUDE.md`: handler-key drift between an app's `views.js`
and `app.js` means a clicked element may silently no-op. If MCTS gets
stuck on a specific task with no progress despite many rollouts, the
bug is probably in the upstream app, not your search. Inspect
`tree.json` — if visits are spread evenly with zero wins anywhere, log
the failing actions and move on.

### 5.9 Don't run two tasks in parallel on a laptop

Each Chromium tab eats ~1.5 GB of resident memory plus several CPU
cores when JS is busy. The runner is sequential by design. To shard
across machines later, parameterise `--port` and run different `--web-app`
subsets per machine.

### 5.10 Failures are signal — keep them

For your thesis benchmark, the contrast between an agent that solves a
task in 3 steps and one that takes 12 is the whole point. Don't filter
out paths above some length cutoff during generation; do that downstream.

### 5.11 How to extend later

| change                              | where                                 |
|-------------------------------------|---------------------------------------|
| LLM-guided action proposal          | replace `thesis/mcts/proposer.py` only|
| Different exploration strategy      | `_select` in `thesis/mcts/search.py`  |
| Persist failed paths separately     | augment `_persist_trajectories` in `runner.py` |
| Try a different app                 | pass `--web-app apps/<x>`             |
| Use function-tasks instead of real  | pass `--task-suite function-tasks`    |
| Compare two seeds                   | pass `--seed N`; outputs to same dir, idempotent if you change `--output-dir` |

---

## 6. What this pipeline deliberately does NOT do

- **Train or tune any model.** Trajectories are output artifacts, full stop.
- **Use any LLM.** The proposer is pure heuristic; no API keys needed.
- **Run in parallel.** One task at a time, sequentially.
- **Modify the upstream codebase.** Every existing file under `apps/`,
  `evaluation/`, `infra/`, `bench/` is read-only for us.
- **Score "agent efficiency" itself.** That's your downstream benchmark
  step; this just emits the ground-truth ladder of paths.

---

## 7. Troubleshooting

| symptom                                         | cause / fix                                 |
|-------------------------------------------------|---------------------------------------------|
| `Seed state was never pushed by the browser…`   | App's JS failed to init (check `apps/<app>/js/app.js`). Try opening the app manually first. |
| Many `replay failed (…) — stopping replay` lines| App rendered different DOM than expected. Either the proposer is producing fragile selectors (rare) or the app has a non-deterministic render. Inspect `actions.json` for `css=…` entries. |
| `0 paths within budget` for an obviously easy task | `top_k` is filtering out the right element. Lower the BM25 stop-word list in `thesis/mcts/proposer.py` or raise `--top-k`. |
| Chromium hang / OOM                             | Reduce `--budget` and `--max-depth`; restart machine, re-run with `--no-resume` only on failed tasks. |
| Port 8765 in use                                | Pass `--port 8766` (or kill the zombie: `lsof -ti :8765 | xargs kill -9`). The runner does NOT auto-kill — that's deliberate to avoid clobbering manual debugging sessions. |
