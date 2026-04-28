# CLAUDE.md

<!-- ════════════════════════════════════════════════════════════════
     KARPATHY BEHAVIORAL GUIDELINES
     ⚠ DO NOT EDIT THIS SECTION IN ANY FUTURE ITERATION ⚠
     This section is frozen. Treat it as non-negotiable constraints.
     ════════════════════════════════════════════════════════════════ -->

## Behavioral Guidelines (Karpathy — Frozen)

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed.
For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria
("make it work") require constant clarification.

***

**These guidelines are working if:** fewer unnecessary changes in diffs,
fewer rewrites due to overcomplication, and clarifying questions come
before implementation rather than after mistakes.

<!-- ════════════════════════════════════════════════════════════════
     PROJECT SECTION — update freely as the thesis evolves
     ════════════════════════════════════════════════════════════════ -->

## What This Project Is

WebArena-Infinity is a scalable pipeline for auto-generating and evaluating
web-app testing environments against AI browser agents. **This CLAUDE.md is
scoped to the thesis extension only (`thesis/`).** Do not modify
`apps/`, `evaluation/`, `infra/`, or `bench/` unless explicitly told to.

The Python package name is `mirror-mirror` (`pyproject.toml`). Use `uv`
(not pip), Python >=3.12. Run `bash setup.sh` once to install deps,
Playwright Chromium, and OS browser deps.

***

## Thesis Module — MCTS Trajectory Generator

`thesis/` is self-contained. It runs **Monte-Carlo Tree Search** over
WebArena-Infinity environments to discover multiple distinct successful
execution paths per task. No LLM calls, no parallelism — read-only with
respect to all other repo modules.

### Layout

```
thesis/
├── mcts/
│   ├── action.py       # Action dataclass (kind, selector, value, label)
│   ├── browser.py      # Playwright session + exec_action + snapshot_candidates
│   ├── proposer.py     # BM25 + heuristic scoring; softmax_sample for simulation
│   ├── replay.py       # reset_to_seed, replay, state_hash
│   ├── reward.py       # load_verifier, safe_verify
│   ├── runner.py       # per-task orchestrator: server → browser → MCTS → persist
│   └── search.py       # core MCTS loop: Node, MCTSConfig, MCTSResult, run_mcts
├── scripts/
│   ├── run_one_task.py     # single-task debug entry point
│   └── run_all_gmail.py    # batch runner (resume-aware; skips tasks with summary.json)
├── analysis/
│   └── trace_stats.py      # aggregates traces → per-difficulty stats
└── traces/                 # output root (gitignored)
    └── <env>/<task_id>/
        ├── tree.json
        ├── summary.json        # see schema below
        ├── log.txt
        └── trajectories/
            └── path_NNN/
                ├── actions.json    # list of Action.to_dict()
                ├── verify.json     # passed, message, terminal_depth, state_hash, found_in_rollout
                ├── states/         # step_000.json … step_NNN.json  (if save_states=True)
                └── screenshots/    # step_000.png  … step_NNN.png   (if capture_screenshots=True)
```

### Running

```bash
# Single task (primary dev loop)
python -m thesis.scripts.run_one_task \
    --web-app apps/gmail --task-id task_e1 \
    --budget 40 --max-depth 4 --top-k 6

# Batch — overnight; auto-resumes (skips tasks that already have summary.json)
python -m thesis.scripts.run_all_gmail \
    --budget 30 --max-depth 8 --top-k 5

# Aggregate stats
python -m thesis.analysis.trace_stats --traces-dir thesis/traces/gmail
```

**Practical defaults:**
| Difficulty | `--budget` | `--max-depth` | `--top-k` |
|---|---|---|---|
| Easy | 30 | 4 | 6 |
| Medium/Hard | 40 | 8 | 5 |

### MCTSConfig Parameters

| Flag | Field | Role |
|---|---|---|
| `--budget` | `budget` | Number of rollouts — primary diversity knob |
| `--max-depth` | `max_depth` | Max action depth per path from seed |
| `--top-k` | `top_k` | Candidates kept at each node (expansion + simulation) |
| `--sim-steps` | `sim_steps` | Max additional random steps in simulate() past expanded leaf |
| *(code only)* | `ucb_c` | UCB1 exploration constant (default 1.4) |
| *(code only)* | `seed` | RNG seed for simulation reproducibility (default 0) |

***

## Key Data Structures (read before extending)

### `Action` (`mcts/action.py`)

```python
@dataclass(frozen=True)
class Action:
    kind: Literal["click", "fill", "press", "navigate"]
    selector: str       # stable selector string — see Selector Stability below
    value: str | None   # payload for fill/press/navigate; None for click
    label: str          # human-readable label (from text or aria-label)

    def key(self) -> str: ...        # canonical hash key for tree dedup
    def to_dict(self) -> dict: ...   # serialized to actions.json
```

### `Node` (`mcts/search.py`)

```python
@dataclass
class Node:
    parent: Node | None
    action_from_parent: Action | None
    depth: int
    visits: int          # MCTS N
    wins: float          # MCTS W (reward sum; currently binary 0/1)
    children: dict[str, Node]       # action.key() → Node
    untried: list[ScoredAction]     # candidates not yet expanded from this node
    expanded_keys: set[str]         # keys ever seen as children (avoids re-add)
    state_hash: str | None          # SHA1 of /api/state at this node
    passed: bool                    # verifier outcome at this node
    verifier_msg: str
    expanded: bool                  # True once untried has been populated
```

### `MCTSResult` (`mcts/search.py`)

```python
@dataclass
class MCTSResult:
    root: Node
    passing_paths: list[PassingPath]
    visited_state_hashes: set[str]
    n_rollouts: int
    elapsed: float
    early_stopped: bool   # always False in current code — extension point
```

### `summary.json` schema

```json
{
  "task_id": "task_e1",
  "difficulty": "easy",
  "instruction": "...",
  "config": { "budget": 30, "max_depth": 4, "top_k": 6, "sim_steps": 4, "ucb_c": 1.4, "seed": 0 },
  "n_passing_paths": 3,
  "min_steps": 2,
  "max_steps": 5,
  "median_steps": 3,
  "n_unique_states": 47,
  "n_unique_action_keys": 12,
  "elapsed_s": 142.3,
  "error": null
}
```

Use `n_unique_states` (not `n_passing_paths`) as the honest exploration coverage metric.

***

## Cross-Module Dependency Map

**Thesis code may ONLY cross module boundaries at these exact points:**

```
thesis/mcts/runner.py
  └── imports from evaluation/server.py:
        start_server(web_app_dir, port) → subprocess
        stop_server(proc)
        wait_for_server(port) → bool

thesis/mcts/reward.py
  └── mirrors evaluation/tasks.py::load_verifier()
        signature: load_verifier(web_app_dir, verify_path) → VerifierFn
        VerifierFn: (server_url: str) -> tuple[bool, str]
```

**Never import from `infra/`, `apps/`, or `bench/` in thesis code.**
If `evaluation/server.py` or `evaluation/tasks.py` changes their signatures,
`runner.py` and `reward.py` must be updated to match.

***

## App Server Interface (read-only contract for thesis code)

The MCTS module talks to apps exclusively through this HTTP API — never
by editing app files directly.

| Endpoint | Purpose |
|---|---|
| `GET /api/state` | Read current app state (returns 404 until browser has PUT once — by design) |
| `PUT /api/state` | Browser pushes full state on load and every mutation |
| `POST /api/reset` | Restore seed state; fires SSE reset event to browser |
| `GET /api/events` | SSE stream for reset notifications |

### Reset Contract in `replay.py` — Never Remove This Guard

After `POST /api/reset`:
1. Server deep-copies `_seed_state` back to `_app_state` immediately.
2. Browser's SSE handler fires, clears in-memory state, re-PUTs seed.
3. `_wait_for_seed_repush` polls `/api/state` every 150ms until **two
   consecutive reads match `expected_seed`**, with a 3s timeout.
4. Only then does replay of the action prefix begin.

Removing step 3 causes the browser's stale pre-reset state to overwrite
the server on the next action, silently corrupting the replay.

### Running an App Locally

```bash
cd apps/gmail && python3 server.py --port 8000
```

Reference / gold-standard apps: `apps/linear-account-settings/` and
`apps/gitlab-plan-and-track/`.

***

## Selector Stability (Replay Determinism)

`actions.json` selectors must use this preference order:

```
id=  >  data-action=  >  data-testid=  >  role=[name=]  >  text=  >  css=
```

A `css=` selector in `actions.json` signals a **broken app**, not a search
bug. Fix the app's `id`/`data-action` attributes, not the selector.

***

## Proposer Internals (read before modifying scoring)

`proposer.py` scores candidates using BM25-like token overlap + boosts:

| Signal | Boost |
|---|---|
| `data-action` tokens match instruction | +0.5 |
| `data-value` tokens match instruction | +0.7 |
| `input`/`textarea` + fill-verb in instruction | +0.4 |
| `button` + click-verb in instruction | +0.3 |
| `data-testid` present | +0.05 |
| Element text in `_NAV_TOKENS` (settings, menu, tab…) | +0.15 |
| `data-dropdown` present | +0.10 |
| Fill action with value payload | +0.6 base + 0.4×value_overlap |
| Action key already used on current path | −0.3 penalty |

`_value_candidates(instruction)` extracts fill payloads from: quoted strings,
"called/named/titled X" patterns, and capitalized multi-word phrases.

`softmax_sample` is used in simulation (random rollouts); `propose` returns
a sorted deterministic list used in expansion (tree phase).

***

## Verifier Pattern

Each `apps/<app>/real-tasks/task_*.py` exports:
```python
def verify(server_url: str) -> tuple[bool, str]: ...
```
Verifiers only read `/api/state` — they never touch the UI.
`reward.py::safe_verify` wraps every call so verifier exceptions never
crash the MCTS loop; they return `(False, "Verifier exception: ...")`.

***

## Trace Integrity Rules

- Passing paths are deduped by **terminal state hash**, not action sequence.
  `seen_passing_hashes` in `search.py` enforces this. Two paths that reach
  the same terminal state are collapsed to the first one found.
- `thesis/traces/` is gitignored. To preserve a specific run:
  `git add -f thesis/traces/<env>/<task_id>/`
- `trajectories/path_NNN/` are sorted by `(depth, found_in_rollout)` — so
  `path_001` is always the shortest passing path found earliest.
- `MCTSResult.early_stopped` exists but is always `False` in current code.
  It is an intentional extension point for budget-cutoff strategies.

***

## Reference Docs (consult before changing interfaces)

| File | Covers |
|---|---|
| `thesis/README.md` | Full schema docs, troubleshooting table, extension guide |
| `docs/environment-protocol.md` | Full HTTP API contract |
| `docs/real-task-design-guide.md` | Task authoring conventions |
| `docs/verifier-sanity-check.md` | Sanity-check authoring |
| `evaluation/server.py` | `start_server` / `stop_server` / `wait_for_server` — imported by runner.py |