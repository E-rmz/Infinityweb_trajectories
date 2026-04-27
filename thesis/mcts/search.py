"""Monte-Carlo Tree Search over browser actions.

Each tree node represents a sequence of actions from the seed state.
The reward for a node is 1 if the standalone verifier passes at any
point along that path (we check after every action — see runner) and
0 otherwise.

Algorithm (one rollout):

  1. **Selection.** Walk from the root using UCB1 over the children
     already in the tree, until we hit a node that has at least one
     untried candidate (or is terminal/passing or at max depth).
  2. **Reset + replay.** Issue POST /api/reset and replay the path's
     action prefix (deterministic given stable selectors).
  3. **Expansion.** Snapshot the current DOM, score candidates with the
     proposer, pick the highest-scoring untried one, execute it, check
     verifier — this becomes a new child node.
  4. **Simulation.** From that new node, take heuristic-weighted random
     steps until we either hit max_depth or the verifier passes.
  5. **Backprop.** Add reward (1 / 0) to N, W of every node on the path.

The tree, every passing path, and every visited state hash are
persisted by the caller (see ``runner.py``).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable

from playwright.async_api import Page

from .action import Action
from .browser import exec_action, snapshot_candidates
from .proposer import ScoredAction, propose, softmax_sample
from .replay import reset_to_seed, state_hash


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass
class Node:
    parent: "Node | None"
    action_from_parent: Action | None
    depth: int = 0

    # MCTS bookkeeping
    visits: int = 0
    wins: float = 0.0
    children: dict[str, "Node"] = field(default_factory=dict)  # action.key() → Node

    # Proposer state for this node — populated on first visit.
    untried: list[ScoredAction] = field(default_factory=list)
    # The keys we've ever seen as children (so re-expansion doesn't re-add).
    expanded_keys: set[str] = field(default_factory=set)

    # Recorded server state hash + verifier outcome at this exact node.
    state_hash: str | None = None
    passed: bool = False
    verifier_msg: str = ""

    # Was this node ever expanded? Lets us tell un-visited from visited-but-expanded.
    expanded: bool = False

    # ----- helpers -----

    def path(self) -> list["Node"]:
        out: list[Node] = []
        n: Node | None = self
        while n is not None:
            out.append(n)
            n = n.parent
        return list(reversed(out))

    def actions(self) -> list[Action]:
        return [n.action_from_parent for n in self.path() if n.action_from_parent is not None]

    def used_keys(self) -> frozenset[str]:
        return frozenset(a.key() for a in self.actions())

    def is_root(self) -> bool:
        return self.parent is None

    def ucb1(self, c: float = 1.4) -> float:
        if self.parent is None or self.visits == 0:
            return float("inf")
        exploit = self.wins / max(self.visits, 1)
        explore = c * math.sqrt(math.log(max(self.parent.visits, 1)) / self.visits)
        return exploit + explore


# ---------------------------------------------------------------------------
# MCTS state container
# ---------------------------------------------------------------------------


@dataclass
class PassingPath:
    actions: list[Action]
    depth: int
    verifier_msg: str
    state_hash: str
    found_in_rollout: int


@dataclass
class MCTSConfig:
    budget: int = 30          # number of rollouts
    max_depth: int = 8        # max tree+sim depth from seed
    top_k: int = 5            # candidates kept at each node
    sim_steps: int = 4        # max additional random steps in simulate()
    ucb_c: float = 1.4
    seed: int = 0


@dataclass
class MCTSResult:
    root: Node
    passing_paths: list[PassingPath]
    visited_state_hashes: set[str]
    n_rollouts: int
    elapsed: float
    early_stopped: bool


# ---------------------------------------------------------------------------
# Side effects passed in by the runner so this file stays loosely coupled.
# ---------------------------------------------------------------------------


# Type aliases
ApplyAction = Callable[[Action], "object"]            # async — executes action on the page
GetState = Callable[[], dict]                         # sync — reads /api/state
RunVerifier = Callable[[], tuple[bool, str]]          # sync — calls the verifier


# ---------------------------------------------------------------------------
# Core MCTS loop
# ---------------------------------------------------------------------------


async def run_mcts(
    *,
    instruction: str,
    page: Page,
    server_url: str,
    expected_seed: dict,
    get_state: GetState,
    run_verifier: RunVerifier,
    config: MCTSConfig,
    log: Callable[[str], None] = print,
) -> MCTSResult:
    """Run MCTS for a single task.

    The function drives the browser via ``page`` (Playwright). It does
    NOT spawn or stop a server — the caller does that.
    """
    rng = random.Random(config.seed)
    root = Node(parent=None, action_from_parent=None, depth=0)
    root.state_hash = state_hash(expected_seed)

    passing_paths: list[PassingPath] = []
    seen_passing_hashes: set[str] = set()
    visited_state_hashes: set[str] = {root.state_hash}
    t0 = time.time()
    early_stopped = False

    # Cheap guard: check verifier once on the seed itself. Sometimes a
    # task is already trivially satisfied (rare for real-tasks but free).
    try:
        passed, msg = run_verifier()
        if passed:
            root.passed = True
            root.verifier_msg = msg
            passing_paths.append(
                PassingPath(actions=[], depth=0, verifier_msg=msg, state_hash=root.state_hash,
                            found_in_rollout=0)
            )
    except Exception:
        pass

    for rollout in range(1, config.budget + 1):
        rollout_t0 = time.time()
        try:
            await _one_rollout(
                root=root,
                instruction=instruction,
                page=page,
                server_url=server_url,
                expected_seed=expected_seed,
                get_state=get_state,
                run_verifier=run_verifier,
                config=config,
                rng=rng,
                rollout_idx=rollout,
                passing_paths=passing_paths,
                seen_passing_hashes=seen_passing_hashes,
                visited_state_hashes=visited_state_hashes,
            )
        except Exception as e:
            log(f"  [rollout {rollout}] aborted: {type(e).__name__}: {e}")
            continue
        log(
            f"  [rollout {rollout}/{config.budget}] {time.time() - rollout_t0:.1f}s  "
            f"paths={len(passing_paths)}  states={len(visited_state_hashes)}"
        )

    return MCTSResult(
        root=root,
        passing_paths=passing_paths,
        visited_state_hashes=visited_state_hashes,
        n_rollouts=config.budget,
        elapsed=time.time() - t0,
        early_stopped=early_stopped,
    )


# ---------------------------------------------------------------------------
# Rollout phases
# ---------------------------------------------------------------------------


async def _one_rollout(
    *,
    root: Node,
    instruction: str,
    page: Page,
    server_url: str,
    expected_seed: dict,
    get_state: GetState,
    run_verifier: RunVerifier,
    config: MCTSConfig,
    rng: random.Random,
    rollout_idx: int,
    passing_paths: list[PassingPath],
    seen_passing_hashes: set[str],
    visited_state_hashes: set[str],
) -> None:
    # 1. Selection (entirely in-memory).
    leaf = _select(root, config)

    # 2. Reset + replay the path to ``leaf``.
    await reset_to_seed(page, server_url, expected_seed)
    actions_so_far = leaf.actions()
    for a in actions_so_far:
        try:
            await exec_action(page, a)
        except Exception:
            # Replay failure: blame this branch by penalising the leaf, abort.
            _backprop(leaf, 0.0)
            return

    # 3. Expansion. If leaf hasn't been expanded, populate untried.
    if not leaf.expanded:
        await _expand_node(leaf, instruction, page, config, get_state)

    # If leaf is at max_depth or has no candidates, bail and back-propagate 0.
    if leaf.depth >= config.max_depth or not leaf.untried:
        _backprop(leaf, 1.0 if leaf.passed else 0.0)
        return

    # Pop the highest-scoring untried action and execute.
    chosen = leaf.untried.pop(0)
    leaf.expanded_keys.add(chosen.action.key())
    try:
        await exec_action(page, chosen.action)
    except Exception:
        _backprop(leaf, 0.0)
        return

    child = Node(parent=leaf, action_from_parent=chosen.action, depth=leaf.depth + 1)
    leaf.children[chosen.action.key()] = child

    # Verify + state hash at the new child.
    try:
        st = get_state()
        child.state_hash = state_hash(st)
        visited_state_hashes.add(child.state_hash)
    except Exception:
        child.state_hash = None

    try:
        passed, msg = run_verifier()
    except Exception as e:
        passed, msg = False, f"verifier exc: {e}"
    child.passed = passed
    child.verifier_msg = msg

    if passed:
        _record_passing(child, passing_paths, seen_passing_hashes, rollout_idx)
        _backprop(child, 1.0)
        return

    # 4. Simulation from the new child.
    sim_passed, sim_passing_node = await _simulate(
        start=child,
        instruction=instruction,
        page=page,
        config=config,
        rng=rng,
        get_state=get_state,
        run_verifier=run_verifier,
        visited_state_hashes=visited_state_hashes,
        rollout_idx=rollout_idx,
        passing_paths=passing_paths,
        seen_passing_hashes=seen_passing_hashes,
    )
    _backprop(child, 1.0 if sim_passed else 0.0)


# ----- selection -------------------------------------------------------------


def _select(root: Node, config: MCTSConfig) -> Node:
    node = root
    while True:
        if node.depth >= config.max_depth:
            return node
        # If the node is unexpanded OR has untried candidates, stop here so
        # the rollout can expand it.
        if (not node.expanded) or node.untried:
            return node
        if not node.children:
            return node
        # Pick the child with highest UCB1 (treat unvisited as +inf).
        best_key, best_node, best_score = None, None, -float("inf")
        for k, ch in node.children.items():
            sc = ch.ucb1(config.ucb_c)
            if sc > best_score:
                best_key, best_node, best_score = k, ch, sc
        if best_node is None:
            return node
        node = best_node


# ----- expansion -------------------------------------------------------------


async def _expand_node(
    node: Node,
    instruction: str,
    page: Page,
    config: MCTSConfig,
    get_state: GetState,
) -> None:
    """Snapshot the current page, score candidates, save top-K as untried."""
    cands = await snapshot_candidates(page)
    used = node.used_keys()
    scored = propose(instruction, cands, top_k=config.top_k, used_keys=used)
    # Drop duplicates already expanded from this node.
    node.untried = [s for s in scored if s.action.key() not in node.expanded_keys]
    node.expanded = True


# ----- simulation ------------------------------------------------------------


async def _simulate(
    *,
    start: Node,
    instruction: str,
    page: Page,
    config: MCTSConfig,
    rng: random.Random,
    get_state: GetState,
    run_verifier: RunVerifier,
    visited_state_hashes: set[str],
    rollout_idx: int,
    passing_paths: list[PassingPath],
    seen_passing_hashes: set[str],
) -> tuple[bool, Node | None]:
    """Heuristic-weighted random rollout. Verifier is checked after every step."""
    cur_actions = list(start.actions())
    cur_depth = start.depth
    used = {a.key() for a in cur_actions}
    while cur_depth < config.max_depth and (cur_depth - start.depth) < config.sim_steps:
        try:
            cands = await snapshot_candidates(page)
        except Exception:
            return False, None
        scored = propose(instruction, cands, top_k=config.top_k, used_keys=frozenset(used))
        if not scored:
            return False, None
        chosen = softmax_sample(scored, rng)
        if chosen is None:
            return False, None
        try:
            await exec_action(page, chosen.action)
        except Exception:
            return False, None
        cur_actions.append(chosen.action)
        used.add(chosen.action.key())
        cur_depth += 1

        try:
            st = get_state()
            h = state_hash(st)
            visited_state_hashes.add(h)
        except Exception:
            h = None
        try:
            passed, msg = run_verifier()
        except Exception as e:
            passed, msg = False, f"verifier exc: {e}"
        if passed:
            # Synthesize a transient "node" record for the passing path.
            sim_node = Node(parent=None, action_from_parent=None, depth=cur_depth)
            sim_node.passed = True
            sim_node.verifier_msg = msg
            sim_node.state_hash = h
            # Build a PassingPath directly from cur_actions (no tree expansion).
            if h not in seen_passing_hashes:
                passing_paths.append(
                    PassingPath(
                        actions=list(cur_actions),
                        depth=cur_depth,
                        verifier_msg=msg,
                        state_hash=h or "",
                        found_in_rollout=rollout_idx,
                    )
                )
                if h:
                    seen_passing_hashes.add(h)
            return True, sim_node
    return False, None


# ----- backprop --------------------------------------------------------------


def _backprop(node: Node, reward: float) -> None:
    n: Node | None = node
    while n is not None:
        n.visits += 1
        n.wins += reward
        n = n.parent


# ----- passing path recording ------------------------------------------------


def _record_passing(
    node: Node,
    passing_paths: list[PassingPath],
    seen_passing_hashes: set[str],
    rollout_idx: int,
) -> None:
    h = node.state_hash or ""
    if h and h in seen_passing_hashes:
        return  # already saw an equivalent terminal state
    actions = node.actions()
    passing_paths.append(
        PassingPath(
            actions=actions,
            depth=node.depth,
            verifier_msg=node.verifier_msg,
            state_hash=h,
            found_in_rollout=rollout_idx,
        )
    )
    if h:
        seen_passing_hashes.add(h)


# ---------------------------------------------------------------------------
# Tree serialization (used by the runner to persist tree.json)
# ---------------------------------------------------------------------------


def serialize_tree(root: Node) -> dict:
    """Convert the in-memory tree to a JSON-friendly dict."""
    def encode(n: Node) -> dict:
        return {
            "depth": n.depth,
            "visits": n.visits,
            "wins": round(n.wins, 4),
            "passed": n.passed,
            "verifier_msg": n.verifier_msg if n.passed else "",
            "state_hash": n.state_hash,
            "action_from_parent": (
                n.action_from_parent.to_dict() if n.action_from_parent else None
            ),
            "children": [encode(c) for c in n.children.values()],
        }

    return encode(root)
