"""Heuristic candidate-action scorer (no LLM, no API calls).

Given the task instruction and a list of ``Candidate`` elements harvested
from the live page, this module ranks them by token overlap and returns
the top-K. The scoring is intentionally simple and dependency-free:

  score = bm25_like(tokens(instruction), tokens(candidate)) + boosts

Boosts encode small priors:
  * +0.5 if the candidate's ``data-action`` attribute matches an
    instruction token (e.g. ``compose`` → button with data-action=compose)
  * +0.3 if a tag is highly task-relevant (``input`` for "type X",
    ``button`` for "click/star/mark/...")
  * +0.2 for candidates exposing ``data-testid`` (more stable selectors)
  * −0.3 recency penalty per repeat of an action already on the path

The proposer also generates a small library of synthesized "fill" actions
when the instruction contains values that look like text payloads (e.g.
quoted strings or words after "type"/"enter"/"named"/"called"). These
synthesized actions reuse the candidate's selector but supply a value
drawn from the instruction.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .action import Action
from .browser import Candidate


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


_STOPWORDS = frozenset(
    """
    a an and as at be but by do does for from has have he her his i if in into is
    it its me my of on or our she so that the their them then there they this to
    too us we will with you your yours hers him himself herself its theirs ours
    can could should would shall may might must one ones any all just into onto
    """.split()
)

# Words we deliberately keep even though they look generic — they map to UI verbs.
_VERB_KEEP = frozenset(
    "star unstar mark unmark archive trash delete restore snooze unsnooze move "
    "label create new compose send save discard reply forward block unblock "
    "switch toggle change set turn enable disable remove add open close show "
    "hide pin unpin import export read unread important spam".split()
)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    # Replace punctuation with spaces, preserve digits.
    text = re.sub(r"[^a-z0-9]+", " ", text)
    toks = []
    for t in text.split():
        if not t:
            continue
        if len(t) <= 1:
            continue
        if t in _STOPWORDS and t not in _VERB_KEEP:
            continue
        toks.append(t)
    return toks


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _candidate_text(c: Candidate) -> str:
    """Concatenate every textual signal we have about a candidate."""
    parts = [
        c.text,
        c.attrs.get("aria-label", ""),
        c.attrs.get("placeholder", ""),
        c.attrs.get("data-action", "").replace("-", " "),
        c.attrs.get("data-testid", "").replace("-", " "),
        c.attrs.get("data-dropdown", "").replace("-", " "),
        c.attrs.get("data-dropdown-id", "").replace("-", " "),
        c.attrs.get("data-value", "").replace("-", " "),
        c.attrs.get("data-snooze", "").replace("_", " "),
        c.attrs.get("data-category", ""),
        c.attrs.get("name", ""),
        c.attrs.get("href", ""),
    ]
    return " ".join(p for p in parts if p)


# Tokens that suggest "navigation / mode-switch" intent. Elements whose
# attribute text contains any of these get a small baseline prior so MCTS
# is willing to explore them even when the instruction shares no tokens
# with their visible label (e.g. clicking "Settings" before toggling the
# theme).
_NAV_TOKENS = frozenset(
    "settings menu dropdown tab open compose more options preferences "
    "appearance display theme account profile general advanced filters "
    "labels picker view picker filter sort".split()
)


def _bm25_overlap(query: list[str], doc: list[str]) -> float:
    """Tiny BM25-ish: length-normalized weighted overlap.

    Implemented locally so we don't pull in a search-engine dep.
    """
    if not query or not doc:
        return 0.0
    qset = set(query)
    dlen = len(doc)
    avgdl = 8.0  # rough average per element
    k1 = 1.4
    b = 0.75
    score = 0.0
    seen_in_doc: dict[str, int] = {}
    for t in doc:
        seen_in_doc[t] = seen_in_doc.get(t, 0) + 1
    for q in qset:
        if q not in seen_in_doc:
            continue
        f = seen_in_doc[q]
        # IDF-free — every query term has equal weight.
        norm = (k1 + 1) * f / (f + k1 * (1 - b + b * dlen / avgdl))
        score += norm
    return score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredAction:
    action: Action
    score: float


def _value_candidates(instruction: str) -> list[str]:
    """Pull plausible string payloads to type into <input>/<textarea>.

    Heuristics:
      * Anything in single/double quotes is a payload.
      * Capitalized multi-word phrases ("Q1 Priority", "Engineering Team").
      * Phrases right after verbs like name/title/called/labeled/labelled/named.
    """
    out: list[str] = []
    # Quoted strings
    for m in re.finditer(r"['\"]([^'\"]+)['\"]", instruction):
        v = m.group(1).strip()
        if v:
            out.append(v)
    # "called X", "named X", "labeled X"
    for m in re.finditer(
        r"\b(?:called|named|labeled|labelled|titled|to)\s+([A-Z][\w \-]{1,40})",
        instruction,
    ):
        out.append(m.group(1).strip().rstrip(".,;:"))
    # Capitalized multi-word run as a fallback (skip if already in quoted).
    for m in re.finditer(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){1,4})\b", instruction):
        out.append(m.group(0).strip())
    # Deduplicate, keep order, drop tiny.
    seen = set()
    uniq = []
    for v in out:
        if len(v) < 2:
            continue
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return uniq


def propose(
    instruction: str,
    candidates: list[Candidate],
    *,
    top_k: int = 5,
    used_keys: frozenset[str] = frozenset(),
) -> list[ScoredAction]:
    """Score and return the top-K untried actions for the current page state.

    ``used_keys`` are ``Action.key()`` values already taken on the current
    path — they get a recency penalty (not banned outright, since some
    actions like "click toggle X" are reversible and reapplying matters).
    """
    qtoks = tokenize(instruction)
    fills = _value_candidates(instruction)

    scored: list[ScoredAction] = []
    for c in candidates:
        ctext = _candidate_text(c)
        ctoks = tokenize(ctext)
        base = _bm25_overlap(qtoks, ctoks)

        # Boosts.
        boost = 0.0
        if c.attrs.get("data-action"):
            da_toks = tokenize(c.attrs["data-action"])
            if any(t in qtoks for t in da_toks):
                boost += 0.5
        if c.attrs.get("data-testid"):
            boost += 0.05
        if c.tag in ("button",) and any(
            v in qtoks for v in ("click", "open", "send", "discard", "save", "create", "add", "remove")
        ):
            boost += 0.3
        if c.tag in ("input", "textarea") and any(
            v in qtoks for v in ("type", "enter", "name", "named", "called", "labeled", "labelled", "search")
        ):
            boost += 0.4

        # Navigation prior: tiny baseline boost for elements that look like
        # they take you to a new section (Settings, dropdown triggers, tabs,
        # menus). Big enough to break the all-zero-score symmetry; small
        # enough not to drown out genuine text matches.
        if any(t in _NAV_TOKENS for t in ctoks):
            boost += 0.15
        # Custom dropdown trigger: a clickable thing that opens a value list.
        if c.attrs.get("data-dropdown"):
            boost += 0.10
        # Dropdown items (data-value + data-dropdown-id) look like terminals;
        # if their value-text matches the instruction, score very high.
        if c.attrs.get("data-value"):
            v_toks = tokenize(c.attrs["data-value"])
            if any(t in qtoks for t in v_toks):
                boost += 0.7

        # Click variant (always available for any clickable element).
        click = Action(kind="click", selector=c.selector, label=c.text or c.attrs.get("aria-label", ""))
        penalty = 0.3 if click.key() in used_keys else 0.0
        scored.append(ScoredAction(click, base + boost - penalty))

        # Fill variants (only for fillable elements).
        if c.kind == "fill":
            for v in fills:
                fa = Action(kind="fill", selector=c.selector, value=v, label=c.text)
                penalty = 0.3 if fa.key() in used_keys else 0.0
                # Fill score ≈ candidate base + a flat fill bonus + value-token overlap.
                vtoks = tokenize(v)
                voverlap = _bm25_overlap(qtoks, vtoks)
                scored.append(ScoredAction(fa, base + boost + 0.6 + 0.4 * voverlap - penalty))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Softmax sampling for simulation rollouts
# ---------------------------------------------------------------------------


def softmax_sample(scored: list[ScoredAction], rng) -> ScoredAction | None:
    """Sample one action proportional to exp(score). Used in MCTS simulate()."""
    if not scored:
        return None
    if len(scored) == 1:
        return scored[0]
    # Numerical stability.
    smax = max(s.score for s in scored)
    weights = [math.exp(s.score - smax) for s in scored]
    total = sum(weights)
    if total <= 0:
        return rng.choice(scored)
    r = rng.uniform(0, total)
    acc = 0.0
    for s, w in zip(scored, weights):
        acc += w
        if r <= acc:
            return s
    return scored[-1]
