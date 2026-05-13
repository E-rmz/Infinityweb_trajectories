"""9-rule validation pass over extracted templates.

Outputs structured rows so triage is greppable. Failures land in
`pipeline/data/templates/<app>.validation.json`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .normalize import CANONICAL_BOUND_KEYS, is_canonical_path
from .schema import AtomicTemplate, EFFECT_KINDS
from .utils import atomic_write_json

_PLACEHOLDER_RE = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b")
_SNAKE_VERB_NOUN_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")

_EFFECT_KIND_VERB_HINTS: dict[str, tuple[str, ...]] = {
    # Aligned with the Stage B prompt's effect_kind taxonomy:
    #   - "trash" / "archive" / "snooze" / "mark" are TOGGLE (boolean flips)
    #   - "delete" / "remove" are reserved for entity removal (delete a filter,
    #     label, contact — not an email move-to-trash).
    "create": ("create", "add", "new"),
    "delete": ("delete", "remove", "discard", "empty"),
    "set":    ("set", "switch", "select", "configure", "change"),
    "toggle": ("toggle", "star", "unstar", "trash", "untrash",
               "archive", "unarchive", "snooze", "unsnooze",
               "mark", "unmark", "spam", "unspam", "important",
               "turn on", "turn off", "enable", "disable",
               "flip", "flag", "unflag"),
    "update": ("update", "modify", "rename", "edit"),
    "send":   ("send", "submit", "reply", "forward"),
}

# Verbs that strongly suggest a toggle action (boolean flip), used by D12.
_TOGGLE_VERBS: tuple[str, ...] = (
    "star", "unstar", "archive", "unarchive", "trash", "untrash",
    "snooze", "unsnooze", "mark", "unmark",
    "spam", "unspam", "important", "unimportant",
    "turn on", "turn off", "enable", "disable", "flag", "unflag",
    "read", "unread",
)

# A boolean-shaped final segment in an affects_paths entry (heuristic).
_BOOL_FIELD_RE = re.compile(r"\.(is[A-Z][a-zA-Z]*|markRead|hasAttachment)$")

# Known array-valued fields whose mutation (add/remove element) is naturally
# `effect_kind="update"` even though the instruction verb is "add" / "remove".
# Used by D9 to skip false-alarm verb-mismatch warnings.
_ARRAY_FIELD_TAILS: tuple[str, ...] = (
    ".labels",                  # emails[id].labels
    ".categories",              # inbox categories list
    ".inboxCategories",
    ".multipleInboxSections",   # ordered config list
)

# A predicate-style selector in an affects_paths entry — i.e. the path
# targets a *set* of entities, not a single instance. Examples:
#   emails[isSpam=True]    emails[isStarred=True]    filters[*new]
# `bound_entities` is *allowed* to be empty for predicate-style tasks.
_PREDICATE_SEL_RE = re.compile(
    r"\[(?:\*new|[a-zA-Z_][a-zA-Z0-9_]*=[^]]+)\]"
)


def _has_predicate_or_create_path(t: AtomicTemplate) -> bool:
    if any(_PREDICATE_SEL_RE.search(p) for p in t.affects_paths):
        return True
    # Also: pure entity-creation has no instance to bind yet.
    if t.effect_kind == "create":
        return True
    return False


def _all_paths_target_array_field(t: AtomicTemplate) -> bool:
    if not t.affects_paths:
        return False
    return all(
        any(p.endswith(tail) for tail in _ARRAY_FIELD_TAILS)
        for p in t.affects_paths
    )


@dataclass
class ValidationIssue:
    task_id: str
    check_id: str
    severity: str  # "error" | "warn"
    message: str


# ── Per-check predicates ────────────────────────────────────────────────────

def _check_1_skeleton_has_all_entity_keys(t: AtomicTemplate) -> list[ValidationIssue]:
    placeholders = set(_PLACEHOLDER_RE.findall(t.intent_skeleton))
    missing = [k for k in t.entities if k not in placeholders]
    if missing:
        return [ValidationIssue(t.task_id, "C1", "error",
            f"entities keys not in intent_skeleton: {missing}")]
    return []


def _check_2_entities_values_in_raw(t: AtomicTemplate) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    for k, v in t.entities.items():
        if isinstance(v, str) and v and v not in t.intent_raw:
            out.append(ValidationIssue(t.task_id, "C2", "error",
                f"entities[{k!r}]={v!r} not found verbatim in intent_raw"))
    return out


def _check_3_skeleton_has_no_concrete_caps(t: AtomicTemplate) -> list[ValidationIssue]:
    # Strip placeholders, then look for capitalized multi-word phrases.
    stripped = _PLACEHOLDER_RE.sub("__PH__", t.intent_skeleton)
    matches = [m.group(1) for m in _CAP_PHRASE_RE.finditer(stripped)]
    leaks = [m for m in matches if m not in t.entities.values()]
    if leaks:
        return [ValidationIssue(t.task_id, "C3", "warn",
            f"possible un-templated concrete phrase(s) in intent_skeleton: {leaks}")]
    return []


def _check_4_effect_kind_in_vocab(t: AtomicTemplate) -> list[ValidationIssue]:
    if t.effect_kind not in EFFECT_KINDS:
        return [ValidationIssue(t.task_id, "D4", "error",
            f"effect_kind={t.effect_kind!r} not in {list(EFFECT_KINDS)}")]
    return []


def _check_5_paths_root_in_schema(t: AtomicTemplate, app_schema: dict) -> list[ValidationIssue]:
    known = set(app_schema.get("state_keys", []))
    out: list[ValidationIssue] = []
    for p in t.affects_paths:
        root = re.split(r"[\[.]", p, maxsplit=1)[0]
        if root not in known:
            out.append(ValidationIssue(t.task_id, "D5", "error",
                f"affects_paths root {root!r} not in app state_keys"))
    return out


def _check_6_paths_root_in_state_reads(t: AtomicTemplate) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    reads = set(t.state_reads)
    for p in t.affects_paths:
        root = re.split(r"[\[.]", p, maxsplit=1)[0]
        if root not in reads:
            out.append(ValidationIssue(t.task_id, "D6", "warn",
                f"affects_paths root {root!r} not in AST state_reads={sorted(reads)}"))
    return out


def _check_7_bound_entities_nonempty(t: AtomicTemplate) -> list[ValidationIssue]:
    if t.bound_entities:
        return []
    # Empty bound_entities is OK for predicate-style or pure-create tasks —
    # no specific instance exists to bind yet. Step 1 will treat those as
    # wildcard for the compatibility predicate.
    if _has_predicate_or_create_path(t):
        return []
    return [ValidationIssue(t.task_id, "D7", "warn",
        "bound_entities is empty — composition compatibility predicate will be a no-op")]


def _check_8_semantic_actions_shape(t: AtomicTemplate) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    if not t.semantic_actions:
        out.append(ValidationIssue(t.task_id, "D8", "error",
            "semantic_actions is empty"))
        return out
    for a in t.semantic_actions:
        if not _SNAKE_VERB_NOUN_RE.match(a):
            out.append(ValidationIssue(t.task_id, "D8", "error",
                f"semantic_actions entry {a!r} is not snake_case verb_noun"))
    return out


def _check_9_effect_kind_plausible(t: AtomicTemplate) -> list[ValidationIssue]:
    if t.effect_kind not in _EFFECT_KIND_VERB_HINTS:
        return []
    # Mutating elements inside a known array field (e.g. emails[id].labels)
    # is naturally `update` even when the verb is "add" or "remove".
    if t.effect_kind == "update" and _all_paths_target_array_field(t):
        return []
    hints = _EFFECT_KIND_VERB_HINTS[t.effect_kind]
    blob = (t.effect_summary + " " + t.intent_raw).lower()
    if not any(h in blob for h in hints):
        return [ValidationIssue(t.task_id, "D9", "warn",
            f"effect_kind={t.effect_kind!r} not corroborated by any hint verb "
            f"in summary/instruction; hints={list(hints)}")]
    return []


def _check_10_bound_keys_canonical(t: AtomicTemplate) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    for k in t.bound_entities:
        if k not in CANONICAL_BOUND_KEYS:
            out.append(ValidationIssue(t.task_id, "D10", "warn",
                f"bound_entities key {k!r} outside canonical vocabulary"))
    return out


def _check_11_paths_canonical_shape(t: AtomicTemplate) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    for p in t.affects_paths:
        if not is_canonical_path(p):
            out.append(ValidationIssue(t.task_id, "D11", "warn",
                f"affects_paths entry {p!r} not in canonical SEGMENT(.SEGMENT)* form"))
    return out


def _check_12_bool_flip_should_be_toggle(t: AtomicTemplate) -> list[ValidationIssue]:
    """Single boolean-shaped affects_path + toggle-verb instruction but kind != toggle."""
    if t.effect_kind == "toggle":
        return []
    bool_paths = [p for p in t.affects_paths if _BOOL_FIELD_RE.search(p)]
    if not bool_paths:
        return []
    if len(t.affects_paths) > len(bool_paths):
        # mixed (e.g. isStarred + starType); don't flag — Stage B is right to
        # avoid "toggle" when more than one field is required.
        return []
    instr = t.intent_raw.lower()
    if any(v in instr for v in _TOGGLE_VERBS):
        return [ValidationIssue(t.task_id, "D12", "warn",
            f"effect_kind={t.effect_kind!r} but affects_paths is a single boolean "
            f"flip ({bool_paths!r}) with a toggle-verb instruction — likely 'toggle'")]
    return []


_CHECKS_PER_TEMPLATE = (
    _check_1_skeleton_has_all_entity_keys,
    _check_2_entities_values_in_raw,
    _check_3_skeleton_has_no_concrete_caps,
    _check_4_effect_kind_in_vocab,
    _check_7_bound_entities_nonempty,
    _check_8_semantic_actions_shape,
    _check_9_effect_kind_plausible,
    _check_10_bound_keys_canonical,
    _check_11_paths_canonical_shape,
    _check_12_bool_flip_should_be_toggle,
)

_CHECKS_WITH_SCHEMA = (
    _check_5_paths_root_in_schema,
)

_CHECKS_WITH_STATE_READS = (
    _check_6_paths_root_in_state_reads,
)


def validate_template(t: AtomicTemplate, app_schema: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for check in _CHECKS_PER_TEMPLATE:
        issues.extend(check(t))
    for check in _CHECKS_WITH_SCHEMA:
        issues.extend(check(t, app_schema))
    for check in _CHECKS_WITH_STATE_READS:
        issues.extend(check(t))
    return issues


def run_validators(
    templates: Iterable[AtomicTemplate], app_schema: dict
) -> dict:
    templates = list(templates)
    all_issues: list[ValidationIssue] = []
    failing_ids: set[str] = set()
    for t in templates:
        issues = validate_template(t, app_schema)
        for i in issues:
            all_issues.append(i)
            if i.severity == "error":
                failing_ids.add(t.task_id)
    return {
        "summary": {
            "n_templates": len(templates),
            "n_errors": sum(1 for i in all_issues if i.severity == "error"),
            "n_warns": sum(1 for i in all_issues if i.severity == "warn"),
            "n_failing_templates": len(failing_ids),
            "failing_task_ids": sorted(failing_ids),
        },
        "issues": [asdict(i) for i in all_issues],
    }


def write_validation_report(out_dir: Path, app: str, report: dict) -> Path:
    path = Path(out_dir) / f"{app}.validation.json"
    atomic_write_json(path, report)
    return path


if __name__ == "__main__":
    # Smoke: hand-craft a deliberately-broken template
    good = AtomicTemplate(
        task_id="task_e1", app="gmail", difficulty="easy",
        intent_raw="Star Sarah Chen's Q1 product roadmap email.",
        intent_skeleton="Star {sender}'s {topic} email.",
        entities={"sender": "Sarah Chen", "topic": "Q1 product roadmap"},
        verifier_path="real-tasks/task_e1.py",
        state_reads=["emails"],
        effect_summary="Sets emails[id=1].isStarred = True.",
        affects_paths=["emails[id=1].isStarred"],
        effect_kind="toggle",
        bound_entities={"target_email_id": 1},
        semantic_actions=["search_email", "click_star"],
        estimated_min_actions=1,
    )
    bad = AtomicTemplate(
        task_id="task_bad", app="gmail", difficulty="easy",
        intent_raw="Star an email.",
        intent_skeleton="Star Sarah Chen's email.",            # leaks → C3
        entities={"missing": "x"},                              # → C1, C2
        verifier_path="real-tasks/x.py",
        state_reads=["emails"],
        effect_summary="Does something.",
        affects_paths=["nonexistent_root.foo"],                 # → D5, D6
        effect_kind="bogus",                                    # → D4
        bound_entities={},                                      # → D7
        semantic_actions=["BadAction", ""],                     # → D8
        estimated_min_actions=1,
    )
    schema = {"state_keys": ["emails", "labels"]}
    report = run_validators([good, bad], schema)
    print("summary:", report["summary"])
    print("issue check_ids:", sorted({i["check_id"] for i in report["issues"]}))
