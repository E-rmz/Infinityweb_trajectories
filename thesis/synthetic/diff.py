"""Structural diff between two ``/api/state`` snapshots.

Produces a ``StateDiff`` that ``codegen`` consumes to emit verify/solver code.

Design notes
------------
* Lists are id-keyed via ``_LIST_ID``; ``blockedSenders`` keys on ``email``.
* New entities are reported with a *stable key* (e.g. label ``name`` instead of
  the volatile auto-incremented ``id``) so verifiers/solvers don't depend on
  re-seed counter values.
* The denylists exclude derived counts (``label.messageCount``,
  ``label.unreadCount``) and internal counters (``_nextEmailId`` etc.). These
  are mirrored in ``apps/gmail/sanity_check_real.py``'s seed loader behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Entity-type registry
# ---------------------------------------------------------------------------

# Top-level state keys that hold a list of entities.
LIST_FIELDS: tuple[str, ...] = (
    "emails",
    "labels",
    "filters",
    "contacts",
    "blockedSenders",
)

# Top-level state keys that hold a singleton dict.
SINGLETON_FIELDS: tuple[str, ...] = ("settings", "currentUser")

# Top-level keys excluded from diffing entirely (internal/derived).
TOP_DENY: frozenset[str] = frozenset({
    "_seedVersion",
    "_seedDataVersion",
    "_nextEmailId",
    "_nextLabelId",
    "_nextFilterId",
})

# Per-entity field paths to ignore (these are derived from email state).
ENTITY_DENY: dict[str, frozenset[tuple[str, ...]]] = {
    "labels": frozenset({("messageCount",), ("unreadCount",)}),
}


def _list_id(entity_type: str, item: dict) -> tuple[tuple[str, Any], ...]:
    """Identifier used to match seed and terminal list elements."""
    if entity_type == "blockedSenders":
        return (("email", item.get("email")),)
    return (("id", item.get("id")),)


def _stable_key(entity_type: str, item: dict) -> tuple[tuple[str, Any], ...]:
    """Identifier used in *generated code* for entities created by the path.

    Auto-incremented ids are unstable across re-seed; we look up new entities
    by their semantically meaningful fields instead.
    """
    if entity_type == "labels":
        return (("name", item.get("name")), ("type", item.get("type", "user")))
    if entity_type == "filters":
        crit = item.get("criteria") or {}
        return (
            ("criteria.from", crit.get("from")),
            ("criteria.subject", crit.get("subject")),
        )
    if entity_type == "emails":
        return (
            ("subject", item.get("subject")),
            ("from", item.get("from")),
        )
    if entity_type == "blockedSenders":
        return (("email", item.get("email")),)
    if entity_type == "contacts":
        return (("email", item.get("email")),)
    return (("id", item.get("id")),)


# ---------------------------------------------------------------------------
# Diff datatypes
# ---------------------------------------------------------------------------


_MISSING = object()


@dataclass(frozen=True)
class FieldChange:
    """One field of one entity changed between seed and terminal."""

    entity_type: str
    entity_key: tuple[tuple[str, Any], ...]   # () for singletons (settings, currentUser)
    field_path: tuple[str, ...]
    seed_value: Any
    terminal_value: Any

    def to_dict(self) -> dict:
        return {
            "kind": "field_change",
            "entity_type": self.entity_type,
            "entity_key": [list(p) for p in self.entity_key],
            "field_path": list(self.field_path),
            "seed_value": self.seed_value,
            "terminal_value": self.terminal_value,
        }


@dataclass(frozen=True)
class EntityCreated:
    entity_type: str
    entity_data: dict
    stable_key: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict:
        return {
            "kind": "entity_created",
            "entity_type": self.entity_type,
            "stable_key": [list(p) for p in self.stable_key],
            "entity_data": self.entity_data,
        }


@dataclass(frozen=True)
class EntityRemoved:
    entity_type: str
    entity_key: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict:
        return {
            "kind": "entity_removed",
            "entity_type": self.entity_type,
            "entity_key": [list(p) for p in self.entity_key],
        }


@dataclass
class StateDiff:
    field_changes: list[FieldChange] = field(default_factory=list)
    created: list[EntityCreated] = field(default_factory=list)
    removed: list[EntityRemoved] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.field_changes or self.created or self.removed)

    @property
    def axes(self) -> set[tuple[str, tuple[str, ...]]]:
        """``(entity_type, field_path)`` pairs touched — used for invariants."""
        return {(c.entity_type, c.field_path) for c in self.field_changes}

    def signature(self) -> str:
        """Canonical string for dedup across paths with the same effect."""
        parts: list[str] = []
        for c in sorted(
            self.field_changes,
            key=lambda x: (x.entity_type, x.entity_key, x.field_path),
        ):
            parts.append(
                f"FC|{c.entity_type}|{c.entity_key}|{c.field_path}|"
                f"{c.seed_value!r}|{c.terminal_value!r}"
            )
        for c in sorted(self.created, key=lambda x: (x.entity_type, x.stable_key)):
            parts.append(f"EC|{c.entity_type}|{c.stable_key}")
        for c in sorted(self.removed, key=lambda x: (x.entity_type, x.entity_key)):
            parts.append(f"ER|{c.entity_type}|{c.entity_key}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "field_changes": [c.to_dict() for c in self.field_changes],
            "created": [c.to_dict() for c in self.created],
            "removed": [c.to_dict() for c in self.removed],
        }


# ---------------------------------------------------------------------------
# Diff implementation
# ---------------------------------------------------------------------------


def diff_states(seed: dict, terminal: dict) -> StateDiff:
    """Return a structured diff: terminal - seed, with internal fields dropped."""
    out = StateDiff()
    keys = (set(seed.keys()) | set(terminal.keys())) - TOP_DENY
    for k in sorted(keys):
        if k in LIST_FIELDS:
            _diff_list(out, k, seed.get(k) or [], terminal.get(k) or [])
        elif k in SINGLETON_FIELDS:
            _diff_dict(out, k, (), seed.get(k) or {}, terminal.get(k) or {})
        else:
            _diff_top_scalar(out, k, seed.get(k), terminal.get(k))
    return out


def _diff_top_scalar(out: StateDiff, key: str, sv: Any, tv: Any) -> None:
    if sv == tv:
        return
    out.field_changes.append(
        FieldChange(
            entity_type=key,
            entity_key=(),
            field_path=(),
            seed_value=sv,
            terminal_value=tv,
        )
    )


def _diff_list(
    out: StateDiff,
    entity_type: str,
    seed_list: list[dict],
    term_list: list[dict],
) -> None:
    seed_by_key = {_list_id(entity_type, e): e for e in seed_list}
    term_by_key = {_list_id(entity_type, e): e for e in term_list}
    for k in term_by_key.keys() - seed_by_key.keys():
        el = term_by_key[k]
        out.created.append(
            EntityCreated(
                entity_type=entity_type,
                entity_data=el,
                stable_key=_stable_key(entity_type, el),
            )
        )
    for k in seed_by_key.keys() - term_by_key.keys():
        out.removed.append(EntityRemoved(entity_type=entity_type, entity_key=k))
    for k in seed_by_key.keys() & term_by_key.keys():
        _diff_dict(out, entity_type, k, seed_by_key[k], term_by_key[k])


def _diff_dict(
    out: StateDiff,
    entity_type: str,
    entity_key: tuple[tuple[str, Any], ...],
    seed_el: dict,
    term_el: dict,
    prefix: tuple[str, ...] = (),
) -> None:
    deny = ENTITY_DENY.get(entity_type, frozenset())
    keys = set(seed_el.keys()) | set(term_el.keys())
    for k in sorted(keys):
        path = prefix + (k,)
        if path in deny:
            continue
        sv = seed_el.get(k, _MISSING)
        tv = term_el.get(k, _MISSING)
        if sv == tv:
            continue
        # Recurse into nested dicts (e.g. filter.criteria).
        if isinstance(sv, dict) and isinstance(tv, dict):
            _diff_dict(out, entity_type, entity_key, sv, tv, prefix=path)
            continue
        out.field_changes.append(
            FieldChange(
                entity_type=entity_type,
                entity_key=entity_key,
                field_path=path,
                seed_value=None if sv is _MISSING else sv,
                terminal_value=None if tv is _MISSING else tv,
            )
        )
