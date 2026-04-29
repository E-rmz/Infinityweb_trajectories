"""Render a self-contained ``task.py`` (verify + solver) from a ``StateDiff``.

The output mirrors the shape of hand-written tasks under ``apps/gmail/real-tasks``
plus an inline ``solver(state)`` modelled on ``apps/gmail/sanity_check_real.py``.

The generated module exposes:

* ``INSTRUCTION``  — draft string from ``instruction.py``
* ``SOURCE``       — provenance dict (env, task_id, path_id, state_hash)
* ``verify(server_url) -> tuple[bool, str]`` — fetches state and asserts
* ``solver(state) -> None`` — mutates ``state`` to the terminal shape so the
  golden-path arm of the validator can push it back to the server.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from .diff import (
    LIST_FIELDS,
    SINGLETON_FIELDS,
    EntityCreated,
    EntityRemoved,
    FieldChange,
    StateDiff,
    _list_id,
)


# ---------------------------------------------------------------------------
# Per-entity-type accessor metadata (used by both verifier and solver)
# ---------------------------------------------------------------------------

# (entity_type) → key field used inside its list. blockedSenders uses email,
# everything else uses numeric/string id.
_KEY_FIELD: dict[str, str] = {
    "emails": "id",
    "labels": "id",
    "filters": "id",
    "contacts": "id",
    "blockedSenders": "email",
}


def _is_listed(entity_type: str) -> bool:
    return entity_type in LIST_FIELDS


def _is_singleton(entity_type: str) -> bool:
    return entity_type in SINGLETON_FIELDS


def _entity_key_value(entity_key: tuple[tuple[str, Any], ...]) -> Any:
    """For a single-key entity (like ``(('id', 5),)``) return the bare value."""
    if len(entity_key) == 1:
        return entity_key[0][1]
    return None


# ---------------------------------------------------------------------------
# Helpers and constants emitted into the generated module
# ---------------------------------------------------------------------------

_HELPERS_BLOCK = '''\
def _get_path(obj, path):
    cur = obj
    for p in path:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


def _set_path(obj, path, value):
    cur = obj
    for p in path[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[path[-1]] = value


def _find_in_list(state, entity_type, key_field, key_value):
    return next(
        (x for x in state.get(entity_type, []) if x.get(key_field) == key_value),
        None,
    )


def _find_label_by_name(state, name, label_type="user"):
    return next(
        (l for l in state.get("labels", [])
         if l.get("name") == name and l.get("type") == label_type),
        None,
    )


def _find_filter_by_criteria(state, criteria_from, criteria_subject):
    for f in state.get("filters", []):
        crit = f.get("criteria") or {}
        if crit.get("from") == criteria_from and crit.get("subject") == criteria_subject:
            return f
    return None
'''


# ---------------------------------------------------------------------------
# Field-change assertions (verifier)
# ---------------------------------------------------------------------------


def _render_field_change_assertion(c: FieldChange) -> list[str]:
    """Emit verify-side lines for one FieldChange."""
    et = c.entity_type
    fp_str = ".".join(c.field_path) if c.field_path else "<self>"
    target = repr(c.terminal_value)

    if _is_listed(et):
        kfield = _KEY_FIELD.get(et, "id")
        kval = _entity_key_value(c.entity_key)
        return [
            f'    item = _find_in_list(state, {et!r}, {kfield!r}, {kval!r})',
            f'    if item is None:',
            f'        return False, "{et}[{kfield}={kval!r}] not found in state."',
            f'    actual = _get_path(item, {tuple(c.field_path)!r})',
            f'    if actual != {target}:',
            f'        return False, ('
            f'"{et}[{kfield}={kval!r}].{fp_str} expected " + repr({target}) + '
            f'", got " + repr(actual))',
        ]
    if _is_singleton(et):
        return [
            f'    actual = _get_path(state.get({et!r}) or {{}}, {tuple(c.field_path)!r})',
            f'    if actual != {target}:',
            f'        return False, ('
            f'"{et}.{fp_str} expected " + repr({target}) + '
            f'", got " + repr(actual))',
        ]
    # Top-level scalar (rare; should be no-op since most are denied).
    return [
        f'    actual = state.get({et!r})',
        f'    if actual != {target}:',
        f'        return False, ('
        f'"{et} expected " + repr({target}) + ", got " + repr(actual))',
    ]


# ---------------------------------------------------------------------------
# Created / removed entity assertions
# ---------------------------------------------------------------------------


def _render_created_assertion(c: EntityCreated, var_idx: int) -> tuple[list[str], str]:
    """Emit lines that bind ``new_<idx>`` to the created entity if found.

    Returns (lines, var_name). The var_name lets later field assertions
    on the new entity refer to it. (For now we just check existence; field
    assertions on the new entity ride along via subsequent FieldChange
    entries that target the same listed key — which only show up when the
    entity exists in seed too. So created-entities have all their fields
    captured in entity_data, and we assert each one.)
    """
    var = f"new_{var_idx}"
    et = c.entity_type
    if et == "labels":
        name = next(v for k, v in c.stable_key if k == "name")
        ltype = next((v for k, v in c.stable_key if k == "type"), "user")
        lines = [
            f'    {var} = _find_label_by_name(state, {name!r}, {ltype!r})',
            f'    if {var} is None:',
            f'        return False, "Expected new label name={name!r} (type={ltype!r}) was not created."',
        ]
    elif et == "filters":
        cf = next((v for k, v in c.stable_key if k == "criteria.from"), None)
        cs = next((v for k, v in c.stable_key if k == "criteria.subject"), None)
        lines = [
            f'    {var} = _find_filter_by_criteria(state, {cf!r}, {cs!r})',
            f'    if {var} is None:',
            f'        return False, "Expected new filter (from={cf!r}, subject={cs!r}) was not created."',
        ]
    elif et == "blockedSenders":
        em = next(v for k, v in c.stable_key if k == "email")
        lines = [
            f'    {var} = _find_in_list(state, "blockedSenders", "email", {em!r})',
            f'    if {var} is None:',
            f'        return False, "Expected blocked sender {em!r} was not created."',
        ]
    elif et == "emails":
        subj = next((v for k, v in c.stable_key if k == "subject"), None)
        frm = next((v for k, v in c.stable_key if k == "from"), None)
        lines = [
            f'    {var} = next((e for e in state.get("emails", [])'
            f' if e.get("subject") == {subj!r} and e.get("from") == {frm!r}), None)',
            f'    if {var} is None:',
            f'        return False, "Expected new email (subject={subj!r}, from={frm!r}) was not created."',
        ]
    elif et == "contacts":
        em = next(v for k, v in c.stable_key if k == "email")
        lines = [
            f'    {var} = _find_in_list(state, "contacts", "email", {em!r})',
            f'    if {var} is None:',
            f'        return False, "Expected new contact {em!r} was not created."',
        ]
    else:
        # Generic id-based fallback.
        idv = next((v for k, v in c.stable_key if k == "id"), None)
        lines = [
            f'    {var} = _find_in_list(state, {et!r}, "id", {idv!r})',
            f'    if {var} is None:',
            f'        return False, "Expected new {et} id={idv!r} was not created."',
        ]

    # Assert every field of the created entity matches what MCTS produced.
    # We skip volatile id-like and derived counts.
    skip_keys = {"id"}
    if et == "labels":
        skip_keys |= {"messageCount", "unreadCount"}
    for k, v in sorted(c.entity_data.items()):
        if k in skip_keys:
            continue
        lines.append(
            f'    if {var}.get({k!r}) != {v!r}:'
        )
        lines.append(
            f'        return False, '
            f'"{et}[{k}] mismatch on created entity: expected " + repr({v!r}) + '
            f'", got " + repr({var}.get({k!r}))'
        )
    return lines, var


def _render_removed_assertion(c: EntityRemoved) -> list[str]:
    et = c.entity_type
    kfield = _KEY_FIELD.get(et, "id")
    kval = _entity_key_value(c.entity_key)
    return [
        f'    if _find_in_list(state, {et!r}, {kfield!r}, {kval!r}) is not None:',
        f'        return False, "{et}[{kfield}={kval!r}] should have been removed but is still present."',
    ]


# ---------------------------------------------------------------------------
# Invariant (axis) assertions
# ---------------------------------------------------------------------------


def _render_invariant_block(
    diff: StateDiff, seed_state: dict
) -> list[str]:
    """For each touched axis on a LISTED entity, assert that all entities not
    in the diffed-set kept their seed value on that axis.

    Singleton axes (settings, currentUser) have no invariant — there's only
    one entity per type.
    """
    out: list[str] = []
    # Group axes by entity_type → list of field_paths and diffed key set.
    by_et: dict[str, dict[tuple[str, ...], set[Any]]] = {}
    for fc in diff.field_changes:
        if not _is_listed(fc.entity_type):
            continue
        et_axes = by_et.setdefault(fc.entity_type, {})
        diffed = et_axes.setdefault(fc.field_path, set())
        diffed.add(_entity_key_value(fc.entity_key))

    for et in sorted(by_et.keys()):
        kfield = _KEY_FIELD.get(et, "id")
        seed_list = seed_state.get(et) or []
        for fp in sorted(by_et[et].keys()):
            diffed_keys = by_et[et][fp]
            # Build seed map by key.
            seed_map = {}
            for item in seed_list:
                k = item.get(kfield)
                v = _walk(item, fp)
                seed_map[k] = v
            fp_str = ".".join(fp)
            out.append(
                f'    _SEED_{et}_{"_".join(fp)} = {seed_map!r}'
            )
            out.append(
                f'    _DIFFED_{et}_{"_".join(fp)} = {set(diffed_keys)!r}'
            )
            out.append(
                f'    for item in state.get({et!r}, []):'
            )
            out.append(
                f'        k = item.get({kfield!r})'
            )
            out.append(
                f'        if k in _DIFFED_{et}_{"_".join(fp)}:'
            )
            out.append(
                f'            continue'
            )
            out.append(
                f'        expected = _SEED_{et}_{"_".join(fp)}.get(k)'
            )
            out.append(
                f'        actual = _get_path(item, {fp!r})'
            )
            out.append(
                f'        if actual != expected:'
            )
            out.append(
                f'            return False, ('
                f'"{et}[{kfield}=" + repr(k) + "].{fp_str} drifted from seed; '
                f'expected " + repr(expected) + ", got " + repr(actual))'
            )
    return out


def _walk(obj: Any, path: tuple[str, ...]) -> Any:
    cur = obj
    for p in path:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _render_solver(diff: StateDiff) -> list[str]:
    """Emit solver(state) lines that apply the diff in-place."""
    lines: list[str] = ["def solver(state):"]

    if diff.is_empty():
        lines.append("    return  # no-op (empty diff)")
        return lines

    # 1) Field changes on existing entities.
    for c in diff.field_changes:
        et = c.entity_type
        if _is_listed(et):
            kfield = _KEY_FIELD.get(et, "id")
            kval = _entity_key_value(c.entity_key)
            lines.append(
                f'    item = _find_in_list(state, {et!r}, {kfield!r}, {kval!r})'
            )
            lines.append(f'    if item is not None:')
            lines.append(
                f'        _set_path(item, {tuple(c.field_path)!r}, {c.terminal_value!r})'
            )
        elif _is_singleton(et):
            lines.append(f'    state.setdefault({et!r}, {{}})')
            lines.append(
                f'    _set_path(state[{et!r}], {tuple(c.field_path)!r}, {c.terminal_value!r})'
            )
        else:
            lines.append(f'    state[{et!r}] = {c.terminal_value!r}')

    # 2) Removed entities.
    for c in diff.removed:
        et = c.entity_type
        kfield = _KEY_FIELD.get(et, "id")
        kval = _entity_key_value(c.entity_key)
        lines.append(
            f'    state[{et!r}] = ['
            f'x for x in state.get({et!r}, []) if x.get({kfield!r}) != {kval!r}]'
        )

    # 3) Created entities. Use server's auto-id counters if listed.
    for c in diff.created:
        et = c.entity_type
        data = dict(c.entity_data)
        # For listed entities, drop the volatile id and let server assign one
        # (mirror sanity_check_real.py: tests just append to the list).
        if et in {"labels", "filters", "emails", "contacts"}:
            data.pop("id", None)
            counter_key = {
                "labels": "_nextLabelId",
                "filters": "_nextFilterId",
                "emails": "_nextEmailId",
            }.get(et)
            if counter_key:
                lines.append(
                    f'    _next = state.get({counter_key!r}, 1)'
                )
                lines.append(
                    f'    state[{counter_key!r}] = _next + 1'
                )
                # Build new entity dict including the assigned id.
                lines.append(
                    f'    _new = dict({data!r})'
                )
                # id format: server uses int for emails, "label_NN" for labels, etc.
                if et == "labels":
                    lines.append(f'    _new["id"] = "label_" + str(_next)')
                elif et == "filters":
                    lines.append(f'    _new["id"] = "filter_" + str(_next)')
                else:
                    lines.append(f'    _new["id"] = _next')
            else:
                lines.append(f'    _new = dict({data!r})')
            lines.append(f'    state.setdefault({et!r}, []).append(_new)')
        elif et == "blockedSenders":
            lines.append(f'    state.setdefault({et!r}, []).append({data!r})')
        else:
            lines.append(f'    state.setdefault({et!r}, []).append({data!r})')

    return lines


# ---------------------------------------------------------------------------
# Verifier rendering
# ---------------------------------------------------------------------------


def _render_verifier(diff: StateDiff, seed_state: dict) -> list[str]:
    lines: list[str] = []
    lines.append("def verify(server_url):")
    lines.append('    r = requests.get(f"{server_url}/api/state")')
    lines.append("    if r.status_code != 200:")
    lines.append('        return False, f"Failed to fetch state: HTTP {r.status_code}"')
    lines.append("    state = r.json()")
    lines.append("")

    if diff.is_empty():
        lines.append("    return True, \"empty diff (no-op trajectory)\"")
        return lines

    # Created entities — check existence and exact field-match first.
    for i, c in enumerate(diff.created):
        block, _ = _render_created_assertion(c, i)
        lines.extend(block)
        lines.append("")

    # Removed entities.
    for c in diff.removed:
        lines.extend(_render_removed_assertion(c))
        lines.append("")

    # Field changes on existing entities.
    for c in diff.field_changes:
        lines.extend(_render_field_change_assertion(c))
        lines.append("")

    # Invariants on touched axes (listed entity types only).
    inv = _render_invariant_block(diff, seed_state)
    if inv:
        lines.append("    # --- invariants on touched axes ---")
        lines.extend(inv)
        lines.append("")

    n_fc = len(diff.field_changes)
    n_cr = len(diff.created)
    n_rm = len(diff.removed)
    n_axes = len({(fc.entity_type, fc.field_path) for fc in diff.field_changes})
    lines.append(
        f'    return True, '
        f'"OK: {n_fc} field assertion(s), {n_cr} created, {n_rm} removed; '
        f'{n_axes} invariant axis/axes preserved."'
    )
    return lines


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def render_task_module(
    *,
    diff: StateDiff,
    seed_state: dict,
    instruction: str,
    source: dict,
) -> str:
    """Return the full ``task.py`` source string."""
    header = f'''\
"""Auto-generated synthetic task.

Source: {source.get("env")}/{source.get("task_id")}/{source.get("path_id")}
Terminal state hash: {source.get("state_hash")}
Generated by thesis/synthetic/codegen.py — do not edit by hand.
"""

import requests

INSTRUCTION = {instruction!r}
DIFFICULTY = "synthetic"
SOURCE = {json.dumps(source, sort_keys=True)}


'''
    body_parts = [
        header,
        _HELPERS_BLOCK,
        "\n",
        "\n".join(_render_verifier(diff, seed_state)),
        "\n\n",
        "\n".join(_render_solver(diff)),
        "\n",
    ]
    return "".join(body_parts)
