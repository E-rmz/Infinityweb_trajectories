"""Canonicalize Stage B output before validation.

Stage B's LLM produces semantically-correct but lexically-drifty output:
the same concept gets different `bound_entities` keys, near-duplicate
`semantic_actions`, and a handful of non-canonical `affects_paths` shapes.
This module merges those drifts into the canonical vocabularies that
Step 1's compatibility predicate, Step 2's persistence check, and Step 3's
vocabulary loader all hard-depend on.

Determinism: every transform is a pure lookup or regex. No LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Canonical bound_entities key vocabulary ─────────────────────────────────
# These are the only keys downstream consumers should ever see.

CANONICAL_BOUND_KEYS: set[str] = {
    # Email
    "target_email_id",          # int | list[int]
    "target_email_from",        # sender literal
    "target_email_subject",     # subject literal or substring
    # Label
    "target_label_id",          # "label_NN"
    "target_label_name",        # human-readable label name
    "target_parent_label_id",   # for nested labels
    # Filter
    "target_filter_id",         # "filter_NN"
    "target_filter_criteria_from",  # email-address substring
    "target_forward_email",     # filter action.forward
    # Setting
    "target_setting_key",       # e.g. "theme", "density"
    "target_setting_value",     # str | bool | int
    # Category (Gmail tabs: primary, social, promotions, …)
    "target_category",
    # Generic
    "target_mark_read",         # bool — filter action
    "target_archive",           # bool — filter action
}

# Direct synonyms — exact key renames.
_BOUND_KEY_SYNONYMS: dict[str, str] = {
    "target_email":                "target_email_id",
    "target_email_ids":            "target_email_id",
    "target_email_subject_substring": "target_email_subject",
    "target_sender_email":         "target_email_from",
    "target_label":                "target_label_id",
    "replacement_label_id":        "target_label_id",
    "category_id":                 "target_category",
    "parent_label_id":             "target_parent_label_id",
    "target_filter_from":          "target_filter_criteria_from",
    # Per-setting keys that the LLM invented — collapse to setting_value
    # (the matching `target_setting_key` is added separately).
    "target_theme":                "target_setting_value",
    "target_density":              "target_setting_value",
    "target_size":                 "target_setting_value",
    "target_undoSendDelay":        "target_setting_value",
    "target_dynamicEmail":         "target_setting_value",
    "target_hoverActions":         "target_setting_value",
    "target_inbox_type":           "target_setting_value",
    "target_categories":           "target_setting_value",
    "target_star_type":            "target_setting_value",
    "target_value":                "target_setting_value",
    # multipleInboxSections — these are structured, not single values, but
    # for v1 we collapse to the matching setting key/value pattern.
    "section_0_query":             "target_setting_value",
    "section_0_name":              "target_setting_value",
    "section_1_query":             "target_setting_value",
    "section_1_name":              "target_setting_value",
    # per-label-id-by-name keys — collapse to a list under target_label_id
    "label_id_work":               "target_label_id",
    "label_id_projects":           "target_label_id",
    "label_id_waiting_for_reply":  "target_label_id",
    # has-attachment is a filter criterion
    "target_has_attachment":       "target_filter_criteria_from",
    # Already-canonical (sanity, no-op):
    **{k: k for k in CANONICAL_BOUND_KEYS},
}

# Per-setting-key inference: when the LLM produced `target_theme: "dark"`,
# we also add `target_setting_key: "theme"`.
_SETTING_KEY_INFERRED_FROM: dict[str, str] = {
    "target_theme":         "theme",
    "target_density":       "density",
    "target_size":          "size",
    "target_undoSendDelay": "undoSendDelay",
    "target_dynamicEmail":  "dynamicEmail",
    "target_hoverActions":  "hoverActions",
    "target_inbox_type":    "inboxType",
    "target_categories":    "inboxCategories",
    "target_star_type":     "starType",
}

# ── Canonical semantic_actions vocabulary ───────────────────────────────────
# These are merged on the way out so Step 3 vocab has no near-duplicates.

_ACTION_SYNONYMS: dict[str, str] = {
    "mark_read_email":     "mark_email_read",
    # update_theme is a special case of update_settings — keep specificity
    # for now since the selectors are likely different DOM paths.
    # (Intentionally NOT collapsed.)
}

# ── Canonical affects_paths shapes ──────────────────────────────────────────

# Path grammar:
#   IDENT       = [a-zA-Z_][a-zA-Z0-9_]*
#   SEGMENT     = IDENT ( '[' SELECTOR ']' )?
#   PATH        = SEGMENT ( '.' SEGMENT )*
# where SELECTOR may itself contain dots, equals, primes, etc.
_IDENT = r"[a-zA-Z_][a-zA-Z0-9_]*"
_SEGMENT = rf"{_IDENT}(?:\[[^\]]+\])?"
_PATH_RE = re.compile(rf"^{_SEGMENT}(?:\.{_SEGMENT})*$")

# Recognize the bare-root form ("filters") for create-style tasks.
_BARE_ROOT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Match an empty selector form: "filters[]" or "filters[].x"
_EMPTY_SEL_RE = re.compile(
    r"^([a-zA-Z_][a-zA-Z0-9_]*)\[\](\.[a-zA-Z0-9_.]+)?$"
)

# Match numeric positional selector: "settings.foo[0].bar" — keep position
# but treat it as a wildcard selector for compatibility.
_NUM_INDEX_RE = re.compile(
    r"(\.[a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]"
)


def _canonicalize_affects_path(p: str, effect_kind: str) -> str:
    """Reshape a single affects_paths entry into a canonical form."""
    p = p.strip()

    # filters[] → filters[*new]  (creating a new entity)
    m = _EMPTY_SEL_RE.match(p)
    if m:
        root, tail = m.group(1), (m.group(2) or "")
        return f"{root}[*new]{tail}"

    # bare root → root[*new] when effect_kind is create-ish
    if _BARE_ROOT_RE.match(p):
        if effect_kind in ("create",):
            return f"{p}[*new]"
        return p  # leave as-is for "the whole thing was reset" cases

    # settings.multipleInboxSections[0].name → settings.multipleInboxSections[*new].name
    if _NUM_INDEX_RE.search(p):
        return _NUM_INDEX_RE.sub(r"\1[*new]", p)

    return p


# ── Effect-kind nudges ──────────────────────────────────────────────────────
# Some Stage B outputs put a boolean-flip task under "update" or "set" when
# "toggle" is closer. The validator (D12) catches this for human review; we
# intentionally DO NOT auto-rewrite effect_kind here, because the LLM is
# sometimes right (the verifier may check more than the boolean field).

# ── Public entry point ──────────────────────────────────────────────────────


@dataclass
class NormalizationResult:
    """What `normalize_stage_b` changed in a single Stage B output."""
    bound_renames:        dict[str, str]   # before_key -> after_key
    bound_setting_added:  list[str]        # newly-injected setting keys
    actions_renames:      dict[str, str]
    paths_rewrites:       dict[str, str]


def normalize_stage_b(stage_b: dict) -> tuple[dict, NormalizationResult]:
    """Return a *new* Stage B dict with canonical fields, plus an audit log."""
    bound = dict(stage_b.get("bound_entities", {}))
    actions = list(stage_b.get("semantic_actions", []))
    paths = list(stage_b.get("affects_paths", []))
    effect_kind = stage_b.get("effect_kind", "")

    renames: dict[str, str] = {}
    added: list[str] = []
    new_bound: dict = {}

    for k, v in bound.items():
        # First: if it's an inferred-from-setting key, record the new setting key
        if k in _SETTING_KEY_INFERRED_FROM:
            setting_key = _SETTING_KEY_INFERRED_FROM[k]
            if "target_setting_key" not in new_bound:
                new_bound["target_setting_key"] = setting_key
                added.append("target_setting_key")
        new_key = _BOUND_KEY_SYNONYMS.get(k, k)
        if new_key != k:
            renames[k] = new_key
        # Merge collisions: keep first-seen value (lists win over scalars)
        if new_key in new_bound:
            cur = new_bound[new_key]
            if isinstance(cur, list) and isinstance(v, list):
                new_bound[new_key] = sorted(set(cur + v),
                                            key=lambda x: (isinstance(x, str), x))
            # otherwise keep the existing value
        else:
            new_bound[new_key] = v

    # semantic_actions: merge synonyms + dedup, preserve order
    action_renames: dict[str, str] = {}
    seen: set[str] = set()
    new_actions: list[str] = []
    for a in actions:
        canon = _ACTION_SYNONYMS.get(a, a)
        if canon != a:
            action_renames[a] = canon
        if canon not in seen:
            seen.add(canon)
            new_actions.append(canon)

    # affects_paths: reshape non-canonical entries
    path_rewrites: dict[str, str] = {}
    new_paths: list[str] = []
    for p in paths:
        cp = _canonicalize_affects_path(p, effect_kind)
        if cp != p:
            path_rewrites[p] = cp
        new_paths.append(cp)

    out = dict(stage_b)
    out["bound_entities"] = new_bound
    out["semantic_actions"] = new_actions
    out["affects_paths"] = new_paths

    return out, NormalizationResult(
        bound_renames=renames,
        bound_setting_added=added,
        actions_renames=action_renames,
        paths_rewrites=path_rewrites,
    )


def is_canonical_path(p: str) -> bool:
    """True if `p` matches the canonical affects_paths grammar."""
    return bool(_PATH_RE.match(p))


if __name__ == "__main__":
    # Smoke: pretend Stage B output with all the drift patterns we saw
    drift = {
        "effect_summary": "Demo",
        "affects_paths": [
            "filters[].criteria.subject",
            "settings.multipleInboxSections[0].name",
            "filters",
            "emails[id=1].isStarred",
        ],
        "effect_kind": "create",
        "bound_entities": {
            "target_email": 5,
            "target_label": "label_3",
            "target_theme": "dark",
            "category_id": "social",
        },
        "semantic_actions": ["mark_read_email", "star_email", "mark_read_email"],
    }
    out, report = normalize_stage_b(drift)
    print("bound_entities:", out["bound_entities"])
    print("semantic_actions:", out["semantic_actions"])
    print("affects_paths:")
    for p in out["affects_paths"]:
        print(f"  {p}  canonical={is_canonical_path(p)}")
    print("audit:", report)
