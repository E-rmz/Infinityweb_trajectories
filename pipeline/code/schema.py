"""AtomicTemplate dataclass + envelope helpers.

The v1 schema for Step 0 output. Stable contract for Step 1 (composer),
Step 2 (composite verifier), Step 3 (vocabulary).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "0.1.0"

EffectKind = Literal["create", "delete", "set", "toggle", "update", "send"]
EFFECT_KINDS: tuple[str, ...] = (
    "create", "delete", "set", "toggle", "update", "send",
)


@dataclass
class AtomicTemplate:
    # ── Identity ────────────────────────────────────────────────────────────
    task_id: str
    app: str
    difficulty: str

    # ── Natural language ────────────────────────────────────────────────────
    intent_raw: str
    intent_skeleton: str
    entities: dict

    # ── Verifier binding ────────────────────────────────────────────────────
    verifier_path: str
    state_reads: list[str]

    # ── Effect model (Stage B) ──────────────────────────────────────────────
    effect_summary: str
    affects_paths: list[str]
    effect_kind: str  # one of EFFECT_KINDS

    # ── Verifier instance binding ───────────────────────────────────────────
    bound_entities: dict

    # ── Search support ──────────────────────────────────────────────────────
    semantic_actions: list[str]
    estimated_min_actions: int

    # ── QA flag (filled after seed-satisfaction batch) ──────────────────────
    seed_satisfies: bool | None = None


def template_to_dict(t: AtomicTemplate) -> dict:
    return asdict(t)


def dict_to_template(d: dict) -> AtomicTemplate:
    return AtomicTemplate(**d)


def wrap_envelope(app: str, templates: list[AtomicTemplate]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "app": app,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "templates": [template_to_dict(t) for t in templates],
    }


def load_templates(path: Path) -> tuple[str, list[AtomicTemplate]]:
    """Load an envelope from disk; refuse mismatched schema_version."""
    payload = json.loads(Path(path).read_text())
    found = payload.get("schema_version")
    if found != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch in {path}: file={found!r} "
            f"loader={SCHEMA_VERSION!r}. Regenerate the templates."
        )
    return payload["app"], [dict_to_template(d) for d in payload["templates"]]


if __name__ == "__main__":
    sample = AtomicTemplate(
        task_id="task_e1",
        app="gmail",
        difficulty="easy",
        intent_raw="Star Sarah Chen's Q1 product roadmap email.",
        intent_skeleton="Star {sender}'s {topic} email.",
        entities={"sender": "Sarah Chen", "topic": "Q1 product roadmap"},
        verifier_path="real-tasks/task_e1.py",
        state_reads=["emails"],
        effect_summary="Sets emails[id=1].isStarred = True; starType != null.",
        affects_paths=["emails[id=1].isStarred", "emails[id=1].starType"],
        effect_kind="toggle",
        bound_entities={"target_email_id": 1, "target_email_from": "Sarah Chen"},
        semantic_actions=["search_email", "select_email", "click_star"],
        estimated_min_actions=1,
    )
    env = wrap_envelope("gmail", [sample])
    rt = dict_to_template(env["templates"][0])
    assert rt == sample, "round-trip mismatch"
    assert env["schema_version"] == SCHEMA_VERSION
    print("schema.py round-trip OK")
