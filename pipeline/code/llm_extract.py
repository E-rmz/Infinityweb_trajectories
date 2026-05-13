"""Stage A (paraphrase) + Stage B (analysis) with a content-hash cache.

Cache key = sha256(model | prompt). Re-running is free.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from . import llm_config
from .llm_client import LLMClient, get_client

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / ".cache" / "llm_extract"

_STAGE_A_TEMPLATE = """\
SYSTEM:
You rewrite atomic web-UI task instructions as parameterized templates.

INSTRUCTION:
{instruction}

Output JSON only:
1. "intent_skeleton": instruction with concrete entities replaced by
   role-based placeholders in curly braces. Use snake_case role names:
   {{sender}}, {{topic}}, {{label_name}}, {{setting_value}}, {{target_id}}, ...
2. "entities": object mapping each placeholder to its literal value from
   the instruction.

Constraints:
- Every placeholder in "intent_skeleton" must appear as a key in "entities".
- Every value in "entities" must appear verbatim in the original INSTRUCTION.
- Don't introduce placeholders for verbs, articles, or non-variable nouns.
- Output a single JSON object, nothing else.
"""

_STAGE_B_TEMPLATE = """\
SYSTEM:
You analyze a programmatic verifier and emit a structured effect description.
The verifier reads server state via GET /api/state and returns (bool, str).

INSTRUCTION:
{instruction}

VERIFIER SOURCE:
{verifier_source}

KNOWN STATE KEYS FOR THIS APP (from schema map):
{state_keys}

KNOWN VERBS PER KEY:
{key_semantics}

Output JSON only with all of:

1. "effect_summary": one sentence in prose using state path syntax.
   Example: "Sets emails[id=1].isStarred = True".

2. "affects_paths": list of canonical paths the verifier requires. Grammar:
       SEGMENT  = IDENT ( '[' SELECTOR ']' )?
       PATH     = SEGMENT ( '.' SEGMENT )*
   Canonical forms:
     - "emails[id=1].isStarred"         entity field
     - "labels[name='Q1 Priority']"     entity existence
     - "settings.theme"                 plain object field
     - "filters[*new]"                  newly-created entity
     - "filters[*new].actions.label"    new entity's field
     - "emails[isSpam=True]"            predicate (set of entities)
   Rules:
     - Every root segment MUST appear in KNOWN STATE KEYS.
     - Never emit a bare root with no selector or field (use [*new] for create).
     - Never emit an empty selector "filters[]" — write "filters[*new]".
     - Never use a numeric positional index "settings.x[0]" — use [*new] when
       the verifier accepts any index, or [N=...] when a specific value matches.

3. "effect_kind": EXACTLY one of these six labels — pick the most specific.
     - "create" : add a new entity (label, filter, contact, draft, …).
     - "delete" : remove an entity outright (NOT trash/archive — those are toggle).
     - "set"    : write a target value into a non-boolean field
                  (e.g., settings.theme = "dark", labels[id].color = "red").
     - "toggle" : flip a single boolean-ish flag on an entity. Use this for:
                  Star / Unstar      → emails[id].isStarred
                  Archive / Unarchive→ emails[id].isArchived
                  Trash / Untrash    → emails[id].isTrashed
                  Mark read / unread → emails[id].isRead
                  Snooze / Unsnooze  → emails[id].isSnoozed
                  Mark important / unmark → emails[id].isImportant
                  Mark spam / unspam → emails[id].isSpam
                  Turn on/off any boolean setting → settings.<bool>
                  These are TOGGLE even if the verifier phrases the change as
                  an "update". Do not say "update" for a single bool flip.
     - "update" : modify a non-boolean field on an existing entity, or
                  add/remove elements from an array field
                  (e.g., emails[id].labels list).
     - "send"   : explicitly send an email/message.

4. "bound_entities": what the verifier source HARDCODES that downstream
   composition needs to know. Use ONLY keys from this canonical vocabulary
   (extending only when none fits):
       target_email_id            (int or list[int])
       target_email_from          (sender literal)
       target_email_subject       (subject literal or substring)
       target_label_id            ("label_NN")
       target_label_name          (human label text)
       target_parent_label_id     ("label_NN" for nested labels)
       target_filter_id           ("filter_NN")
       target_filter_criteria_from(email-address substring used in criteria)
       target_forward_email       (filter action.forward target)
       target_setting_key         (e.g. "theme", "density", "undoSendDelay")
       target_setting_value       (str | bool | int the setting must equal)
       target_category            ("primary"|"social"|"promotions"|…)
       target_mark_read           (bool)
       target_archive             (bool)
   Rules:
     - If the verifier targets a setting, ALWAYS emit BOTH target_setting_key
       AND target_setting_value (never invent a per-setting key like
       "target_theme" or "target_density").
     - If the verifier targets a SET of emails (predicate), use
       target_email_id with a list value.
     - "bound_entities" should be {{}} only when the verifier truly targets
       no specific instance (e.g. "empty the spam folder").
   Values must be literals copied from the verifier source.

5. "semantic_actions": list of snake_case action-type strings (verb_object)
   the agent would plausibly perform. Use a stable vocabulary:
     star_email, unstar_email,
     archive_email, unarchive_email,
     trash_email, untrash_email,
     mark_email_read, mark_email_unread,
     mark_important, unmark_important,
     snooze_email, unsnooze_email,
     spam_email, unspam_email,
     apply_label, remove_label,
     create_label, delete_label, update_label,
     create_filter, update_filter, delete_filter,
     open_settings, update_settings,
     move_email_category, move_to_inbox,
     send_email, save_draft, discard_draft
   Use one entry per logically-distinct action. Do not invent a new verb when
   an existing one fits.

Return one JSON object, nothing else.
"""


def _cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}|{prompt}".encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _extract_json(text: str) -> dict:
    """Best-effort JSON object extraction from LLM output."""
    text = text.strip()
    # strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _call_with_cache(
    prompt: str,
    *,
    client: LLMClient | None = None,
    force_refresh: bool = False,
    no_llm: bool = False,
) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(prompt, llm_config.MODEL_SLUG)
    path = _cache_path(key)
    if path.exists() and not force_refresh:
        return json.loads(path.read_text())
    if no_llm:
        raise RuntimeError(
            f"Cache miss with --no-llm. Key={key[:8]}… prompt_head={prompt[:80]!r}"
        )
    client = client or get_client()
    raw = client.complete(prompt)
    parsed = _extract_json(raw)
    path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
    return parsed


def stage_a(
    instruction: str,
    *,
    client: LLMClient | None = None,
    force_refresh: bool = False,
    no_llm: bool = False,
) -> dict[str, Any]:
    prompt = _STAGE_A_TEMPLATE.format(instruction=instruction)
    return _call_with_cache(
        prompt, client=client, force_refresh=force_refresh, no_llm=no_llm
    )


def stage_b(
    instruction: str,
    verifier_source: str,
    app_schema: dict,
    *,
    client: LLMClient | None = None,
    force_refresh: bool = False,
    no_llm: bool = False,
) -> dict[str, Any]:
    prompt = _STAGE_B_TEMPLATE.format(
        instruction=instruction,
        verifier_source=verifier_source,
        state_keys=json.dumps(app_schema.get("state_keys", []), indent=2),
        key_semantics=json.dumps(app_schema.get("key_semantics", {}), indent=2),
    )
    return _call_with_cache(
        prompt, client=client, force_refresh=force_refresh, no_llm=no_llm
    )
