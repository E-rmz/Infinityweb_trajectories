"""Draft-instruction templater.

Mechanical placeholder until the LLM stage replaces it with natural
prose. We just stitch together the action sequence using ``label`` and
``value`` fields from each ``actions.json`` entry.
"""

from __future__ import annotations


def template_from_actions(actions: list[dict]) -> str:
    """Build a readable synthetic-task instruction from an action list.

    Examples::

        click(email-star-1: ☆)
        fill(compose-to: alice@example.com)
        click(theme-dark)

    Joined with `` → `` between steps and prefixed with ``"Perform: "``.
    """
    if not actions:
        return "Perform: <no actions>"
    parts = [_describe(a) for a in actions]
    return "Perform: " + " → ".join(parts)


def _describe(action: dict) -> str:
    kind = action.get("kind", "?")
    selector = action.get("selector", "")
    label = (action.get("label") or "").strip()
    value = action.get("value")
    short_sel = _short_selector(selector)

    if kind == "fill":
        v = "" if value is None else str(value)
        return f"fill({short_sel}: {v!r})" if v else f"fill({short_sel})"
    if kind == "press":
        return f"press({short_sel}: {value or 'Enter'})"
    if kind == "navigate":
        return f"navigate({value or '?'})"
    # click / unknown
    if label:
        return f"{kind}({short_sel}: {label})"
    return f"{kind}({short_sel})"


def _short_selector(selector: str) -> str:
    """Strip the selector prefix so the instruction reads cleanly."""
    for prefix in (
        "id=",
        "data-action=",
        "data-testid=",
        "data-dropdown=",
        "dropdown-item=",
        "data-snooze=",
        "role=",
        "text=",
        "css=",
    ):
        if selector.startswith(prefix):
            return selector[len(prefix):]
    return selector
