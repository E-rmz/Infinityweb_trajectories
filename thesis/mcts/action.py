"""Action representation for MCTS over a web app DOM.

An ``Action`` is a single replayable browser interaction. It must be
deterministic (same selector + value → same effect) so the MCTS can
replay any path from seed by issuing ``POST /api/reset`` and re-running
the recorded actions in order.

Selector format
---------------
We use a small, locator-string syntax that we resolve in ``browser.py``
without relying on auto-generated CSS classes:

    "id=settingsBtn"
    "data-action=open-settings"
    "data-testid=compose-to"
    "role=button[name=Compose]"
    "text=Switch to dark mode"
    "css=.toolbar button.refresh"        # last-resort fallback

The first four are stable across renders; the last is a pragmatic escape
hatch for elements without semantic anchors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ActionKind = Literal["click", "fill", "press", "navigate"]


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    selector: str
    value: str | None = None
    label: str = ""

    def key(self) -> str:
        """Canonical hashable repr for tree de-duplication."""
        return f"{self.kind}|{self.selector}|{self.value or ''}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(
            kind=d["kind"],
            selector=d["selector"],
            value=d.get("value"),
            label=d.get("label", ""),
        )

    def __str__(self) -> str:
        v = f"={self.value!r}" if self.value is not None else ""
        return f"{self.kind}({self.selector}){v}"
