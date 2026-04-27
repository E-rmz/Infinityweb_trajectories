"""Playwright wrapper for MCTS browser-action search.

This module owns one Chromium tab per running task and exposes three
high-level operations:

  * ``open_browser(server_url)`` — launch headless Chromium, navigate,
    and wait for the app to push its seed state (``GET /api/state``
    returns 200 once the browser has issued its first ``PUT /api/state``).
  * ``snapshot_candidates(page)`` — harvest visible interactive elements
    with replay-stable selectors (id / data-action / data-testid / role).
  * ``exec_action(page, action)`` — execute one ``Action`` and wait for
    the resulting state PUT to settle.

The wrapper deliberately does **not** restart Chromium between rollouts
(that's a few seconds of overhead). Instead, ``replay.py`` calls
``POST /api/reset`` and we replay the action prefix.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import requests
from playwright.async_api import (
    Browser,
    BrowserContext,
    ElementHandle,
    Page,
    Playwright,
    async_playwright,
)


# ---------------------------------------------------------------------------
# Candidate elements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A single interactive DOM element observed at one tree node."""

    selector: str          # one of: id=, data-action=, data-testid=, role=, text=, css=
    kind: str              # "click" | "fill"
    text: str              # human-readable label (visible text or aria-label)
    tag: str               # html tag name
    attrs: dict            # subset of attributes useful for scoring


# ---------------------------------------------------------------------------
# Browser lifecycle
# ---------------------------------------------------------------------------


class BrowserSession:
    """Owns one persistent Playwright browser + page for a single task run."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._ctx: BrowserContext | None = None
        self.page: Page | None = None

    async def start(self, server_url: str) -> None:
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-extensions",
            ],
        )
        # Use a context so we can clear localStorage cleanly on reset.
        self._ctx = await self._browser.new_context()
        self.page = await self._ctx.new_page()
        await self.page.goto(server_url, wait_until="domcontentloaded")
        # Wait for the browser's first PUT /api/state to land.
        await self._wait_for_seed(server_url)

    async def _wait_for_seed(self, server_url: str, timeout_s: float = 10.0) -> None:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = requests.get(f"{server_url}/api/state", timeout=2)
                if r.status_code == 200 and r.json():
                    return
            except (requests.RequestException, ValueError):
                pass
            await asyncio.sleep(0.25)
        raise RuntimeError(
            "Seed state was never pushed by the browser within "
            f"{timeout_s}s. App may have failed to load."
        )

    async def stop(self) -> None:
        try:
            if self._ctx is not None:
                await self._ctx.close()
            if self._browser is not None:
                await self._browser.close()
            if self._pw is not None:
                await self._pw.stop()
        except Exception:
            pass

    async def screenshot(self, dst: Path) -> None:
        if self.page is None:
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            await self.page.screenshot(path=str(dst), full_page=False)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Candidate enumeration (stable selectors only)
# ---------------------------------------------------------------------------


# JS evaluated in-page to pull a structured list of interactive elements.
# Returns a list of objects with the strongest stable selector for each
# element. We collect:
#   * id=...           when an element has a non-empty id
#   * data-action=...  when an element has data-action (gmail uses this heavily)
#   * data-testid=...  next-best stable hook
#   * role=...[name=]  via accessible role + name (no LLM needed)
#   * text=...         visible text fallback
#   * css=...          last-resort css selector with nth-of-type
#
# Visibility is enforced via offsetParent (skip display:none / detached).
_HARVEST_JS = r"""
() => {
    const results = [];

    function isVisible(el) {
        if (!el) return false;
        if (el.hidden) return false;
        // offsetParent is null for display:none, fixed-positioned ancestors not in DOM, etc.
        if (el.offsetParent === null && el.tagName !== 'BODY') return false;
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return false;
        return true;
    }

    function visibleText(el) {
        // Prefer aria-label, then placeholder, then trimmed innerText (cap len).
        const aria = el.getAttribute('aria-label');
        if (aria && aria.trim()) return aria.trim();
        const ph = el.getAttribute('placeholder');
        if (ph && ph.trim()) return ph.trim();
        const tt = el.getAttribute('title');
        if (tt && tt.trim()) return tt.trim();
        const t = (el.innerText || el.textContent || '').trim();
        return t.length > 80 ? t.slice(0, 80) : t;
    }

    function inferKind(el) {
        const tag = el.tagName.toLowerCase();
        if (tag === 'input') {
            const type = (el.getAttribute('type') || 'text').toLowerCase();
            if (['checkbox', 'radio', 'button', 'submit', 'reset'].includes(type)) return 'click';
            return 'fill';
        }
        if (tag === 'textarea') return 'fill';
        if (tag === 'select') return 'click';   // these apps avoid native <select>, but fall back to click
        return 'click';
    }

    function selectorFor(el) {
        // Priority: id > data-action > custom-dropdown anchors > data-testid >
        //           aria-role+name > visible text > css.
        const id = el.id;
        if (id && /^[A-Za-z][A-Za-z0-9_-]*$/.test(id)) {
            return 'id=' + id;
        }
        // Custom dropdown trigger: <div class="dropdown-trigger" data-dropdown="X">
        const dd = el.getAttribute('data-dropdown');
        if (dd && /^[A-Za-z][A-Za-z0-9_-]*$/.test(dd)) {
            return 'data-dropdown=' + dd;
        }
        // Custom dropdown item: <div class="dropdown-item" data-value="X" data-dropdown-id="Y">
        const dv = el.getAttribute('data-value');
        const ddi = el.getAttribute('data-dropdown-id');
        if (dv && ddi && /^[A-Za-z][A-Za-z0-9_-]*$/.test(ddi)) {
            return 'dropdown-item=' + ddi + '/' + dv;
        }
        // Snooze picker item.
        const ds = el.getAttribute('data-snooze');
        if (ds) {
            return 'data-snooze=' + ds;
        }
        const da = el.getAttribute('data-action');
        if (da) {
            // data-action is not unique on its own (multiple star buttons), so add an index hint.
            const sameDA = document.querySelectorAll(`[data-action="${da}"]`);
            if (sameDA.length === 1) return 'data-action=' + da;
            const idx = Array.from(sameDA).indexOf(el);
            return `data-action=${da}#${idx}`;
        }
        const dt = el.getAttribute('data-testid');
        if (dt) {
            const same = document.querySelectorAll(`[data-testid="${dt}"]`);
            if (same.length === 1) return 'data-testid=' + dt;
            const idx = Array.from(same).indexOf(el);
            return `data-testid=${dt}#${idx}`;
        }
        const role = el.getAttribute('role') || (
            el.tagName === 'BUTTON' ? 'button' :
            el.tagName === 'A' ? 'link' :
            el.tagName === 'INPUT' ? 'textbox' :
            null
        );
        const txt = visibleText(el);
        if (role && txt) {
            const re = new RegExp('^' + txt.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '$');
            const same = Array.from(document.querySelectorAll(`[role="${role}"], ${el.tagName.toLowerCase()}`))
                .filter(n => visibleText(n) === txt && isVisible(n));
            if (same.length === 1) return `role=${role}[name=${txt}]`;
            const idx = same.indexOf(el);
            return `role=${role}[name=${txt}]#${idx}`;
        }
        if (txt) {
            return 'text=' + txt;
        }
        // CSS path as last resort
        const path = [];
        let n = el;
        while (n && n.nodeType === 1 && n !== document.body) {
            const tag = n.tagName.toLowerCase();
            const parent = n.parentElement;
            const sibs = parent ? Array.from(parent.children).filter(s => s.tagName === n.tagName) : [n];
            const idx = sibs.indexOf(n) + 1;
            path.unshift(`${tag}:nth-of-type(${idx})`);
            n = parent;
        }
        return 'css=' + path.join(' > ');
    }

    // Selectors we consider interactive. WebArena-Infinity apps deliberately
    // avoid native <select>/<dialog>; instead they render custom widgets
    // (.custom-dropdown / .dropdown-item / [data-dropdown] etc.). We harvest
    // those too so MCTS can interact with theme pickers, snooze pickers,
    // category tabs, and label pickers.
    const SEL = [
        'a[href]:not([href=""])',
        'button',
        'input:not([type=hidden]):not([disabled])',
        'textarea:not([disabled])',
        '[role=button]',
        '[role=link]',
        '[role=tab]',
        '[role=menuitem]',
        '[role=option]',
        '[data-action]',
        '[data-dropdown]',
        '[data-dropdown-id]',
        '[data-value]',
        '[data-snooze]',
        '[data-category]',
        '[contenteditable=""]',
        '[contenteditable=true]',
        '.dropdown-item',
        '.dropdown-trigger',
        '.custom-dropdown',
    ];

    const seen = new Set();
    const all = document.querySelectorAll(SEL.join(','));
    for (const el of all) {
        if (seen.has(el)) continue;
        seen.add(el);
        if (!isVisible(el)) continue;
        // Skip svg children — only their interactive ancestors matter
        if (el.closest('svg') && el.tagName !== 'svg') continue;
        const kind = inferKind(el);
        const text = visibleText(el);
        const sel = selectorFor(el);
        const attrs = {
            'data-action': el.getAttribute('data-action') || '',
            'data-testid': el.getAttribute('data-testid') || '',
            'data-dropdown': el.getAttribute('data-dropdown') || '',
            'data-dropdown-id': el.getAttribute('data-dropdown-id') || '',
            'data-value': el.getAttribute('data-value') || '',
            'data-snooze': el.getAttribute('data-snooze') || '',
            'data-category': el.getAttribute('data-category') || '',
            'aria-label': el.getAttribute('aria-label') || '',
            'placeholder': el.getAttribute('placeholder') || '',
            'name': el.getAttribute('name') || '',
            'type': el.getAttribute('type') || '',
            'href': el.getAttribute('href') || '',
            'role': el.getAttribute('role') || '',
            'class': el.getAttribute('class') || '',
        };
        results.push({
            selector: sel,
            kind: kind,
            text: text,
            tag: el.tagName.toLowerCase(),
            attrs: attrs,
        });
    }
    return results;
}
"""


async def snapshot_candidates(page: Page) -> list[Candidate]:
    """Return the visible, interactive DOM elements at the current page state."""
    raw = await page.evaluate(_HARVEST_JS)
    out: list[Candidate] = []
    for r in raw:
        out.append(
            Candidate(
                selector=r["selector"],
                kind=r["kind"],
                text=r["text"],
                tag=r["tag"],
                attrs=r["attrs"],
            )
        )
    return out


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------


def _resolve(page: Page, selector: str):
    """Convert our selector mini-syntax to a Playwright Locator."""
    if selector.startswith("id="):
        return page.locator(f"#{selector[3:]}")
    if selector.startswith("data-dropdown="):
        # Custom dropdown trigger.
        return page.locator(f'[data-dropdown="{selector[len("data-dropdown="):]}"]')
    if selector.startswith("dropdown-item="):
        # "<dropdown-id>/<value>"
        rest = selector[len("dropdown-item="):]
        ddi, _, val = rest.partition("/")
        return page.locator(
            f'[data-dropdown-id="{ddi}"][data-value="{val}"]'
        )
    if selector.startswith("data-snooze="):
        return page.locator(f'[data-snooze="{selector[len("data-snooze="):]}"]')
    if selector.startswith("data-action="):
        rest = selector[len("data-action="):]
        if "#" in rest:
            value, idx = rest.rsplit("#", 1)
            return page.locator(f'[data-action="{value}"]').nth(int(idx))
        return page.locator(f'[data-action="{rest}"]')
    if selector.startswith("data-testid="):
        rest = selector[len("data-testid="):]
        if "#" in rest:
            value, idx = rest.rsplit("#", 1)
            return page.locator(f'[data-testid="{value}"]').nth(int(idx))
        return page.locator(f'[data-testid="{rest}"]')
    if selector.startswith("role="):
        # role=button[name=Compose]   or  role=button[name=X]#2
        body = selector[len("role="):]
        idx = 0
        if "#" in body:
            body, idx_str = body.rsplit("#", 1)
            idx = int(idx_str)
        # body looks like:  button[name=Compose]
        if "[name=" in body:
            role, rest = body.split("[name=", 1)
            name = rest.rstrip("]")
            loc = page.get_by_role(role, name=name, exact=True)
        else:
            loc = page.get_by_role(body)
        return loc.nth(idx)
    if selector.startswith("text="):
        return page.get_by_text(selector[len("text="):], exact=True).first
    if selector.startswith("css="):
        return page.locator(selector[len("css="):])
    # Fallback: treat as CSS selector
    return page.locator(selector)


async def exec_action(page: Page, action) -> None:
    """Execute a single Action. Raises on failure (caller logs + skips)."""
    locator = _resolve(page, action.selector)
    # Settling wait between actions: cap to 500ms — these are local apps.
    if action.kind == "click":
        await locator.click(timeout=2500)
    elif action.kind == "fill":
        # Fills with optional value; empty string clears.
        await locator.fill(action.value or "", timeout=2500)
    elif action.kind == "press":
        await locator.press(action.value or "Enter", timeout=2500)
    elif action.kind == "navigate":
        await page.goto(action.value or page.url)
    else:
        raise ValueError(f"Unknown action kind: {action.kind!r}")
    # Brief settle: state PUTs are fire-and-forget; give the network a beat.
    try:
        await page.wait_for_load_state("networkidle", timeout=600)
    except Exception:
        # Some PUTs never go idle (SSE keep-alives). 200ms hard wait is enough.
        await asyncio.sleep(0.2)
