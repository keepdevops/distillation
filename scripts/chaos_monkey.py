"""Chaos monkey UI stress tester for the Distillation Launcher Gradio app.

Randomly interacts with the UI: changes dropdowns, sliders, checkboxes,
radios, and text inputs — then reports errors, JS exceptions, and hangs.
Does NOT click Launch/Stop to avoid triggering real training runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Page, sync_playwright, Error as PlaywrightError

logger = logging.getLogger("chaos_monkey")

# ── Config ───────────────────────────────────────────────────────────────────

SAFE_SKIP_LABELS = {"Launch", "Stop"}   # buttons that trigger real side effects


@dataclass
class ChaosConfig:
    url: str = "http://127.0.0.1:7860"
    iterations: int = 50
    delay_min: float = 0.2
    delay_max: float = 1.0
    seed: int | None = None
    report_path: str = "chaos_report.json"
    headless: bool = False
    tab_pause: float = 0.5     # seconds after switching tab
    actions_per_tab: int = 5   # interactions to perform per tab visit


# ── Result tracking ──────────────────────────────────────────────────────────

@dataclass
class Incident:
    iteration: int
    kind: str          # "js_error", "page_crash", "timeout", "unexpected_close"
    message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RunReport:
    config: dict
    interactions: int = 0
    incidents: list[Incident] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def add_incident(self, it: int, kind: str, msg: str) -> None:
        self.incidents.append(Incident(it, kind, msg))
        logger.warning("[%d] %s: %s", it, kind, msg[:200])

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "interactions": self.interactions,
            "incidents": [vars(i) for i in self.incidents],
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "summary": {
                "total_incidents": len(self.incidents),
                "by_kind": _count_by(self.incidents, "kind"),
            },
        }


def _count_by(items: list, attr: str) -> dict:
    out: dict[str, int] = {}
    for item in items:
        out[getattr(item, attr)] = out.get(getattr(item, attr), 0) + 1
    return out


# ── Interaction helpers ───────────────────────────────────────────────────────

def _random_string(rng: random.Random) -> str:
    choices = [
        "",
        " ",
        "a" * 500,
        "\x00\x01\x02",
        "../../etc/passwd",
        "<script>alert(1)</script>",
        rng.choice(["hello", "foo", "bar", "test_model", "Qwen/Qwen2-0.5B"]),
        str(rng.randint(-999999, 999999)),
        "null",
        "None",
        "undefined",
    ]
    return rng.choice(choices)


# Gradio-specific selectors targeting only visible, interactable elements.
# Each entry: (selector, weight) — higher weight = picked more often.
_ELEMENT_SELECTORS: list[tuple[str, int]] = [
    ("input[type='range']",     4),   # sliders — many per tab, easy to fuzz
    ("input[type='number']",    3),   # numeric inputs paired with sliders
    ("input[type='radio']",     3),   # stage / backend / q-bits radios
    ("input[type='checkbox']",  3),   # checkboxes
    ("input[role='listbox']",   3),   # Gradio dropdown text inputs
    ("textarea",                2),   # text areas (Data Prep, Eval, Expert)
    ("button.reset-button",     2),   # ↺ reset-to-default buttons (safe)
    ("button.secondary",        1),   # Refresh buttons (safe)
]

# Buttons whose text exactly matches these are never clicked.
_SKIP_BUTTON_TEXT = {"Launch", "Stop", "Use via API", "Settings"}


def _visible(locator, count: int = 50) -> list:
    """Return all visible instances of a locator (up to count)."""
    out = []
    for el in locator.all()[:count]:
        try:
            if el.is_visible():
                out.append(el)
        except PlaywrightError:
            pass
    return out


def _weighted_pool(page: Page) -> list:
    """Build a pool of (element, selector_name) pairs from visible elements."""
    pool: list = []
    for sel, weight in _ELEMENT_SELECTORS:
        els = _visible(page.locator(sel))
        for el in els:
            pool.extend([(el, sel)] * weight)
    return pool


def _interact_slider(target, rng: random.Random) -> None:
    min_v = float(target.evaluate("el => el.min") or 0)
    max_v = float(target.evaluate("el => el.max") or 100)
    step  = float(target.evaluate("el => el.step") or 1)
    # Mix: random in range, boundary values, and out-of-range attempts
    choices = [
        rng.uniform(min_v, max_v),
        min_v, max_v,
        min_v - step,       # below min
        max_v + step,       # above max
        0, -1,
    ]
    val = rng.choice(choices)
    target.evaluate(
        f"el => {{ el.value = {val}; "
        f"el.dispatchEvent(new Event('input')); "
        f"el.dispatchEvent(new Event('change')); }}"
    )


def _interact_number(target, rng: random.Random) -> None:
    min_v = target.evaluate("el => el.min")
    max_v = target.evaluate("el => el.max")
    fuzz_vals = ["", "-1", "0", "9999999", "1e10", "NaN", "Infinity", "-Infinity",
                 str(rng.randint(-9999, 9999)), "1.234567890123456789"]
    if min_v:
        fuzz_vals += [str(float(min_v) - 1)]
    if max_v:
        fuzz_vals += [str(float(max_v) + 1)]
    target.click(timeout=2000)
    target.evaluate("el => { el.select ? el.select() : el.setSelectionRange(0, el.value.length); }")
    target.type(rng.choice(fuzz_vals), delay=5)
    target.press("Tab")


def _interact_dropdown(target, rng: random.Random, page: Page) -> None:
    """Click a Gradio dropdown input, optionally type a fuzz value."""
    target.click(timeout=2000)
    page.wait_for_timeout(300)
    # If options appeared, randomly pick one or type junk
    options = _visible(page.locator("li[role='option']"), 20)
    if options and rng.random() < 0.6:
        rng.choice(options).click(timeout=2000)
    else:
        target.evaluate("el => { el.select ? el.select() : el.setSelectionRange(0, el.value.length); }")
        target.type(_random_string(rng), delay=5)
        target.press("Escape")


def interact_with_tab(page: Page, tab_name: str, rng: random.Random,
                      report: RunReport, it: int, actions_per_tab: int) -> None:
    """Perform `actions_per_tab` random interactions on the currently visible tab."""
    pool = _weighted_pool(page)
    if not pool:
        return

    for _ in range(actions_per_tab):
        if not pool:
            break
        el, sel = rng.choice(pool)

        try:
            if not el.is_visible():
                continue

            label_text = (el.text_content() or "").strip()
            if label_text in _SKIP_BUTTON_TEXT:
                continue

            if "range" in sel:
                _interact_slider(el, rng)
            elif "number" in sel:
                _interact_number(el, rng)
            elif "radio" in sel or "checkbox" in sel:
                el.click(timeout=2000)
            elif "listbox" in sel:
                _interact_dropdown(el, rng, page)
            elif "textarea" in sel:
                el.click(timeout=2000)
                el.evaluate("e => e.setSelectionRange(0, e.value.length)")
                el.type(_random_string(rng), delay=5)
            else:
                # reset-button / secondary button
                el.click(timeout=2000)

            report.interactions += 1
            logger.debug("[%d] %s  tab=%r  label=%r", it, sel, tab_name, label_text[:40])

        except PlaywrightError as e:
            msg = str(e)
            if "Timeout" not in msg and "detached" not in msg:
                report.add_incident(it, "interaction_error", f"{sel} / {tab_name}: {msg[:200]}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_chaos(cfg: ChaosConfig) -> RunReport:
    rng = random.Random(cfg.seed)
    report = RunReport(config=vars(cfg))
    js_errors: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=cfg.headless)
        context = browser.new_context()
        page = context.new_page()

        # Capture JS console errors and page crashes
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))
        page.on("crash", lambda _: report.add_incident(-1, "page_crash", "page crashed"))

        logger.info("Navigating to %s …", cfg.url)
        try:
            page.goto(cfg.url, wait_until="networkidle", timeout=30_000)
        except PlaywrightError as e:
            logger.error("Could not reach %s: %s", cfg.url, e)
            report.add_incident(0, "navigation_error", str(e))
            browser.close()
            return report

        # Discover tab order once
        tab_elements = page.locator("button[role='tab']").all()
        tab_names = [(t, (t.text_content() or "").strip()) for t in tab_elements]
        logger.info("Found %d tabs: %s", len(tab_names),
                    [n for _, n in tab_names])
        logger.info("Running %d iterations (%d actions/tab) …",
                    cfg.iterations, cfg.actions_per_tab)
        t0 = time.monotonic()

        tab_cycle = list(range(len(tab_names)))
        tab_idx = 0

        for i in range(1, cfg.iterations + 1):
            # Drain accumulated JS errors
            while js_errors:
                report.add_incident(i, "js_error", js_errors.pop(0))

            # Cycle through tabs deterministically, with occasional random jump
            if rng.random() < 0.2 and tab_names:
                tab_idx = rng.randrange(len(tab_names))
            else:
                tab_idx = (tab_idx + 1) % len(tab_names) if tab_names else 0

            if tab_names:
                tab_el, tab_name = tab_names[tab_idx]
                try:
                    tab_el.click(timeout=3000)
                    page.wait_for_timeout(int(cfg.tab_pause * 1000))
                except PlaywrightError as e:
                    report.add_incident(i, "tab_click_error", str(e))
                    tab_name = "unknown"
            else:
                tab_name = "unknown"

            interact_with_tab(page, tab_name, rng, report, i, cfg.actions_per_tab)

            delay = rng.uniform(cfg.delay_min, cfg.delay_max)
            time.sleep(delay)

            if i % 10 == 0:
                logger.info("  iteration %d/%d  interactions=%d  incidents=%d",
                            i, cfg.iterations, report.interactions, len(report.incidents))

        report.elapsed_seconds = time.monotonic() - t0
        browser.close()

    return report


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chaos monkey for the Distillation Launcher UI")
    p.add_argument("--url", default="http://127.0.0.1:7860", help="Gradio app URL")
    p.add_argument("--iterations", type=int, default=50, help="Number of random interactions")
    p.add_argument("--delay-min", type=float, default=0.2, help="Min delay between actions (s)")
    p.add_argument("--delay-max", type=float, default=1.0, help="Max delay between actions (s)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--report", default="chaos_report.json", help="Output report path")
    p.add_argument("--headless", action="store_true", help="Run browser headlessly")
    p.add_argument("--tab-pause", type=float, default=0.5, help="Seconds to wait after tab switch")
    p.add_argument("--actions-per-tab", type=int, default=5, help="Interactions per tab visit")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = ChaosConfig(
        url=args.url,
        iterations=args.iterations,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        seed=args.seed,
        report_path=args.report,
        headless=args.headless,
        tab_pause=args.tab_pause,
        actions_per_tab=args.actions_per_tab,
    )

    report = run_chaos(cfg)
    report_dict = report.to_dict()

    out = Path(cfg.report_path)
    out.write_text(json.dumps(report_dict, indent=2))
    logger.info("Report written to %s", out)

    # Print summary
    s = report_dict["summary"]
    print(f"\n{'='*50}")
    print(f"  Iterations:   {cfg.iterations}")
    print(f"  Interactions: {report.interactions}")
    print(f"  Incidents:    {s['total_incidents']}")
    if s["by_kind"]:
        for kind, count in s["by_kind"].items():
            print(f"    {kind}: {count}")
    print(f"  Elapsed:      {report.elapsed_seconds:.1f}s")
    print(f"{'='*50}\n")

    sys.exit(1 if report.incidents else 0)


if __name__ == "__main__":
    main()
