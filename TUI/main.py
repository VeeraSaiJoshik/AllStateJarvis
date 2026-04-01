#!/usr/bin/env python3
"""
DSA Vault — Competition Template Search + Advisor
Run: python main.py
"""
import json
import re
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import (
    Input, ListView, ListItem, Static, Label,
    Header, TabbedContent, TabPane, TextArea,
)
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.binding import Binding
from textual import on, work
from textual.reactive import reactive

from engine import SearchEngine

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3"


CATEGORY_ICONS = {
    "graphs": "◈",
    "trees": "⬡",
    "dp": "◇",
    "strings": "◉",
    "math_nt": "∑",
    "data_structures": "▣",
}

CATEGORY_NAMES = {
    "graphs": "GRAPHS",
    "trees": "TREES",
    "dp": "DYNAMIC PROGRAMMING",
    "strings": "STRINGS",
    "math_nt": "MATH / NUMBER THEORY",
    "data_structures": "DATA STRUCTURES",
}

CSS = """
Screen {
    background: #0d1117;
}

Header {
    background: #161b22;
    color: #58a6ff;
    text-style: bold;
    height: 1;
}

/* ── Tabs ── */
TabbedContent {
    height: 1fr;
    margin: 0 1;
}

TabPane {
    padding: 0;
}

/* ── Search tab ── */
#search-input {
    margin: 1 0 1 0;
    background: #21262d;
    border: tall #30363d;
    color: #e6edf3;
    height: 3;
}

#search-input:focus {
    border: tall #58a6ff;
}

#search-main {
    height: 1fr;
}

#results-panel {
    width: 36%;
    border: solid #30363d;
}

#results-header {
    background: #161b22;
    color: #8b949e;
    padding: 0 1;
    height: 1;
    text-style: bold;
}

#results-list {
    height: 1fr;
    background: #0d1117;
}

ListView > ListItem {
    padding: 0 1;
    height: 1;
    background: #0d1117;
    color: #c9d1d9;
}

ListView > ListItem:hover {
    background: #21262d;
}

ListView > ListItem.--highlight {
    background: #1c2d3e;
    color: #79c0ff;
    text-style: bold;
}

#preview-panel {
    width: 1fr;
    border: solid #30363d;
    margin-left: 1;
}

#preview-header {
    background: #161b22;
    color: #8b949e;
    padding: 0 1;
    height: 1;
    text-style: bold;
}

#preview-scroll {
    height: 1fr;
    background: #0d1117;
}

#preview-content {
    padding: 1 2;
    color: #e6edf3;
}

/* ── Advisor tab ── */
#advisor-main {
    height: 1fr;
}

#problem-panel {
    height: 40%;
    border: solid #30363d;
    margin-bottom: 1;
}

#problem-header {
    background: #161b22;
    color: #8b949e;
    padding: 0 1;
    height: 1;
    text-style: bold;
}

#problem-input {
    height: 1fr;
    background: #0d1117;
    color: #e6edf3;
    border: none;
}

#problem-input:focus {
    border: none;
}

#analysis-panel {
    height: 1fr;
    border: solid #30363d;
}

#analysis-header {
    background: #161b22;
    color: #8b949e;
    padding: 0 1;
    height: 1;
    text-style: bold;
}

#analysis-scroll {
    height: 1fr;
    background: #0d1117;
}

#analysis-content {
    padding: 1 2;
    color: #e6edf3;
}

/* ── Coach log tab ── */
#coach-log-scroll {
    height: 1fr;
    background: #0d1117;
}

#coach-log {
    padding: 1 2;
    color: #8b949e;
}

/* ── Status bar ── */
#status {
    height: 1;
    background: #161b22;
    color: #8b949e;
    padding: 0 1;
    margin: 0 1;
}
"""


class DSAVault(App):
    """DSA Vault — Competition Template Search + AI Advisor"""

    CSS = CSS

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+y", "copy_code", "Copy Code", show=True),
        Binding("ctrl+g", "generate_solution", "Generate", show=True),
        Binding("ctrl+l", "show_log", "Log", show=False),
        Binding("ctrl+r", "reload", "Reload", show=False),
        Binding("j", "next_result", "↓", show=False),
        Binding("k", "prev_result", "↑", show=False),
        Binding("down", "next_result", "↓", show=False),
        Binding("up", "prev_result", "↑", show=False),
        Binding("ctrl+enter", "analyze", "Analyze Problem", show=True),
    ]

    selected: reactive[dict | None] = reactive(None)

    def __init__(self):
        super().__init__()
        self.engine = SearchEngine()
        self._results: list[dict] = []
        self._coach_log: list[str] = []

    # ── Layout ────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with TabbedContent(initial="search"):

            # ── Tab 1: Search ──────────────────────────────────────────────
            with TabPane("Search  /", id="search"):
                yield TextArea(
                    "",
                    placeholder="  Search: DFS, binary search, knapsack, KMP, segment tree...",
                    id="search-input",
                    soft_wrap=True,
                )
                with Horizontal(id="search-main"):
                    with Vertical(id="results-panel"):
                        yield Label(" TEMPLATES", id="results-header", markup=False)
                        yield ListView(id="results-list")
                    with Vertical(id="preview-panel"):
                        yield Label(" CODE PREVIEW  (ctrl+y to copy)", id="preview-header", markup=False)
                        with ScrollableContainer(id="preview-scroll"):
                            yield Static("", id="preview-content", markup=False)

            # ── Tab 2: Advisor ─────────────────────────────────────────────
            with TabPane("Advisor  ·", id="advisor"):
                with Vertical(id="advisor-main"):
                    with Vertical(id="problem-panel"):
                        yield Label(
                            " PASTE PROBLEM HERE  (ctrl+enter to analyze)",
                            id="problem-header",
                            markup=False,
                        )
                        yield TextArea(
                            "",
                            id="problem-input",
                            language=None,
                            show_line_numbers=False,
                            soft_wrap=True,
                        )
                    with Vertical(id="analysis-panel"):
                        yield Label(" RECOMMENDATIONS", id="analysis-header", markup=False)
                        with ScrollableContainer(id="analysis-scroll"):
                            yield Static(
                                "Paste a problem above and press ctrl+enter to get algorithm recommendations.",
                                id="analysis-content",
                                markup=False,
                            )

            # ── Tab 3: Coach log (hidden until ctrl+l) ─────────────────────
            with TabPane("·", id="coach-log-tab"):
                with ScrollableContainer(id="coach-log-scroll"):
                    yield Static(
                        "No generation runs yet. Ctrl+G from the search bar to start.",
                        id="coach-log",
                        markup=False,
                    )

        yield Static(
            " ctrl+y:Copy  ctrl+g:Generate  ctrl+enter:Analyze  j/k:Navigate  ctrl+r:Reload  ctrl+q:Quit",
            id="status",
            markup=False,
        )

    def on_mount(self) -> None:
        self.title = "DSA Vault"
        count = len(self.engine.solutions)
        self.sub_title = f"{count} templates"
        self._refresh_list(self.engine.get_all())
        self.query_one("#search-input").focus()

    # ── Search tab ─────────────────────────────────────────────────────────

    def _refresh_list(self, solutions: list[dict]) -> None:
        self._results = solutions
        lv = self.query_one("#results-list", ListView)
        lv.clear()
        for sol in solutions:
            icon = CATEGORY_ICONS.get(sol["category"], "•")
            item = ListItem(Label(f" {icon} {sol['name']}", markup=False))
            item._solution = sol  # type: ignore[attr-defined]
            lv.append(item)
        if solutions:
            lv.index = 0
            self._show_preview(solutions[0])

    @on(TextArea.Changed, "#search-input")
    def on_search_changed(self, event: TextArea.Changed) -> None:
        q = self.query_one("#search-input", TextArea).text.strip()
        results = self.engine.search(q) if q else self.engine.get_all()
        self._refresh_list(results)

    @on(ListView.Highlighted)
    def on_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item and hasattr(event.item, "_solution"):
            self._show_preview(event.item._solution)  # type: ignore[attr-defined]

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        if event.item and hasattr(event.item, "_solution"):
            self._show_preview(event.item._solution)  # type: ignore[attr-defined]

    def _show_preview(self, sol: dict) -> None:
        self.selected = sol
        icon = CATEGORY_ICONS.get(sol["category"], "•")
        self.query_one("#preview-header", Label).update(
            f" {icon} {sol['name']}  ({sol['category']})  ctrl+y to copy"
        )
        lines = [
            f"# {sol['name']}",
            f"# Tags      : {', '.join(sol['tags'])}",
        ]
        if sol.get("complexity"):
            lines.append(f"# Complexity: {sol['complexity']}")
        if sol.get("description"):
            lines.append(f"# Info      : {sol['description']}")
        lines += ["", "─" * 60, "", sol.get("code", "# No code found")]
        self.query_one("#preview-content", Static).update("\n".join(lines))

    def action_next_result(self) -> None:
        lv = self.query_one("#results-list", ListView)
        if lv.index is not None and lv.index < len(self._results) - 1:
            lv.index += 1

    def action_prev_result(self) -> None:
        lv = self.query_one("#results-list", ListView)
        if lv.index is not None and lv.index > 0:
            lv.index -= 1

    # ── Advisor tab ────────────────────────────────────────────────────────

    def action_analyze(self) -> None:
        problem = self.query_one("#problem-input", TextArea).text.strip()
        if not problem:
            self._set_analysis("No problem text found. Paste a problem first.")
            return
        self._run_advisor(problem)

    def _run_advisor(self, problem: str) -> None:
        self._set_status("Analyzing...")

        # Use the new classify() method
        result = self.engine.classify(problem, top_k=10)

        # Extract results
        primary_category = result["primary_category"]
        confidence = result["confidence"]
        detected_patterns = result["detected_patterns"]
        reranked = result["ranked_solutions"]

        # Build output
        lines: list[str] = []

        # Show primary category prominently
        if primary_category != "unknown":
            icon = CATEGORY_ICONS.get(primary_category, "•")
            category_name = CATEGORY_NAMES.get(primary_category, primary_category.upper())
            conf_bar = "█" * int(confidence / 10)
            lines.append(f"PRIMARY CATEGORY: {icon} {category_name}")
            lines.append(f"Confidence: {conf_bar} {confidence}%")
            lines.append("")

        # Show detected signals
        if detected_patterns:
            lines.append("DETECTED PATTERNS:")
            for label, pattern_conf in detected_patterns[:8]:  # Show top 8 patterns
                conf_indicator = "●" if pattern_conf >= 75 else "◐" if pattern_conf >= 60 else "○"
                lines.append(f"  {conf_indicator} {label}")
            lines.append("")

        # Show recommended templates with confidence indicators
        lines.append("RECOMMENDED TEMPLATES:")
        lines.append("")
        for i, sol in enumerate(reranked, 1):
            icon = CATEGORY_ICONS.get(sol["category"], "•")
            lines.append(f"  {i}. {icon} {sol['name']}")
            lines.append(f"     Category  : {sol['category']}")
            if sol.get("complexity"):
                lines.append(f"     Complexity: {sol['complexity']}")
            if sol.get("description"):
                desc = sol["description"].split(".")[0].strip()
                lines.append(f"     Notes     : {desc}.")
            lines.append("")

        # Hint about template selection
        if reranked:
            lines.append("─" * 50)
            lines.append("Tip: Switch to the Search tab to view full template code.")

        if not detected_patterns:
            lines.append("─" * 50)
            lines.append("No strong keyword signals found.")
            lines.append("Results are ranked by BM25 text similarity only.")
            lines.append("Try pasting more of the problem statement.")

        self._set_analysis("\n".join(lines))
        self._set_status("Done — switch to Search tab to view template code")

    def _set_analysis(self, text: str) -> None:
        self.query_one("#analysis-content", Static).update(text)

    # ── Coach ─────────────────────────────────────────────────────────────

    def action_generate_solution(self) -> None:
        problem = self.query_one("#search-input", TextArea).text.strip()
        if not problem:
            self._set_status("Type a problem/topic in the search bar first, then Ctrl+G")
            self.set_timer(3, self._reset_status)
            return
        self._append_log(f"─── New generation ───────────────────────────────")
        self._append_log(f"Input: {problem}")
        self._set_status("Coach: calling ollama...")
        self._run_coach(problem)

    def action_show_log(self) -> None:
        tc = self.query_one(TabbedContent)
        tc.active = "coach-log-tab"

    @work(thread=True)
    def _run_coach(self, problem: str) -> None:
        prompt = (
            "You are an expert competitive programmer. "
            "Generate a complete, contest-ready Python implementation for the following problem or algorithm topic.\n\n"
            f"Topic: {problem}\n\n"
            "Output ONLY a single Python file with this exact structure — no explanation, no markdown fences:\n\n"
            '"""\n'
            "NAME: <concise algorithm name>\n"
            "TAGS: <comma-separated lowercase tags, e.g. graph, bfs, shortest-path>\n"
            "DESCRIPTION: <one sentence: when and why to use this>\n"
            "COMPLEXITY: Time: O(...), Space: O(...)\n"
            'CODE:\n"""\n\n'
            "# full Python implementation here, with comments\n\n"
            "Rules:\n"
            "- The triple-quoted docstring must be at the very top.\n"
            "- CODE: must be the last field before the closing triple-quote.\n"
            "- After the closing triple-quote write the real implementation.\n"
            "- Include at least one complete, runnable example in comments."
        )

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }).encode()

        try:
            self.app.call_from_thread(self._append_log, f"POST {OLLAMA_URL} (model={OLLAMA_MODEL})")
            req = urllib.request.Request(
                OLLAMA_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read())
            code = result.get("response", "").strip()
        except urllib.error.URLError as e:
            self.app.call_from_thread(self._append_log, f"ERROR: Cannot reach ollama — {e.reason}")
            self.app.call_from_thread(self._set_status, "Coach: ollama unreachable — check log (ctrl+l)")
            return
        except Exception as e:
            self.app.call_from_thread(self._append_log, f"ERROR: {e}")
            self.app.call_from_thread(self._set_status, "Coach: failed — check log (ctrl+l)")
            return

        if not code:
            self.app.call_from_thread(self._append_log, "ERROR: Empty response from ollama")
            self.app.call_from_thread(self._set_status, "Coach: empty response")
            return

        self.app.call_from_thread(self._append_log, f"Response received ({len(code)} chars)")

        # Strip accidental markdown fences
        code = re.sub(r"^```(?:python)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)

        gen_dir = Path("solutions/generated")
        gen_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", problem.lower())[:40].strip("_")
        filepath = gen_dir / f"{slug}.py"
        filepath.write_text(code)

        self.app.call_from_thread(self._append_log, f"Saved → solutions/generated/{slug}.py")
        self.app.call_from_thread(self._finish_generation, slug)

    def _finish_generation(self, slug: str) -> None:
        self.engine.reload()
        q = self.query_one("#search-input", Input).value.strip()
        results = self.engine.search(q) if q else self.engine.get_all()
        self._refresh_list(results)
        count = len(self.engine.solutions)
        self.sub_title = f"{count} templates"
        self._append_log(f"Catalog reloaded — {count} templates total.")
        self._set_status(f"Coach: '{slug}' added to catalog  (ctrl+l to view log)")
        self.set_timer(4, self._reset_status)

    def _append_log(self, msg: str) -> None:
        self._coach_log.append(msg)
        try:
            log_widget = self.query_one("#coach-log", Static)
            log_widget.update("\n".join(self._coach_log[-80:]))
            scroll = self.query_one("#coach-log-scroll", ScrollableContainer)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    # ── Copy ──────────────────────────────────────────────────────────────

    def action_copy_code(self) -> None:
        if not self.selected:
            self._set_status("Nothing selected to copy")
            self.set_timer(2, self._reset_status)
            return
        code = self.selected.get("code", "")
        copied = False
        try:
            import pyperclip
            pyperclip.copy(code)
            copied = True
        except Exception:
            pass
        if not copied:
            try:
                proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                proc.communicate(code.encode())
                copied = True
            except Exception:
                pass
        name = self.selected["name"]
        self._set_status(f"Copied '{name}' to clipboard!" if copied else "Copy failed — install pyperclip")
        self.set_timer(2.5, self._reset_status)

    # ── Reload ────────────────────────────────────────────────────────────

    def action_reload(self) -> None:
        self.engine.reload()
        q = self.query_one("#search-input", TextArea).text.strip()
        results = self.engine.search(q) if q else self.engine.get_all()
        self._refresh_list(results)
        count = len(self.engine.solutions)
        self.sub_title = f"{count} templates"
        self._set_status(f"Reloaded — {count} templates")
        self.set_timer(2, self._reset_status)

    # ── Status helpers ────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self.query_one("#status", Static).update(f" {msg}")

    def _reset_status(self) -> None:
        self.query_one("#status", Static).update(
            " ctrl+y:Copy  ctrl+enter:Analyze  j/k:Navigate  ctrl+r:Reload  ctrl+q:Quit"
        )


if __name__ == "__main__":
    app = DSAVault()
    app.run()
