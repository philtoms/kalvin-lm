"""KScript TUI Application - Interactive development environment for KScript and Agent."""

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.css.query import NoMatches
from textual.widgets import Footer, Header, ListView

from kalvin.agent import Agent
from kalvin.abstract import KLine
from kalvin.events import RationaliseEvent
from kscript import KScript, CompiledEntry

# Type alias for entry identity keys used in tracking sets
EntryKey = tuple[int, tuple[int, ...]]
from kscript.decompiler import Decompiler
from ui.kscript.dialogs import LoadScriptDialog, SaveStateDialog, LoadStateDialog
from ui.kscript.regions import EditorRegion, ResponsesRegion, ToolbarRegion
from ui.kscript.regions.responses import ResponseItem
from ui.kscript.regions.toolbar import ExecutionState

logger = logging.getLogger(__name__)

DEFAULT_SCRIPTS_DIR = Path("data/scripts")

# State file paths for dev mode (harness-driven save/restore)
_STATE_DIR = Path(__file__).parent.parent.parent  # project root
AGENT_STATE_FILE = _STATE_DIR / ".tmp.bin"
UI_STATE_FILE = _STATE_DIR / ".tmp.json"


class KScriptApp(App):
    """KScript TUI Application for interactive KScript development."""

    CSS = """
    Screen {
        layout: vertical;
    }

    .main-container {
        layout: vertical;
        height: 1fr;
        padding: 0;
    }

    .content-container {
        layout: horizontal;
        height: 1fr;
    }
    """

    TITLE = "KScript TUI"
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+r", "run_script", "Run/Stop"),
        ("ctrl+s", "step_script", "Step"),
        ("ctrl+o", "load_script", "Load.ks"),
        ("ctrl+shift+s", "save_state", "Save.k"),
        ("ctrl+shift+o", "load_state", "Load.k"),
        ("ctrl+l", "clear_responses", "Clear"),
    ]

    def __init__(self, dev_mode: bool = True, auto_compile_interval: float = 1.0) -> None:
        super().__init__()
        self._dev_mode = dev_mode
        self._agent: Optional[Agent] = None
        self._decompiler: Decompiler = Decompiler()
        self._execution_state: ExecutionState = ExecutionState.IDLE
        self._cancelled: bool = False
        self._last_script_dir: Path = DEFAULT_SCRIPTS_DIR
        self._last_state_dir: Path = Path("data")
        self._auto_compile_interval: float = auto_compile_interval
        # Tracking state for test harness (KB-008)
        self._submitted: set[EntryKey] = set()
        self._satisfied: set[EntryKey] = set()
        # Selection tracking for ratification (KB-012)
        self._selected_proposal: KLine | None = None
        self._selected_entry_key: EntryKey | None = None
        # Event correlation state (KB-010)
        self._compiled_entries: list[CompiledEntry] = []
        self._expectations: dict[EntryKey, list[KLine]] = {}
        self._fast_path_results: dict[EntryKey, bool] = {}
        # Run mode flag — distinguishes Run from Step in event callback
        self._run_mode_active: bool = False

    def on_mount(self) -> None:
        """Initialize Agent instance on app start."""
        DEFAULT_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)

        if self._dev_mode:
            self._restore_state()
            signal.signal(signal.SIGTERM, self._on_sigterm)
        else:
            self._agent = Agent()
            self._setup_events()

    def _setup_events(self) -> None:
        """Subscribe to Agent rationalisation events.

        Implements event correlation (HRN-11), fast-path auto-satisfaction
        (HRN-3), slow-path proposal display (HRN-4, HRN-17), and
        multiple-proposal tracking (HRN-18).

        Fast-path vs slow-path is determined from the event itself:
        fast-path events always have query == proposal (the Agent passes
        the same kline as both). This avoids a timing dependency on
        _fast_path_results being stored before the callback fires.
        """
        if not self._agent:
            return

        app = self

        def on_event(event: RationaliseEvent) -> None:
            # Ignore idle sentinel from cogitator
            if event.kind == "done":
                return

            # Only handle ground/frame rationalisation events
            if event.kind not in ("ground", "frame"):
                app.log(f"Ignoring unknown event kind: {event.kind}")
                return

            query = event.query
            proposal = event.proposal
            significance = event.significance

            # Determine fast-path: query == proposal structurally
            is_fast_path = app._structural_match(query, proposal)

            # Correlate event to a compiled entry (HRN-11)
            matched_key: Optional[EntryKey] = None
            for entry in app._compiled_entries:
                if app._structural_match(query, entry):
                    matched_key = app._entry_key(entry)
                    break

            if is_fast_path and matched_key is not None:
                # Fast-path auto-satisfaction (HRN-3)
                app._satisfied.add(matched_key)
                decompiled = app._decompile_response([proposal])
                for level, source in decompiled:
                    app._add_event_response(
                        level, source, "pass", significance,
                        kline=proposal, entry_key=matched_key,
                    )

            elif matched_key is not None:
                # Slow-path proposal for a known expectation (HRN-4, HRN-17)
                if matched_key not in app._expectations:
                    app._expectations[matched_key] = []
                app._expectations[matched_key].append(proposal)

                decompiled = app._decompile_response([proposal])
                for level, source in decompiled:
                    app._add_event_response(
                        level, source, "pending", significance,
                        kline=proposal, entry_key=matched_key,
                    )

            else:
                # Unmatched event — no compiled entry correlates (HRN-17)
                # Display as pending for human review, execution continues
                app.log(f"Unmatched event: kind={event.kind} sig={significance:#x}")
                decompiled = app._decompile_response([proposal])
                for level, source in decompiled:
                    app._add_event_response(
                        level, source, "pending", significance,
                        kline=proposal, entry_key=None,
                    )

        self._agent.events.subscribe(on_event)

    def _add_event_response(
        self,
        level: str,
        decompiled_source: str,
        status: str,
        significance: int,
        kline: KLine | None = None,
        entry_key: EntryKey | None = None,
    ) -> None:
        """Add a response item from an event, with status and significance.

        Centralises the call to ResponsesRegion.add_response so the
        event callback stays clean.
        """
        responses = self.query_one(ResponsesRegion)
        responses.add_response(
            level=level,
            decompiled_source=decompiled_source,
            status=status,
            significance=significance,
            kline=kline,
            entry_key=entry_key,
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(classes="main-container"):
            yield ToolbarRegion()
            with Horizontal(classes="content-container"):
                yield EditorRegion()
                yield ResponsesRegion()
        yield Footer()

    # === Toolbar Event Handlers ===

    def on_toolbar_region_load_script(self, event: ToolbarRegion.LoadScript) -> None:
        """Handle Load.ks button."""
        self.action_load_script()

    def on_toolbar_region_save_state(self, event: ToolbarRegion.SaveState) -> None:
        """Handle Save.k button."""
        self.action_save_state()

    def on_toolbar_region_load_state(self, event: ToolbarRegion.LoadState) -> None:
        """Handle Load.k button."""
        self.action_load_state()

    def on_toolbar_region_run(self, event: ToolbarRegion.Run) -> None:
        """Handle Run/Stop toggle button."""
        self.action_run_script()

    def on_toolbar_region_step(self, event: ToolbarRegion.Step) -> None:
        """Handle Step button."""
        self.action_step_script()

    def on_toolbar_region_clear(self, event: ToolbarRegion.Clear) -> None:
        """Handle Clear button."""
        self.action_clear_responses()

    def on_toolbar_region_ratify(self, event: ToolbarRegion.Ratify) -> None:
        """Handle Ratify button — countersign the selected proposal.

        Implements HRN-9: calls agent.countersign(selected_proposal) with
        the proposal as-is. On success, adds the entry key to _satisfied.
        """
        if self._selected_proposal is None or self._agent is None:
            return

        result = self._agent.countersign(self._selected_proposal)

        if result and self._selected_entry_key is not None:
            self._satisfied.add(self._selected_entry_key)
            # Update the selected response item's status to "pass"
            self._update_selected_response_status("pass")

        # Clear selection state
        self._selected_proposal = None
        self._selected_entry_key = None

        # Disable Ratify button
        toolbar = self.query_one(ToolbarRegion)
        toolbar.set_ratify_enabled(False)

    def _update_selected_response_status(self, status: str) -> None:
        """Update the status of the currently selected response item.

        Finds the selected ResponseItem in the ResponsesRegion's ListView
        and updates its status field, then refreshes the display.
        """
        try:
            responses = self.query_one(ResponsesRegion)
            list_view = responses.query_one("#responses-list", ListView)
            if list_view.index is not None and list_view.index < len(list_view.children):
                item = list_view.children[list_view.index]
                if isinstance(item, ResponseItem):
                    item.status = status
                    item.refresh()
        except (NoMatches, AttributeError):
            pass  # Best-effort UI update — widget may not be mounted

    # === Responses Event Handlers ===

    def on_responses_region_response_clicked(self, event: ResponsesRegion.ResponseClicked) -> None:
        """Handle click on a response item — track selection for ratification.

        Stores the selected proposal KLine and entry key, and enables the
        Ratify button. The old 'append to editor' behavior is removed:
        Step mode is for inspection and ratification, not editing.
        """
        # Store selection state for ratification
        self._selected_proposal = event.kline
        self._selected_entry_key = event.entry_key

        # Enable Ratify button
        toolbar = self.query_one(ToolbarRegion)
        toolbar.set_ratify_enabled(True)

    # === State Management ===

    def _set_state(self, state: ExecutionState) -> None:
        """Update execution state and toolbar."""
        self._execution_state = state
        toolbar = self.query_one(ToolbarRegion)
        toolbar.set_state(state)
        self._update_toolbar_progress()

    def _update_toolbar_progress(self) -> None:
        """Push satisfaction progress counts to the toolbar."""
        toolbar = self.query_one(ToolbarRegion)
        submitted = getattr(self, "_submitted", set())
        satisfied = getattr(self, "_satisfied", set())
        total = len(submitted)
        sat = len(satisfied)
        pending = total - sat
        toolbar.set_progress(sat, total, pending)

    # === Dev Mode State (SIGTERM save, mount restore) ===

    def _on_sigterm(self, signum, frame) -> None:
        """Handle SIGTERM from harness: save state and exit."""
        self._save_state()
        sys.exit(0)

    def _save_state(self) -> None:
        """Save Agent and UI state to temp files."""
        if self._agent:
            self._agent.save(AGENT_STATE_FILE)

        editor = self.query_one(EditorRegion)
        ui_state = {
            "editor_content": editor.get_script(),
            "last_script_dir": str(self._last_script_dir),
            "last_state_dir": str(self._last_state_dir),
            "execution_state": self._execution_state.name,
            "submitted": [[sig, list(nodes)] for sig, nodes in self._submitted],
            "satisfied": [[sig, list(nodes)] for sig, nodes in self._satisfied],
        }
        with open(UI_STATE_FILE, "w") as f:
            json.dump(ui_state, f, indent=2)

    def _restore_state(self) -> None:
        """Restore Agent and UI state from temp files."""
        # Restore Agent
        if AGENT_STATE_FILE.exists():
            try:
                self._agent = Agent.load(AGENT_STATE_FILE)
                self.log("Restored Agent state")
            except Exception as e:
                self.log("Failed to load agent state: {e}")
                self._agent = Agent()
        else:
            self._agent = Agent()

        self._setup_events()

        # Restore UI state
        if UI_STATE_FILE.exists():
            try:
                with open(UI_STATE_FILE) as f:
                    ui_state = json.load(f)
                editor = self.query_one(EditorRegion)
                if "editor_content" in ui_state:
                    editor.set_script(ui_state["editor_content"])
                if "last_script_dir" in ui_state:
                    self._last_script_dir = Path(ui_state["last_script_dir"])
                if "last_state_dir" in ui_state:
                    self._last_state_dir = Path(ui_state["last_state_dir"])
                if "execution_state" in ui_state:
                    self._execution_state = ExecutionState[ui_state["execution_state"]]
                if "submitted" in ui_state:
                    self._submitted = {(pair[0], tuple(pair[1])) for pair in ui_state["submitted"]}
                if "satisfied" in ui_state:
                    self._satisfied = {(pair[0], tuple(pair[1])) for pair in ui_state["satisfied"]}
                self.log("Restored UI state")
            except (json.JSONDecodeError, Exception) as e:
                self.log(f"Failed to load UI state: {e}")

        # Clean up temp files
        AGENT_STATE_FILE.unlink(missing_ok=True)
        UI_STATE_FILE.unlink(missing_ok=True)

    # === Script Compilation ===

    def _compile_script(self) -> Optional[list[CompiledEntry]]:
        """Compile the current editor content to KLine entries.

        On success, stores the result in self._compiled_entries for event
        correlation. On error, displays a ✗ response item (HRN-14) and
        returns None.

        Returns:
            List of CompiledEntry objects, or None on error.
        """
        editor = self.query_one(EditorRegion)
        script = editor.get_script()

        if not script.strip():
            return None

        try:
            agent = KScript(script, dev=self._dev_mode)
            self._compiled_entries = agent.entries
            return agent.entries
        except Exception as e:
            self.log(f"Compilation error: {e}")
            self._show_compilation_error(str(e))
            return None

    def _entry_to_kline(self, entry: CompiledEntry) -> KLine:
        """Convert a CompiledEntry to a KLine for Agent.

        Since CompiledEntry extends KLine, this is a simple cast.

        Args:
            entry: The compiled entry to convert.

        Returns:
            A KLine suitable for rationalise().
        """
        return entry

    def _decompile_response(self, klines: list[KLine]) -> list[tuple[str, str]]:
        """Decompile a list of KLines to KScript source.

        Args:
            klines: List of KLines from rationalise response.

        Returns:
            Decompile KScript source string.
        """
        if not klines:
            return []
        entries = self._decompiler.decompile(klines)
        return [(e.level, e.to_kscript())  for e in entries]

    def _show_compilation_error(self, error_message: str) -> None:
        """Display a compilation error as a ✗ response item in the panel.

        HRN-14: Compilation errors are surfaced as mismatch response items
        with status='mismatch' and significance=0.

        Args:
            error_message: The error message to display.
        """
        responses = self.query_one(ResponsesRegion)
        responses.add_response(
            level="S4",
            decompiled_source=error_message,
            status="mismatch",
            significance=0,
            kline=None,
            entry_key=None,
        )

    # === Tracking State Helpers (KB-008) ===

    @staticmethod
    def _entry_key(entry: CompiledEntry) -> EntryKey:
        """Return a hashable identity key for a compiled entry.

        The key is (signature, tuple(nodes)), suitable for set membership tests.
        """
        return (entry.signature, tuple(entry.nodes))

    def _get_pending(self, entries: list[CompiledEntry]) -> list[CompiledEntry]:
        """Return entries not yet in _submitted, preserving input order."""
        return [e for e in entries if self._entry_key(e) not in self._submitted]

    # === Structural Match Helper (KB-010) ===

    @staticmethod
    def _structural_match(a: KLine, b: KLine) -> bool:
        """Check structural equality of two KLines by signature and nodes.

        This is the canonical match function referenced in specs/harness.md
        §Event Correlation (HRN-11) and §Satisfaction (HRN-4).

        Args:
            a: First KLine to compare.
            b: Second KLine to compare.

        Returns:
            True if both signature and nodes match exactly.
        """
        return a.signature == b.signature and a.nodes == b.nodes

    # === Auto-Countersign (Run Mode) ===

    def _auto_countersign(self, entry: CompiledEntry, proposal: KLine) -> bool:
        """Auto-countersign a proposal if it structurally matches the entry.

        Called from the event callback when a proposal arrives during Run
        mode.  Checks structural match (HRN-4) and, on match, calls
        agent.countersign() and marks the entry as satisfied.

        Args:
            entry: The compiled entry that the proposal is checked against.
            proposal: The proposal kline from the rationalisation event.

        Returns:
            True if the proposal matched and was countersigned.
            False if no match — entry stays pending for human review (HRN-17).
        """
        if self._structural_match(entry, proposal):
            self._agent.countersign(proposal)
            self._satisfied.add(self._entry_key(entry))
            return True
        return False

    # === Actions ===

    def action_load_script(self) -> None:
        """Open dialog to load a .ks script file."""
        self.push_screen(
            LoadScriptDialog(
                title="Load KScript",
                initial_path=str(self._last_script_dir),
            ),
            self._handle_load_script,
        )

    def _handle_load_script(self, filepath: Optional[str]) -> None:
        """Handle result from LoadScriptDialog."""
        if not filepath:
            return

        path = Path(filepath)
        if not path.exists():
            return

        # Update sticky directory
        self._last_script_dir = path.parent

        # Load script into editor
        content = path.read_text()
        editor = self.query_one(EditorRegion)
        editor.set_script(content)

    def action_save_state(self) -> None:
        """Open dialog to save Agent state."""
        self.push_screen(
            SaveStateDialog(
                title="Save Agent State",
                initial_path=str(self._last_state_dir),
            ),
            self._handle_save_state,
        )

    def _handle_save_state(self, filepath: Optional[str]) -> None:
        """Handle result from SaveStateDialog."""
        if not filepath or not self._agent:
            return

        path = Path(filepath)
        self._last_state_dir = path.parent

        # Save Agent
        self._agent.save(path)
        self.log(f"Saved Agent state to {path}")

    def action_load_state(self) -> None:
        """Open dialog to load Agent state."""
        self.push_screen(
            LoadStateDialog(
                title="Load Agent State",
                initial_path=str(self._last_state_dir),
            ),
            self._handle_load_state,
        )

    def _handle_load_state(self, filepath: Optional[str]) -> None:
        """Handle result from LoadStateDialog."""
        if not filepath:
            return

        path = Path(filepath)
        if not path.exists():
            return

        self._last_state_dir = path.parent

        # Load Agent
        self._agent = Agent.load(path)
        self._setup_events()
        self.log(f"Loaded Agent state from {path}")

    def action_run_script(self) -> None:
        """Toggle Run mode: compile → diff → submit all pending.

        Implements HRN-5: all pending entries are submitted sequentially
        without pausing.  Replaces the old dev-mode vs non-dev-mode
        branching with a single harness-aware logic path.
        """
        if self._execution_state == ExecutionState.RUNNING:
            # Toggle off — cancel in-flight submission
            self._cancelled = True
            self._run_mode_active = False
            self._set_state(ExecutionState.IDLE)
            return

        # Toggle on
        self._cancelled = False
        self._run_mode_active = True
        self._set_state(ExecutionState.RUNNING)

        # Compile fresh entries from editor content
        entries = self._compile_script()
        if not entries:
            self._run_mode_active = False
            self._set_state(ExecutionState.IDLE)
            return

        # Diff: find entries not yet submitted
        pending = self._get_pending(entries)
        if not pending:
            # All entries already submitted — nothing to do
            self._run_mode_active = False
            self._set_state(ExecutionState.IDLE)
            return

        # Submit all pending entries via async worker
        self.run_worker(self._submit_all_pending(pending))

    async def _submit_all_pending(self, entries: list[CompiledEntry]) -> None:
        """Submit all pending entries sequentially to the Agent.

        Implements HRN-5: all pending entries are submitted without pausing.
        For each entry:
        - Convert to KLine and call rationalise
        - Fast-path (True): add to _satisfied immediately (HRN-3)
        - Slow-path (False): event callback handles the proposal;
          auto-countersign happens there, not here
        - Yield control between entries to keep UI responsive

        Args:
            entries: List of pending CompiledEntry objects to submit.
        """
        for entry in entries:
            if self._cancelled:
                break

            kline = self._entry_to_kline(entry)
            key = self._entry_key(entry)

            if self._agent:
                result = self._agent.rationalise(kline)
                self._submitted.add(key)
                self._fast_path_results[key] = result
                if result:
                    # Fast-path auto-satisfaction (HRN-3)
                    self._satisfied.add(key)

            await asyncio.sleep(0)

        # Run complete — return to idle
        self._run_mode_active = False
        self._set_state(ExecutionState.IDLE)

    def action_step_script(self) -> None:
        """Execute one pending KLine: compile → diff → submit first pending → halt.

        Implements HRN-7 (Step submits one and halts).
        """
        # Step disabled during Run
        if self._execution_state == ExecutionState.RUNNING:
            return

        # Compile editor content
        entries = self._compile_script()
        if not entries:
            # _compile_script already displayed the error response
            return

        # Diff against submitted
        pending = self._get_pending(entries)
        if not pending:
            return

        # Submit first pending entry
        entry = pending[0]
        kline = self._entry_to_kline(entry)
        key = self._entry_key(entry)

        if self._agent:
            result = self._agent.rationalise(kline)
            self._submitted.add(key)
            if result:
                # Fast path: auto-satisfied (event handler displays ✓)
                self._satisfied.add(key)
            # Slow path: event handler displays ◌, user may ratify later

        # Halt
        self._set_state(ExecutionState.HALTED)

    def action_clear_responses(self) -> None:
        """Clear the responses list and reset all tracking state."""
        responses = self.query_one(ResponsesRegion)
        responses.clear()
        self._decompiler.clear()
        self._submitted.clear()
        self._satisfied.clear()
        # Reset selection state and disable Ratify button
        self._selected_proposal = None
        self._selected_entry_key = None
        toolbar = self.query_one(ToolbarRegion)
        toolbar.set_ratify_enabled(False)


def main() -> None:
    """Run the KScript TUI app."""
    parser = argparse.ArgumentParser(description="KScript TUI - Interactive development environment")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable dev mode (state restore, SIGTERM handler for harness)",
    )
    args = parser.parse_args()

    app = KScriptApp(dev_mode=args.dev)
    app.run()


if __name__ == "__main__":
    main()
