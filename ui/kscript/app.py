"""KScript TUI Application - Interactive development environment for KScript and Kalvin."""

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
from textual.widgets import Footer, Header

from kalvin import Kalvin
from kalvin.abstract import KLine
from kscript import KScript, CompiledEntry
from kscript.decompiler import Decompiler
from ui.kscript.dialogs import LoadScriptDialog, SaveStateDialog, LoadStateDialog
from ui.kscript.regions import EditorRegion, ResponsesRegion, ToolbarRegion
from ui.kscript.regions.toolbar import ExecutionState

logger = logging.getLogger(__name__)

DEFAULT_SCRIPTS_DIR = Path("data/scripts")

# State file paths for dev mode (harness-driven save/restore)
_STATE_DIR = Path(__file__).parent.parent.parent  # project root
MODEL_STATE_FILE = _STATE_DIR / ".tmp.bin"
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
        self._kalvin: Optional[Kalvin] = None
        self._decompiler: Decompiler = Decompiler()
        self._execution_state: ExecutionState = ExecutionState.IDLE
        self._pending_entries: list[CompiledEntry] = []
        self._current_entry_index: int = 0
        self._cancelled: bool = False
        self._last_script_dir: Path = DEFAULT_SCRIPTS_DIR
        self._last_state_dir: Path = Path("data")
        self._auto_compile_interval: float = auto_compile_interval

    def on_mount(self) -> None:
        """Initialize Kalvin instance on app start."""
        DEFAULT_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)

        if self._dev_mode:
            self._restore_state()
            signal.signal(signal.SIGTERM, self._on_sigterm)
        else:
            self._kalvin = Kalvin()

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

    # === Responses Event Handlers ===

    def on_responses_region_response_clicked(self, event: ResponsesRegion.ResponseClicked) -> None:
        """Handle click on a response item - halt and append to editor."""
        # Stop if running
        if self._execution_state == ExecutionState.RUNNING:
            self._cancelled = True
            self._set_state(ExecutionState.IDLE)

        # Append decompiled KScript source to editor
        editor = self.query_one(EditorRegion)
        editor.append_to_script(event.decompiled_source)

    # === State Management ===

    def _set_state(self, state: ExecutionState) -> None:
        """Update execution state and toolbar."""
        self._execution_state = state
        toolbar = self.query_one(ToolbarRegion)
        toolbar.set_state(state)

    # === Dev Mode State (SIGTERM save, mount restore) ===

    def _on_sigterm(self, signum, frame) -> None:
        """Handle SIGTERM from harness: save state and exit."""
        self._save_state()
        sys.exit(0)

    def _save_state(self) -> None:
        """Save Kalvin model and UI state to temp files."""
        if self._kalvin:
            self._kalvin.save(MODEL_STATE_FILE)

        editor = self.query_one(EditorRegion)
        ui_state = {
            "editor_content": editor.get_script(),
            "last_script_dir": str(self._last_script_dir),
            "last_state_dir": str(self._last_state_dir),
            "execution_state": self._execution_state.name,
        }
        with open(UI_STATE_FILE, "w") as f:
            json.dump(ui_state, f, indent=2)

    def _restore_state(self) -> None:
        """Restore Kalvin model and UI state from temp files."""
        # Restore Kalvin model
        if MODEL_STATE_FILE.exists():
            try:
                self._kalvin = Kalvin.load(MODEL_STATE_FILE)
                self.log("Restored Kalvin model state")
            except Exception as e:
                self.log(f"Failed to load model state: {e}")
                self._kalvin = Kalvin()
        else:
            self._kalvin = Kalvin()

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
                self.log("Restored UI state")
            except (json.JSONDecodeError, Exception) as e:
                self.log(f"Failed to load UI state: {e}")

        # Clean up temp files
        MODEL_STATE_FILE.unlink(missing_ok=True)
        UI_STATE_FILE.unlink(missing_ok=True)

    # === Script Compilation ===

    def _compile_script(self) -> Optional[list[CompiledEntry]]:
        """Compile the current editor content to KLine entries.

        Returns:
            List of CompiledEntry objects, or None on error.
        """
        editor = self.query_one(EditorRegion)
        script = editor.get_script()

        if not script.strip():
            return None

        try:
            model = KScript(script, dev=self._dev_mode)
            return model.entries
        except Exception as e:
            self.log(f"Compilation error: {e}")
            return None

    def _entry_to_kline(self, entry: CompiledEntry) -> KLine:
        """Convert a CompiledEntry to a KLine for Kalvin.

        Since CompiledEntry extends KLine, this is a simple cast.

        Args:
            entry: The compiled entry to convert.

        Returns:
            A KLine suitable for rationalise().
        """
        return entry

    def _decompile_response(self, klines: list[KLine]) -> str:
        """Decompile a list of KLines to KScript source.

        Args:
            klines: List of KLines from rationalise response.

        Returns:
            Decompile KScript source string.
        """
        if not klines:
            return ""
        entries = self._decompiler.decompile(klines)
        return "\n".join(e.to_kscript() for e in entries)

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
        """Open dialog to save Kalvin model state."""
        self.push_screen(
            SaveStateDialog(
                title="Save Kalvin State",
                initial_path=str(self._last_state_dir),
            ),
            self._handle_save_state,
        )

    def _handle_save_state(self, filepath: Optional[str]) -> None:
        """Handle result from SaveStateDialog."""
        if not filepath or not self._kalvin:
            return

        path = Path(filepath)
        self._last_state_dir = path.parent

        # Save Kalvin model
        self._kalvin.save(path)
        self.log(f"Saved Kalvin state to {path}")

    def action_load_state(self) -> None:
        """Open dialog to load Kalvin model state."""
        self.push_screen(
            LoadStateDialog(
                title="Load Kalvin State",
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

        # Load Kalvin model
        self._kalvin = Kalvin.load(path)
        self.log(f"Loaded Kalvin state from {path}")

    def action_run_script(self) -> None:
        """Toggle auto-compile loop on/off."""
        if self._execution_state == ExecutionState.RUNNING:
            # Stop the loop
            self._cancelled = True
            self._set_state(ExecutionState.IDLE)
            return

        # Start the loop
        self._cancelled = False
        self._set_state(ExecutionState.RUNNING)
        if self._dev_mode:
            self.run_worker(self._auto_compile_tick())
            self._set_state(ExecutionState.IDLE)
        else:
            self.run_worker(self._auto_compile_loop())

    async def _auto_compile_loop(self) -> None:
        """Periodically compile editor content and submit new entries."""
        # Immediate first tick
        await self._auto_compile_tick()
        while not self._cancelled:
            await asyncio.sleep(self._auto_compile_interval)
            if self._cancelled:
                return
            await self._auto_compile_tick()

    async def _auto_compile_tick(self) -> None:
        """Compile current editor content and add new responses."""
        entries = self._compile_script()
        if not entries:
            return
        responses = self.query_one(ResponsesRegion)
        for entry in entries:
            if self._cancelled:
                return
            kline = self._entry_to_kline(entry)
            if self._kalvin:
                response_klines = self._kalvin.rationalise(kline)
                if response_klines is None or len(response_klines) == 0:
                    response_klines = [kline]
                decompiled = self._decompile_response(response_klines)
                responses.add_response(kline, decompiled)
            await asyncio.sleep(0)

    def action_step_script(self) -> None:
        """Execute one KLine at a time."""
        if self._execution_state == ExecutionState.RUNNING:
            return

        # If no pending entries, compile fresh
        if not self._pending_entries:
            entries = self._compile_script()
            if not entries:
                return
            self._pending_entries = entries
            self._current_entry_index = 0

        # Execute single entry
        if self._current_entry_index < len(self._pending_entries):
            entry = self._pending_entries[self._current_entry_index]
            kline = self._entry_to_kline(entry)

            if self._kalvin:
                response_klines = self._kalvin.rationalise(kline)
                if response_klines is None or len(response_klines) == 0:
                    response_klines = [kline]
                decompiled = self._decompile_response(response_klines)
                responses = self.query_one(ResponsesRegion)
                responses.add_response(kline, decompiled)

            self._current_entry_index += 1

            # Check if done
            if self._current_entry_index >= len(self._pending_entries):
                self._set_state(ExecutionState.IDLE)
                self._pending_entries = []
                self._current_entry_index = 0
            else:
                self._set_state(ExecutionState.HALTED)

    def action_clear_responses(self) -> None:
        """Clear the responses list."""
        responses = self.query_one(ResponsesRegion)
        responses.clear()
        self._decompiler.clear()


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
