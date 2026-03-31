"""Toolbar region with action buttons and status indicator."""

from enum import Enum
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Static
from textual.css.query import NoMatches


class ExecutionState(Enum):
    """Execution state for the KScript engine."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    HALTED = "HALTED"


class ToolbarRegion(Horizontal):
    """Toolbar with action buttons and status indicator."""

    DEFAULT_CSS = """
    ToolbarRegion {
        height: 3;
        padding: 0 1;
        align: left middle;
    }

    ToolbarRegion Button {
        min-width: 8;
        margin-left: 1;
        margin-right: 1;
    }

    ToolbarRegion .status-indicator {
        margin-left: 2;
        padding: 0 2;
        content-align: center middle;
    }

    ToolbarRegion .status-idle {
        color: $text-muted;
    }

    ToolbarRegion .status-running {
        color: $success;
    }

    ToolbarRegion .status-halted {
        color: $warning;
    }
    """

    execution_state: reactive[ExecutionState] = reactive(ExecutionState.IDLE)

    class LoadScript(Message):
        """Request to load a .ks script file."""
        pass

    class SaveState(Message):
        """Request to save Kalvin model state."""
        pass

    class LoadState(Message):
        """Request to load Kalvin model state."""
        pass

    class Run(Message):
        """Request to toggle run/compile loop."""
        pass

    class Step(Message):
        """Request to step one KLine."""
        pass

    class Resume(Message):
        """Request to resume execution."""
        pass

    class Clear(Message):
        """Request to clear responses."""
        pass

    def compose(self) -> ComposeResult:
        yield Button("Load.ks", id="load-script-btn")
        yield Button("Save.k", id="save-state-btn")
        yield Button("Load.k", id="load-state-btn")
        yield Button("Run", id="run-btn")
        yield Button("Step", id="step-btn")
        yield Button("Clear", id="clear-btn")
        yield Static(self._get_status_text(), id="status-indicator", classes="status-indicator status-idle")

    def _get_status_text(self) -> str:
        """Get status text based on current state."""
        state = self.execution_state
        indicator = {
            ExecutionState.IDLE: "○",
            ExecutionState.RUNNING: "●",
            ExecutionState.HALTED: "◐",
        }
        return f"{indicator[state]} {state.value}"

    def _get_status_class(self) -> str:
        """Get CSS class based on current state."""
        return {
            ExecutionState.IDLE: "status-idle",
            ExecutionState.RUNNING: "status-running",
            ExecutionState.HALTED: "status-halted",
        }[self.execution_state]

    def watch_execution_state(self, old_state: ExecutionState, new_state: ExecutionState) -> None:
        """Update status indicator when state changes."""
        # Widget may not be mounted yet during initial reactive set
        try:
            indicator = self.query_one("#status-indicator", Static)
        except NoMatches:
            return

        indicator.update(self._get_status_text())
        indicator.set_classes(f"status-indicator {self._get_status_class()}")

        # Update button states
        self._update_button_states()

    def _update_button_states(self) -> None:
        """Enable/disable buttons based on execution state."""
        state = self.execution_state

        run_btn = self.query_one("#run-btn", Button)
        step_btn = self.query_one("#step-btn", Button)

        run_btn.label = "Stop" if state == ExecutionState.RUNNING else "Run"
        step_btn.disabled = state == ExecutionState.RUNNING

    def on_mount(self) -> None:
        """Initialize button states."""
        self._update_button_states()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "load-script-btn":
            self.post_message(self.LoadScript())
        elif button_id == "save-state-btn":
            self.post_message(self.SaveState())
        elif button_id == "load-state-btn":
            self.post_message(self.LoadState())
        elif button_id == "run-btn":
            self.post_message(self.Run())
        elif button_id == "step-btn":
            self.post_message(self.Step())
        elif button_id == "clear-btn":
            self.post_message(self.Clear())

    def set_state(self, state: ExecutionState) -> None:
        """Set the execution state."""
        self.execution_state = state
