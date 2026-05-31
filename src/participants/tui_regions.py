"""Minimal Textual widgets for the TUI harness participant.

- **EventLog**: scrollable list displaying incoming KAgent events as text lines.
- **InputBar**: horizontal bar with a text input field and Send button for
  composing free-form messages to the Trainer.
- **RatifyBar**: horizontal bar with a Ratify button (disabled by default) and
  a status indicator.
- **EventItem**: a single event display item showing action and message summary.
- **InputBar**: single-line text input with a Send button that posts
  ``InputBar.Submitted`` messages on Enter or button click.
"""

from __future__ import annotations

import time
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, ListItem, RichLog, Static


class EventItem(ListItem):
    """A single event display item showing action and message summary."""

    def __init__(self, action: str, message: Any, timestamp: float | None = None) -> None:
        super().__init__()
        self.event_action = action
        self.event_message = message
        self.event_timestamp = timestamp or time.time()

    @property
    def event_data(self) -> dict[str, Any]:
        """Return the raw event data for ratification."""
        return {"action": self.event_action, "message": self.event_message}

    def compose(self) -> ComposeResult:
        ts = time.strftime("%H:%M:%S", time.localtime(self.event_timestamp))
        summary = str(self.event_message)
        if len(summary) > 80:
            summary = summary[:77] + "..."
        yield Static(f"[{ts}] {self.event_action}: {summary}")


class EventLog(Vertical):
    """Scrollable list displaying incoming KAgent events as text lines.

    Each line shows timestamp, action, and a summary of the message payload.
    """

    DEFAULT_CSS = """
    EventLog {
        height: 1fr;
        padding: 0 1;
        border: round $primary;
    }
    EventLog RichLog {
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._events: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield RichLog(id="event-log-viewer", highlight=True, markup=False)

    def add_event(self, frame: dict[str, Any]) -> None:
        """Add a harness event frame to the display.

        Parameters
        ----------
        frame:
            A full JSON frame dict with ``address``, ``action``, ``message``
            keys.
        """
        self._events.append(frame)
        ts = time.strftime("%H:%M:%S")
        action = frame.get("action", "?")
        message = frame.get("message")
        summary = str(message) if message is not None else ""
        if len(summary) > 120:
            summary = summary[:117] + "..."

        try:
            log = self.query_one("#event-log-viewer", RichLog)
            log.write(f"[{ts}] {action}: {summary}")
        except Exception:
            # Widget not mounted — data is still stored in _events
            pass

    @property
    def events(self) -> list[dict[str, Any]]:
        """Return all stored event frames."""
        return list(self._events)


class InputBar(Horizontal):
    """Horizontal bar with a text input field and Send button.

    The human types a free-form message and submits it via Enter key or the
    Send button. On submission, an ``InputBar.Submitted`` message is posted
    carrying the text, and the input field is cleared. Empty or whitespace-only
    submissions are silently ignored.

    Spec references: HRNS-26, HRNS-28.
    """

    DEFAULT_CSS = """
    InputBar {
        height: 3;
        padding: 0 1;
        align: left middle;
    }
    InputBar Input {
        width: 1fr;
        margin-right: 1;
    }
    InputBar Button {
        min-width: 10;
    }
    """

    class Submitted(Message):
        """Posted when the user submits text from the input bar.

        Attributes
        ----------
        text:
            The trimmed text the user entered.
        """

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type a message…", id="input-bar-field")
        yield Button("Send", id="input-send-btn")

    def _submit(self) -> None:
        """Read the input value, post Submitted if non-empty, then clear."""
        input_widget = self.query_one("#input-bar-field", Input)
        value = input_widget.value.strip()
        if value:
            self.post_message(self.Submitted(value))
        input_widget.value = ""

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in the input field."""
        if event.input.id == "input-bar-field":
            event.prevent_default()
            self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Send button click."""
        if event.button.id == "input-send-btn":
            self._submit()


class RatifyBar(Horizontal):
    """Horizontal bar with a Ratify button and status indicator.

    The Ratify button is disabled by default. Call ``enable_ratify()`` when
    an event is selected. When clicked, posts a ``RatifyClicked`` message
    carrying the selected event's raw message payload.
    """

    DEFAULT_CSS = """
    RatifyBar {
        height: 3;
        padding: 0 1;
        align: left middle;
    }
    RatifyBar Button {
        min-width: 10;
        margin-right: 1;
    }
    RatifyBar .status-text {
        margin-left: 2;
        padding: 0 2;
        content-align: center middle;
        color: $text-muted;
    }
    """

    class RatifyClicked(Message):
        """Posted when the Ratify button is clicked.

        Attributes
        ----------
        event_data:
            The raw ``message`` payload from the selected harness event frame.
        """

        def __init__(self, event_data: Any) -> None:
            self.event_data = event_data
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self._selected_event_data: Any = None

    def compose(self) -> ComposeResult:
        yield Button("Ratify", id="ratify-btn", disabled=True)
        yield Static("No event selected", id="ratify-status", classes="status-text")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ratify-btn" and self._selected_event_data is not None:
            self.post_message(self.RatifyClicked(self._selected_event_data))

    def enable_ratify(self, event_data: Any) -> None:
        """Enable the Ratify button with the given event data."""
        self._selected_event_data = event_data
        try:
            btn = self.query_one("#ratify-btn", Button)
            btn.disabled = False
            status = self.query_one("#ratify-status", Static)
            summary = str(event_data)
            if len(summary) > 40:
                summary = summary[:37] + "..."
            status.update(f"Selected: {summary}")
        except Exception:
            # Widget not mounted — data is still stored in _selected_event_data
            pass

    def disable_ratify(self) -> None:
        """Disable the Ratify button and clear selection."""
        self._selected_event_data = None
        try:
            btn = self.query_one("#ratify-btn", Button)
            btn.disabled = True
            status = self.query_one("#ratify-status", Static)
            status.update("No event selected")
        except Exception:
            # Widget not mounted — data is still cleared
            pass


class InputBar(Horizontal):
    """Single-line text input with a Send button.

    When the user presses Enter in the input field or clicks the Send button,
    the widget posts an ``InputBar.Submitted`` message carrying the typed text
    and clears the input field.
    """

    DEFAULT_CSS = """
    InputBar {
        height: 3;
        padding: 0 1;
        align: left middle;
    }
    InputBar Input {
        width: 1fr;
    }
    InputBar Button {
        min-width: 8;
        margin-left: 1;
    }
    """

    class Submitted(Message):
        """Posted when the user submits text via Enter or Send button.

        Attributes
        ----------
        text:
            The text that was submitted.
        """

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type a message…", id="input-field")
        yield Button("Send", id="send-btn")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in the input field."""
        if event.input.id == "input-field" and event.value:
            self.post_message(self.Submitted(event.value))
            event.input.value = ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Send button click."""
        if event.button.id == "send-btn":
            try:
                input_widget = self.query_one("#input-field", Input)
                text = input_widget.value
                if text:
                    self.post_message(self.Submitted(text))
                    input_widget.value = ""
            except Exception:
                # Widget not mounted — nothing to do
                pass

    def clear(self) -> None:
        """Clear the input field."""
        try:
            input_widget = self.query_one("#input-field", Input)
            input_widget.value = ""
        except Exception:
            # Widget not mounted — nothing to do
            pass
