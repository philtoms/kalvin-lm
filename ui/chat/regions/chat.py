"""Chat region component."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, Label, TextArea

PLACEHOLDER_TEXT = "Response will appear here..."


class ChatRegion(Container):
    """Chat region with input and response display."""

    DEFAULT_CSS = """
    ChatRegion {
        height: 1fr;
        padding: 1;
        border: solid $primary;
    }

    ChatRegion .chat-title {
        text-style: bold;
        margin-bottom: 1;
    }

    ChatRegion .response-container {
        height: 1fr;
    }

    ChatRegion .response-area {
        height: 1fr;
        padding: 1;
        background: $surface;
        border: solid $panel;
    }

    ChatRegion .response-area:focus {
        border: solid $accent;
    }

    ChatRegion .response-buttons {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }

    ChatRegion .input-row {
        height: 3;
        margin-top: 1;
    }

    ChatRegion #chat-input {
        width: 1fr;
    }
    """

    _placeholder_active: bool = True

    def compose(self) -> ComposeResult:
        yield Label("Chat", classes="chat-title")
        with Vertical(classes="response-container"):
            yield TextArea(PLACEHOLDER_TEXT, id="response-text", classes="response-area")
            with Horizontal(classes="response-buttons"):
                yield Button("Submit", id="submit-response-btn", variant="primary")
        with Horizontal(classes="input-row"):
            yield Input(placeholder="Type your message...", id="chat-input")
            yield Button("Send", id="send-btn", variant="primary")

    def on_descendant_focus(self, event) -> None:
        """Clear placeholder when TextArea gets focus."""
        if event.widget and event.widget.id == "response-text" and self._placeholder_active:
            self.query_one("#response-text", TextArea).text = ""
            self._placeholder_active = False

    def on_descendant_blur(self, event) -> None:
        """Show placeholder if empty when TextArea loses focus."""
        if event.widget and event.widget.id == "response-text":
            text_area = self.query_one("#response-text", TextArea)
            if not text_area.text.strip():
                text_area.text = PLACEHOLDER_TEXT
                self._placeholder_active = True

    def get_input(self) -> str:
        """Get the current input text."""
        return self.query_one("#chat-input", Input).value

    def clear_input(self) -> None:
        """Clear the input field."""
        self.query_one("#chat-input", Input).value = ""

    def get_response(self) -> str:
        """Get the current response text (excludes placeholder)."""
        if self._placeholder_active:
            return ""
        return self.query_one("#response-text", TextArea).text

    def set_response(self, text: str) -> None:
        """Set the response text."""
        self._placeholder_active = False
        self.query_one("#response-text", TextArea).text = text

    def append_response(self, text: str) -> None:
        """Append text to the response."""
        if self._placeholder_active:
            self._placeholder_active = False
            self.query_one("#response-text", TextArea).text = text
        else:
            current = self.query_one("#response-text", TextArea).text
            new_text = current + "\n" + text if current else text
            self.query_one("#response-text", TextArea).text = new_text

    def clear_response(self) -> None:
        """Clear the response area."""
        self._placeholder_active = True
        self.query_one("#response-text", TextArea).text = PLACEHOLDER_TEXT
