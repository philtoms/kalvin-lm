"""Chat region component."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Input, Label, Static


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

    ChatRegion .response-area {
        height: 1fr;
        padding: 1;
        background: $surface;
        border: solid $panel;
        overflow-y: auto;
    }

    ChatRegion .input-row {
        height: 3;
        margin-top: 1;
    }

    ChatRegion #chat-input {
        width: 1fr;
    }

    ChatRegion #send-btn {
        width: auto;
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Chat", classes="chat-title")
        yield Static("Response will appear here...", id="response-text", classes="response-area")
        with Horizontal(classes="input-row"):
            yield Input(placeholder="Type your message...", id="chat-input")
            yield Button("Send", id="send-btn", variant="primary")

    def get_input(self) -> str:
        """Get the current input text."""
        return self.query_one("#chat-input", Input).value

    def clear_input(self) -> None:
        """Clear the input field."""
        self.query_one("#chat-input", Input).value = ""

    def set_response(self, text: str) -> None:
        """Set the response text."""
        self.query_one("#response-text", Static).update(text)

    def append_response(self, text: str) -> None:
        """Append text to the response."""
        current = self.query_one("#response-text", Static).renderable
        if isinstance(current, str):
            new_text = current + "\n" + text if current else text
        else:
            new_text = text
        self.query_one("#response-text", Static).update(new_text)
