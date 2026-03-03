"""Chat region component with KScript editor."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, Label, TextArea

SCRIPT_PLACEHOLDER = (
    "# Enter KScript here...\n# Example:\n#   greeting = hello world\n#   question > greeting ?"
)


class ChatRegion(Container):
    """Chat region with KScript editor and direct chat input."""

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

    ChatRegion .editor-container {
        height: 1fr;
    }

    ChatRegion .script-editor {
        height: 1fr;
        padding: 1;
        background: $surface;
        border: solid $panel;
    }

    ChatRegion .script-editor:focus {
        border: solid $accent;
    }

    ChatRegion .editor-buttons {
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
        yield Label("Script Editor", classes="chat-title")
        with Vertical(classes="editor-container"):
            yield TextArea(
                SCRIPT_PLACEHOLDER,
                id="script-text",
                classes="script-editor",
                language="python",  # Use Python highlighting as closest match
            )
            with Horizontal(classes="editor-buttons"):
                yield Button("Clear", id="clear-script-btn", variant="default")
                yield Button("Run", id="run-script-btn", variant="success")
        with Horizontal(classes="input-row"):
            yield Input(placeholder="Direct chat message...", id="chat-input")
            yield Button("Chat", id="send-btn", variant="primary")

    def on_descendant_focus(self, event) -> None:
        """Clear placeholder when TextArea gets focus."""
        if event.widget and event.widget.id == "script-text" and self._placeholder_active:
            self.query_one("#script-text", TextArea).text = ""
            self._placeholder_active = False

    def get_input(self) -> str:
        """Get the current chat input text."""
        return self.query_one("#chat-input", Input).value

    def clear_input(self) -> None:
        """Clear the chat input field."""
        self.query_one("#chat-input", Input).value = ""

    def get_script(self) -> str:
        """Get the current script text (excludes placeholder)."""
        if self._placeholder_active:
            return ""
        return self.query_one("#script-text", TextArea).text

    def set_script(self, text: str) -> None:
        """Set the script text."""
        self._placeholder_active = False
        self.query_one("#script-text", TextArea).text = text

    def append_script(self, text: str) -> None:
        """Append text to the script."""
        if self._placeholder_active:
            self._placeholder_active = False
            self.query_one("#script-text", TextArea).text = text
        else:
            current = self.query_one("#script-text", TextArea).text
            new_text = current + "\n" + text if current else text
            self.query_one("#script-text", TextArea).text = new_text

    def clear_script(self) -> None:
        """Clear the script area."""
        self._placeholder_active = True
        self.query_one("#script-text", TextArea).text = SCRIPT_PLACEHOLDER
