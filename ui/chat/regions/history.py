"""Chat history region component with KScript and chat output display."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Label, ListView, ListItem, Static


class ChatHistoryRegion(Container):
    """Chat history region with output display and selectable history."""

    DEFAULT_CSS = """
    ChatHistoryRegion {
        width: 1fr;
        height: 1fr;
        padding: 1;
        border: solid $primary;
    }

    ChatHistoryRegion .history-title {
        text-style: bold;
        margin-bottom: 1;
    }

    ChatHistoryRegion ListView {
        height: 1fr;
        border: solid $panel;
        background: $surface;
    }

    ChatHistoryRegion ListItem {
        padding: 1;
    }

    ChatHistoryRegion ListItem:hover {
        background: $accent 20%;
    }

    ChatHistoryRegion ListItem.-active {
        background: $accent 40%;
    }

    ChatHistoryRegion .chat-preview {
        height: auto;
    }

    ChatHistoryRegion .preview-type {
        color: $accent;
        text-style: bold;
    }

    ChatHistoryRegion .output-area {
        height: 1fr;
        padding: 1;
        background: $surface;
        border: solid $panel;
        overflow-y: auto;
    }

    ChatHistoryRegion .output-label {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._history: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Label("Output History", classes="history-title")
        yield ListView(id="history-list")

    def update_history(self, history: list[dict]) -> None:
        """Update the history list with entries."""
        self._history = history
        list_view = self.query_one("#history-list", ListView)
        list_view.clear()

        for i, entry in enumerate(history):
            entry_type = entry.get("type", "chat")
            preview = self._make_preview(entry, i)
            list_view.append(
                ListItem(
                    Static(preview, classes="chat-preview"),
                )
            )

    def _make_preview(self, entry: dict, index: int) -> str:
        """Create a preview string for a history entry."""
        entry_type = entry.get("type", "chat")

        if entry_type == "script":
            # KScript compilation result
            script = entry.get("script", "")
            preview = script[:40].replace("\n", " ")
            if len(script) > 40:
                preview += "..."
            return f"[Script] {preview}"
        else:
            # Direct chat
            chat = entry.get("chat", "")
            preview = chat[:40]
            if len(chat) > 40:
                preview += "..."
            return f"[Chat] {preview}"

    def get_selected_index(self) -> int:
        """Get the index of the selected item."""
        list_view = self.query_one("#history-list", ListView)
        if list_view.highlighted_child:
            for i, child in enumerate(list_view.children):
                if child is list_view.highlighted_child:
                    return i
        return -1

    def get_entry_at(self, index: int) -> dict | None:
        """Get history entry at the specified index."""
        if 0 <= index < len(self._history):
            return self._history[index]
        return None

    def add_entry(self, entry: dict) -> None:
        """Add a new entry to history and update the view."""
        self._history.append(entry)
        self.update_history(self._history)

    def get_history(self) -> list[dict]:
        """Get the full history list."""
        return self._history.copy()

    def clear(self) -> None:
        """Clear all history."""
        self._history = []
        self.query_one("#history-list", ListView).clear()
