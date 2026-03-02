"""Chat history region component."""

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Label, ListView, ListItem, Static


class ChatHistoryRegion(Container):
    """Chat history region with selectable list of previous chats."""

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
    """

    def __init__(self) -> None:
        super().__init__()
        self._chats: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Label("History", classes="history-title")
        yield ListView(id="history-list")

    def update_history(self, chats: list[dict]) -> None:
        """Update the history list with chat entries."""
        self._chats = chats
        list_view = self.query_one("#history-list", ListView)
        list_view.clear()

        for i, chat in enumerate(chats):
            preview = chat.get("chat", "")[:50]
            if len(chat.get("chat", "")) > 50:
                preview += "..."
            list_view.append(
                ListItem(
                    Static(preview or f"Chat {i + 1}", classes="chat-preview"),
                    id=f"chat-item-{i}"
                )
            )

    def get_selected_index(self) -> int:
        """Get the index of the selected item."""
        list_view = self.query_one("#history-list", ListView)
        if list_view.highlighted_child:
            item_id = list_view.highlighted_child.id
            if item_id:
                return int(item_id.replace("chat-item-", ""))
        return -1

    def get_chat_at(self, index: int) -> dict | None:
        """Get chat at the specified index."""
        if 0 <= index < len(self._chats):
            return self._chats[index]
        return None
