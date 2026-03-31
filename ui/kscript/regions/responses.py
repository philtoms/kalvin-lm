"""Responses region for displaying Kalvin response KLines."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import ListView, ListItem, Static

from kalvin.abstract import KLine


class ResponseItem(ListItem):
    """A single response item in the list displaying decompiled KScript."""

    DEFAULT_CSS = """
    ResponseItem {
        padding: 0 1;
    }
    ResponseItem:focus {
        background: $surface-lighten-1;
    }
    """

    def __init__(self, kline: KLine, decompiled_source: str) -> None:
        """Initialize with a KLine and its decompiled source.

        Args:
            kline: The KLine response.
            decompiled_source: The decompiled KScript source to display.
        """
        self.kline = kline
        self.decompiled_source = decompiled_source
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Static(self.decompiled_source)


class ResponsesRegion(Vertical):
    """Scrollable list of Kalvin response KLines."""

    DEFAULT_CSS = """
    ResponsesRegion {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }

    ResponsesRegion ListView {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(self) -> None:
        self._seen_responses: set[str] = set()
        super().__init__()

    class ResponseClicked(Message):
        """Emitted when a response item is clicked."""

        def __init__(self, kline: KLine, decompiled_source: str) -> None:
            self.kline = kline
            self.decompiled_source = decompiled_source
            super().__init__()

    def compose(self) -> ComposeResult:
        yield ListView(id="responses-list")

    def add_response(self, kline: KLine, decompiled_source: str) -> None:
        """Append a KLine response with its decompiled source to the list.

        Only adds the response if its signature hasn't been seen before.

        Args:
            kline: The KLine response to add.
            decompiled_source: The decompiled KScript source to display.
        """
        if decompiled_source in self._seen_responses:
            return
        self._seen_responses.add(decompiled_source)
        list_view = self.query_one("#responses-list", ListView)
        item = ResponseItem(kline, decompiled_source)
        list_view.append(item)
        # Scroll to the new item
        list_view.index = len(list_view) - 1

    def clear(self) -> None:
        """Clear all responses from the list."""
        self._seen_responses.clear()
        list_view = self.query_one("#responses-list", ListView)
        list_view.clear()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection of a response item."""
        if event.list_view.id == "responses-list" and isinstance(event.item, ResponseItem):
            self.post_message(
                self.ResponseClicked(event.item.kline, event.item.decompiled_source)
            )
