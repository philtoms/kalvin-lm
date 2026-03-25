"""Responses region for displaying Kalvin response KLines."""

import json
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import ListView, ListItem, Static

from kalvin.abstract import KLine


class ResponseItem(ListItem):
    """A single response item in the list."""

    DEFAULT_CSS = """
    ResponseItem {
        padding: 0 1;
    }
    ResponseItem:focus {
        background: $surface-lighten-1;
    }
    """

    def __init__(self, kline: KLine) -> None:
        """Initialize with a KLine.

        Args:
            kline: The KLine response to display.
        """
        self.kline = kline
        super().__init__()

    def compose(self) -> ComposeResult:
        # Format as JSON: {"signature": nodes}
        nodes = self.kline.nodes
        if nodes is None:
            node_repr = None
        elif isinstance(nodes, int):
            node_repr = nodes
        else:
            node_repr = nodes

        json_str = json.dumps({str(self.kline.signature): node_repr})
        yield Static(json_str)


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

    class ResponseClicked(Message):
        """Emitted when a response item is clicked."""
        def __init__(self, kline: KLine) -> None:
            self.kline = kline
            super().__init__()

    def compose(self) -> ComposeResult:
        yield ListView(id="responses-list")

    def add_response(self, kline: KLine) -> None:
        """Append a KLine response to the list.

        Args:
            kline: The KLine response to add.
        """
        list_view = self.query_one("#responses-list", ListView)
        item = ResponseItem(kline)
        list_view.append(item)
        # Scroll to the new item
        list_view.index = len(list_view) - 1

    def clear(self) -> None:
        """Clear all responses from the list."""
        list_view = self.query_one("#responses-list", ListView)
        list_view.clear()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection of a response item."""
        if event.list_view.id == "responses-list" and isinstance(event.item, ResponseItem):
            self.post_message(self.ResponseClicked(event.item.kline))
