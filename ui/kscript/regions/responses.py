"""Responses region for displaying Kalvin response KLines."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, ListView, ListItem, Static

from kalvin.abstract import KLine
from kalvin.significance import Int32Significance


class FilterBar(Horizontal):
    """Container for filter buttons with horizontal layout."""

    DEFAULT_CSS = """
    FilterBar {
        height: 3;
        width: 100%;
        padding: 0;
        margin-bottom: 1;
    }

    FilterBar Button {
        min-width: 5;
        height: 3;
        padding: 0 2;
        margin: 0 1;
    }

    FilterBar Button.active {
        background: $primary;
        color: $background;
    }

    FilterBar Button.inactive {
        background: $surface;
        color: $foreground;
    }
    """


class ResponseItem(ListItem):
    """A single response item in the list displaying decompiled KScript."""

    DEFAULT_CSS = """
    ResponseItem {
        padding: 0 1;
    }
    ResponseItem:focus {
        background: $surface-lighten-1;
    }
    ResponseItem.hidden {
        display: none;
    }
    """

    def __init__(self, kline: KLine, decompiled_source: str, level: str) -> None:
        """Initialize with a KLine and its decompiled source.

        Args:
            kline: The KLine response.
            decompiled_source: The decompiled KScript source to display.
            level: The significance level (S1, S2, S3, S4, MCS).
        """
        self.kline = kline
        self.decompiled_source = decompiled_source
        self.level = level
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Static(self.decompiled_source)


class ResponsesRegion(Vertical):
    """Scrollable list of Kalvin response KLines with sig level filters."""

    DEFAULT_CSS = """
    ResponsesRegion {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }

    ResponsesRegion FilterBar {
        height: 3;
        width: 100%;
        padding: 0;
        margin-bottom: 1;
    }

    ResponsesRegion ListView {
        width: 100%;
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        self._seen_responses: set[str] = set()
        self._sig = Int32Significance()
        # Filter state: default S1 on
        self._filters: dict[str, bool] = {
            "S1": True,
            "S2": False,
            "S3": False,
            "S4": False,
            "MCS": False,
        }
        super().__init__()

    class ResponseClicked(Message):
        """Emitted when a response item is clicked."""

        def __init__(self, kline: KLine, decompiled_source: str) -> None:
            self.kline = kline
            self.decompiled_source = decompiled_source
            super().__init__()

    class FilterToggled(Message):
        """Emitted when a filter button is toggled."""

        def __init__(self, level: str, active: bool) -> None:
            self.level = level
            self.active = active
            super().__init__()

    def compose(self) -> ComposeResult:
        with FilterBar():
            yield Button("S1", id="filter-s1", classes="active" if self._filters["S1"] else "inactive")
            yield Button("S2", id="filter-s2", classes="active" if self._filters["S2"] else "inactive")
            yield Button("S3", id="filter-s3", classes="active" if self._filters["S3"] else "inactive")
            yield Button("S4", id="filter-s4", classes="active" if self._filters["S4"] else "inactive")
            yield Button("MCS", id="filter-mcs", classes="active" if self._filters["MCS"] else "inactive")
        yield ListView(id="responses-list")

    def _get_kline_level(self, kline: KLine) -> str:
        """Get the significance level for a KLine."""
        level = self._sig.get_level(kline.signature)
        # Check for MCS: S2 with signature == OR of all nodes
        if level == "S2":
            nodes = kline.as_node_list()
            if len(nodes) >= 2:
                base_token = self._sig.strip_significance(kline.signature)
                nodes_or = 0
                for node in nodes:
                    nodes_or |= node
                if base_token == nodes_or:
                    return "MCS"
        return level

    def _is_filter_active(self, level: str) -> bool:
        """Check if a filter level is active."""
        return self._filters.get(level, False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle filter button press."""
        button_id = event.button.id
        if button_id and button_id.startswith("filter-"):
            level = button_id.replace("filter-", "").upper()
            if level in self._filters:
                # Toggle state
                new_state = not self._filters[level]
                self._filters[level] = new_state
                # Update button classes
                event.button.set_class(new_state, "active")
                event.button.set_class(not new_state, "inactive")
                self._update_list_visibility()
                self.post_message(self.FilterToggled(level, new_state))

    def _update_list_visibility(self) -> None:
        """Update visibility of items based on active filters."""
        list_view = self.query_one("#responses-list", ListView)
        for item in list_view.children:
            if isinstance(item, ResponseItem):
                item.set_class(self._is_filter_active(item.level), "visible")
                item.set_class(not self._is_filter_active(item.level), "hidden")

    def add_response(self, kline: KLine, decompiled_source: str) -> None:
        """Append a KLine response with its decompiled source to the list.

        Only adds the response if its signature hasn't been seen before.

        Args:
            kline: The KLine response to add.
            decompiled_source: The decompiled KScript source to display.
        """
        if decompiled_source in self._seen_responses:
            return

        level = self._get_kline_level(kline)
        self._seen_responses.add(decompiled_source)

        list_view = self.query_one("#responses-list", ListView)
        item = ResponseItem(kline, decompiled_source, level)

        # Set visibility based on current filter state
        is_visible = self._is_filter_active(level)
        item.set_class(not is_visible, "hidden")

        list_view.append(item)
        # Scroll to the new item only if visible
        if is_visible:
            list_view.index = len(list_view) - 1

    def clear(self) -> None:
        """Clear all responses from the list."""
        self._seen_responses.clear()
        list_view = self.query_one("#responses-list", ListView)
        list_view.clear()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection of a response item."""
        if event.list_view.id == "responses-list" and isinstance(event.item, ResponseItem):
            self.post_message(self.ResponseClicked(event.item.kline, event.item.decompiled_source))
