"""Responses region for displaying Model response KLines."""

import time

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

    def __init__(self, level: str, decompiled_source: str) -> None:
        """Initialize with a KLine and its decompiled source.

        Args:
            level: The significance level (S1, S2, S3, S4, MCS).
            decompiled_source: The decompiled KScript source to display.
        """
        self.level = level
        self.decompiled_source = decompiled_source
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Static(self.decompiled_source)


class ResponsesRegion(Vertical):
    """Scrollable list of Model response KLines with sig level filters."""

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
        # Filter state: default on
        self._filters: dict[str, bool] = {
            "S1": True,
            "S2": True,
            "S3": True,
            "S4": True,
            "MCS": True,
        }
        # Track last click time for double-click detection
        self._last_click_time: float = time.time()
        self._double_click_threshold: float = 0.5  # secs
        super().__init__()

    class ResponseClicked(Message):
        """Emitted when a response item is clicked."""

        def __init__(self, decompiled_source: str) -> None:
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
        # Check for MCS: S2 with signature == AND of all nodes
        if level == "S2":
            nodes = kline.as_node_list()
            if len(nodes) >= 2:
                base_token = self._sig.strip(kline.signature)
                for node in nodes:
                    if not base_token & node:
                        break
                return "MCS"
        return level

    def _is_filter_active(self, level: str) -> bool:
        """Check if a filter level is active."""
        return self._filters.get(level, False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle filter button press (single: toggle, double: solo)."""
        button_id = event.button.id
        if button_id and button_id.startswith("filter-"):
            level = button_id.replace("filter-", "").upper()
            if level in self._filters:
                now = time.time()

                # Check for double-click
                dc_time = now - self._last_click_time
                self._last_click_time = now
                if dc_time < self._double_click_threshold:
                    # Double-click: solo this level (ON), all others OFF
                    for lvl in self._filters:
                        self._filters[lvl] = (lvl == level)
                    # Update all button classes
                    for btn in self.query(Button):
                        btn_id = btn.id or ""
                        if btn_id.startswith("filter-"):
                            btn_level = btn_id.replace("filter-", "").upper()
                            self._filters[btn_level] = False
                            btn.set_class(False, "active")
                            btn.set_class(True, "inactive")
                    event.button.set_class(True, "active")
                    event.button.set_class(False, "inactive")
                else:

                    # Single click: toggle this level
                    new_state = not self._filters[level]
                    self._filters[level] = new_state
                    event.button.set_class(new_state, "active")
                    event.button.set_class(not new_state, "inactive")

                self._update_list_visibility()
                self.post_message(self.FilterToggled(level, self._filters[level]))

    def _update_list_visibility(self) -> None:
        """Update visibility of items based on active filters."""
        list_view = self.query_one("#responses-list", ListView)
        for item in list_view.children:
            if isinstance(item, ResponseItem):
                item.set_class(self._is_filter_active(item.level), "visible")
                item.set_class(not self._is_filter_active(item.level), "hidden")

    def add_response(self, level: str, decompiled_source: str) -> None:
        """Append a KLine response with its decompiled source to the list.

        Only adds the response if it hasn't been seen before.

        Args:
            kline: The KLine response to add.
            decompiled_source: The decompiled KScript source to display.
        """
        if decompiled_source in self._seen_responses:
            return

        self._seen_responses.add(decompiled_source)

        list_view = self.query_one("#responses-list", ListView)
        item = ResponseItem(level, decompiled_source)

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
            self.post_message(self.ResponseClicked(event.item.decompiled_source))
