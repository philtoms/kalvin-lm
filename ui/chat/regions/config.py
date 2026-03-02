"""Configuration region component."""

import time
from typing import Optional

from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Input, Label

from ui.chat.dialogs import FileLoadDialog


class ConfigRegion(Container):
    """Configuration region with model and grammar file paths."""

    DEFAULT_CSS = """
    ConfigRegion {
        height: auto;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }

    ConfigRegion .config-title {
        text-style: bold;
        margin-bottom: 1;
    }

    ConfigRegion .config-row {
        height: 3;
        margin-bottom: 1;
    }

    ConfigRegion .config-label {
        width: 10;
        height: 3;
        text-align: right;
        padding-right: 1;
        content-align: right middle;
    }

    ConfigRegion Input {
        width: 1fr;
    }

    ConfigRegion Input:hover {
        border: $accent;
    }
    """

    def __init__(
        self,
        model_path: str = "~/dev/ai/kalvin/data/kalvin.bin",
        grammar_path: str = "/Volumes/USB-Backup/ai/data/tidy-ts/simplestories-1_grammar.json",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.grammar_path = grammar_path

    def compose(self) -> ComposeResult:
        yield Label("Configuration", classes="config-title")
        with Horizontal(classes="config-row"):
            yield Label("Model:", classes="config-label")
            yield Input(value=self.model_path, placeholder="Path to model file", id="model-input")
            yield Button("Browse", id="browse-model")
        with Horizontal(classes="config-row"):
            yield Label("Grammar:", classes="config-label")
            yield Input(value=self.grammar_path, placeholder="Path to grammar file", id="grammar-input")
            yield Button("Browse", id="browse-grammar")

    def on_click(self, event: events.Click) -> None:
        """Handle click events on input fields - double-click opens file dialog."""
        if event.widget and isinstance(event.widget, Input):
            if event.widget.id in ("model-input", "grammar-input"):
                now = time.time()
                click_key = f"_last_click_{event.widget.id}"
                last_click = getattr(self, click_key, 0)

                if now - last_click < 0.5:
                    if event.widget.id == "model-input":
                        self._open_model_dialog()
                    elif event.widget.id == "grammar-input":
                        self._open_grammar_dialog()
                    setattr(self, click_key, 0)
                else:
                    setattr(self, click_key, now)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle browse button presses."""
        if event.button.id == "browse-model":
            self._open_model_dialog()
        elif event.button.id == "browse-grammar":
            self._open_grammar_dialog()

    def _open_model_dialog(self) -> None:
        """Open the model file selection dialog."""
        self.app.push_screen(
            FileLoadDialog("Select Model File", self.get_model_path()),
            lambda path: self._set_model_path(path)
        )

    def _open_grammar_dialog(self) -> None:
        """Open the grammar file selection dialog."""
        self.app.push_screen(
            FileLoadDialog("Select Grammar File", self.get_grammar_path()),
            lambda path: self._set_grammar_path(path)
        )

    def _set_model_path(self, path: Optional[str]) -> None:
        """Set the model path after file selection."""
        if path:
            self.query_one("#model-input", Input).value = path

    def _set_grammar_path(self, path: Optional[str]) -> None:
        """Set the grammar path after file selection."""
        if path:
            self.query_one("#grammar-input", Input).value = path

    def get_model_path(self) -> str:
        """Get the current model path."""
        return self.query_one("#model-input", Input).value

    def get_grammar_path(self) -> str:
        """Get the current grammar path."""
        return self.query_one("#grammar-input", Input).value
