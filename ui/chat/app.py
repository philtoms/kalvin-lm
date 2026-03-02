"""Kalvin Textual Chat Application."""

from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Footer, Header, Input

from ui.chat.regions import ChatRegion, ConfigRegion


class KalvinApp(App):
    """Kalvin Chat Application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    .main-container {
        layout: vertical;
        height: 1fr;
        padding: 1;
    }
    """

    TITLE = "Kalvin Chat"
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_response", "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[object] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(classes="main-container"):
            yield ConfigRegion()
            yield ChatRegion()
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "send-btn":
            self._handle_send()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (Enter key)."""
        if event.input.id == "chat-input":
            self._handle_send()

    def _handle_send(self) -> None:
        """Handle sending a message."""
        chat_region = self.query_one(ChatRegion)
        config_region = self.query_one(ConfigRegion)

        user_input = chat_region.get_input()
        if not user_input.strip():
            return

        model_path = config_region.get_model_path()
        grammar_path = config_region.get_grammar_path()

        chat_region.append_response(f"> {user_input}")
        chat_region.clear_input()

        # TODO: Load model and generate response
        response = f"[Model: {Path(model_path).name}]\nProcessing: {user_input}"
        chat_region.append_response(response)

    def action_clear_response(self) -> None:
        """Clear the response area."""
        chat_region = self.query_one(ChatRegion)
        chat_region.set_response("")


def main() -> None:
    """Run the Kalvin app."""
    app = KalvinApp()
    app.run()


if __name__ == "__main__":
    main()
