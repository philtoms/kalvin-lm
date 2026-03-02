"""Kalvin Textual Chat Application."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Footer, Header, Input, ListView

from ui.chat.dialogs import OpenDialog, SaveDialog
from ui.chat.regions import ChatHistoryRegion, ChatRegion, ConfigRegion

DEFAULT_CHATS_DIR = Path("data/chats")


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

    .chat-container {
        layout: horizontal;
        height: 1fr;
    }

    /* Global button styles */
    Button {
        min-width: 8;
        margin-left: 1;
        margin-right: 1;
    }

    Button:focus {
        text-style: bold;
    }
    """

    TITLE = "Kalvin Chat"
    BINDINGS = [
        ("ctrl+r", "restart", "Restart"),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+l", "clear_response", "Clear"),
        ("ctrl+s", "save", "Save"),
        ("ctrl+a", "save_as", "Save As"),
        ("ctrl+o", "open", "Open"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[object] = None
        self._chat_history: list[dict] = []
        self._current_file: Optional[str] = None

    def _ensure_chats_dir(self) -> None:
        """Ensure the chats directory exists."""
        DEFAULT_CHATS_DIR.mkdir(parents=True, exist_ok=True)

    def _save_to_file(self, filename: str) -> None:
        """Save chat history to the specified file."""
        chat_region = self.query_one(ChatRegion)
        response_text = chat_region.get_response()

        # Add current exchange to history
        current_exchange = {
            "chat": chat_region.get_input(),
            "response": response_text
        }

        # Include any previous history plus current
        chats = self._chat_history.copy()
        if current_exchange["chat"] or current_exchange["response"]:
            chats.append(current_exchange)

        filepath = DEFAULT_CHATS_DIR / f"{filename}.json"
        with open(filepath, "w") as f:
            json.dump({"chats": chats}, f, indent=2)

        self._current_file = filename

    def action_save(self) -> None:
        """Save chat history - save if opened, save as if new."""
        self._ensure_chats_dir()
        if self._current_file:
            self._save_to_file(self._current_file)
        else:
            self.push_screen(
                SaveDialog(
                    title="Save Chat",
                    initial_path=str(DEFAULT_CHATS_DIR),
                    file_type=".json",
                ),
                self._handle_save,
            )

    def action_save_as(self) -> None:
        """Always open Save As dialog."""
        self._ensure_chats_dir()
        self.push_screen(
            SaveDialog(
                title="Save Chat As",
                initial_path=str(DEFAULT_CHATS_DIR),
                file_type=".json",
            ),
            self._handle_save,
        )

    def _handle_save(self, filepath: Optional[str]) -> None:
        """Handle save dialog result."""
        if not filepath:
            return
        path = Path(filepath)
        self._save_to_file(path.stem)

    def action_open(self) -> None:
        """Open chat history from file."""
        self._ensure_chats_dir()
        self.push_screen(
            OpenDialog(
                title="Open Chat",
                initial_path=str(DEFAULT_CHATS_DIR),
                file_type=".json",
            ),
            self._handle_open,
        )

    def _handle_open(self, filepath: Optional[str]) -> None:
        """Handle open dialog result."""
        if not filepath:
            return

        path = Path(filepath)
        if not path.exists():
            return

        with open(path, "r") as f:
            data = json.load(f)

        self._chat_history = data.get("chats", [])
        self._current_file = path.stem

        # Update history view and display the last chat
        self._update_history_view()
        chat_region = self.query_one(ChatRegion)
        if self._chat_history:
            last = self._chat_history[-1]
            chat_region.set_response(last.get("response", ""))

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(classes="main-container"):
            yield ConfigRegion()
            with Horizontal(classes="chat-container"):
                yield ChatRegion()
                yield ChatHistoryRegion()
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "send-btn":
            self._handle_send()
        elif event.button.id == "submit-response-btn":
            self._handle_response_submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (Enter key)."""
        if event.input.id == "chat-input":
            self._handle_send()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection in history list."""
        if event.list_view.id == "history-list":
            history_region = self.query_one(ChatHistoryRegion)
            index = history_region.get_selected_index()
            chat = history_region.get_chat_at(index)
            if chat:
                chat_region = self.query_one(ChatRegion)
                chat_region.set_response(chat.get("response", ""))

    def _update_history_view(self) -> None:
        """Update the history region with current chats."""
        history_region = self.query_one(ChatHistoryRegion)
        history_region.update_history(self._chat_history)

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

        # Track in history
        self._chat_history.append({"chat": user_input, "response": response})
        self._update_history_view()

    def _handle_response_submit(self) -> None:
        """Handle submitting the response text."""
        chat_region = self.query_one(ChatRegion)
        response_text = chat_region.get_response()
        if not response_text.strip():
            return

        # TODO: Process the submitted response (e.g., send to model, save, etc.)
        # For now, just log it
        self.log(f"Response submitted: {response_text[:50]}...")

    def action_clear_response(self) -> None:
        """Clear the response area."""
        chat_region = self.query_one(ChatRegion)
        chat_region.clear_response()

    def action_restart(self) -> None:
        """Restart the app with fresh module reload."""
        self.exit()
        os.execv(sys.executable, [sys.executable, "-m", "ui.chat"])


def main() -> None:
    """Run the Kalvin app."""
    app = KalvinApp()
    app.run()


if __name__ == "__main__":
    main()
