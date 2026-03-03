"""Kalvin Textual Chat Application."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Footer, Header, Input, ListView

from kalvin import Kalvin
from ui.chat.dialogs import OpenDialog, SaveDialog
from ui.chat.regions import ChatHistoryRegion, ChatRegion, ConfigRegion

DEFAULT_CHATS_DIR = Path("data/chats")
DEFAULT_MODEL_PATH = Path("~/dev/ai/kalvin/data/kalvin.bin").expanduser()
DEFAULT_GRAMMAR_PATH = "/Volumes/USB-Backup/ai/data/tidy-ts/simplestories-1_grammar.json"


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
        self._kalvin: Optional[Kalvin] = None
        self._model_error: Optional[str] = None
        self._chat_history: list[dict] = []
        self._current_file: Optional[str] = None

    def _ensure_chats_dir(self) -> None:
        """Ensure the chats directory exists."""
        DEFAULT_CHATS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_dictionary(self, grammar_path: str) -> dict | None:
        """Load dictionary from grammar file."""
        import json
        try:
            grammar_path_expanded = Path(grammar_path).expanduser()
            if not grammar_path_expanded.exists():
                self.log(f"Grammar file not found: {grammar_path}")
                return None
            with open(grammar_path_expanded, "r") as f:
                str_dict = json.load(f)
            dictionary = {}
            for key, value in str_dict.items():
                dictionary[int(key)] = value
            return dictionary
        except Exception as e:
            self.log(f"Failed to load dictionary: {e}")
            return None

    def _load_nlp_type(self, grammar_path: str, nlp_detail: str = "nlp_type32") -> dict | None:
        """Load NLP type mapping from grammar file."""
        import json
        try:
            nlp_path = grammar_path.replace("grammar", nlp_detail)
            nlp_path_expanded = Path(nlp_path).expanduser()
            if not nlp_path_expanded.exists():
                self.log(f"NLP type file not found: {nlp_path}")
                return None
            with open(nlp_path_expanded, "r") as f:
                return json.load(f)
        except Exception as e:
            self.log(f"Failed to load NLP type: {e}")
            return None

    def _load_model(self, model_path: str, grammar_path: str) -> None:
        """Load or reload the Kalvin model with the given paths."""
        self._kalvin = None
        self._model_error = None

        try:
            model_path_expanded = Path(model_path).expanduser()
            if not model_path_expanded.exists():
                self._model_error = f"Model file not found: {model_path}"
                self.log(self._model_error)
                return

            # Load model from file
            self._kalvin = Kalvin.load(model_path_expanded)

            # Update dictionary and nlp_type from grammar path
            dictionary = self._load_dictionary(grammar_path)
            if dictionary:
                self._kalvin.dictionary = dictionary

            nlp_type = self._load_nlp_type(grammar_path)
            if nlp_type:
                self._kalvin.nlp_type = nlp_type

            self.log(f"Model loaded: {model_path}")
        except Exception as e:
            self._model_error = f"Failed to load model: {e}"
            self.log(self._model_error)

    def _on_config_change(self, model_path: str, grammar_path: str) -> None:
        """Handle config changes - reload model."""
        self._load_model(model_path, grammar_path)

    def on_mount(self) -> None:
        """Load model on app start."""
        self._load_model(str(DEFAULT_MODEL_PATH), DEFAULT_GRAMMAR_PATH)

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
            yield ConfigRegion(
                str(DEFAULT_MODEL_PATH),
                DEFAULT_GRAMMAR_PATH,
                on_config_change=self._on_config_change,
            )
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

        user_input = chat_region.get_input()
        if not user_input.strip():
            return

        chat_region.append_response(f"> {user_input}")
        chat_region.clear_input()

        # Generate response using loaded model
        if self._kalvin:
            try:
                result = self._kalvin.encode(user_input)
                if result:
                    # result is a KLine with s_key attribute
                    s_key = result.s_key
                    if s_key:
                        response = self._kalvin.decode(s_key)
                        if not response:
                            response = f"[Encoded: s_key={s_key}]"
                    else:
                        response = "[No s_key in result]"
                else:
                    response = "[No result from encode]"
            except Exception as e:
                response = f"[Error: {e}]"
        elif self._model_error:
            response = f"[Model Error: {self._model_error}]"
        else:
            response = "[Model not loaded]"

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
