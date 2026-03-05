"""Kalvin Textual Chat Application with KScript integration."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Footer, Header, Input, ListView

from kalvin import Kalvin
from kscript import compile_script
from ui.chat.dialogs import OpenDialog, SaveDialog
from ui.chat.regions import ChatHistoryRegion, ChatRegion, ConfigRegion

DEFAULT_CHATS_DIR = Path("data/chats")
DEFAULT_MODEL_PATH = Path("~/dev/ai/kalvin/data/kalvin.bin").expanduser()
DEFAULT_GRAMMAR_PATH = "/Volumes/USB-Backup/ai/data/tidy-ts/simplestories-1_grammar.json"


class KalvinApp(App):
    """Kalvin Chat Application with KScript support."""

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
        ("ctrl+l", "clear_script", "Clear Script"),
        ("ctrl+s", "save", "Save"),
        ("ctrl+a", "save_as", "Save As"),
        ("ctrl+o", "open", "Open"),
        ("ctrl+e", "run_script", "Run Script"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._kalvin: Optional[Kalvin] = None
        self._model_error: Optional[str] = None
        self._history: list[dict] = []
        self._current_file: Optional[str] = None
        self._current_file_type: str = "json"  # "json" or "ks"

    def _ensure_chats_dir(self) -> None:
        """Ensure the chats directory exists."""
        DEFAULT_CHATS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_dictionary(self, grammar_path: str) -> dict | None:
        """Load dictionary from grammar file."""
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

    def _save_to_file(self, filename: str, file_type: str = "json") -> None:
        """Save to the specified file."""
        chat_region = self.query_one(ChatRegion)
        script = chat_region.get_script()

        if file_type == "ks":
            # Save as KScript file
            filepath = DEFAULT_CHATS_DIR / f"{filename}.ks"
            with open(filepath, "w") as f:
                f.write(script if script else "")
        else:
            # Save as JSON chat file (includes script and history)
            data = {
                "script": script,
                "history": self._history,
            }
            filepath = DEFAULT_CHATS_DIR / f"{filename}.json"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        self._current_file = filename
        self._current_file_type = file_type

    def action_save(self) -> None:
        """Save - save if opened, save as if new."""
        self._ensure_chats_dir()
        if self._current_file:
            self._save_to_file(self._current_file, self._current_file_type)
        else:
            self.push_screen(
                SaveDialog(
                    title="Save",
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
                title="Save As",
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
        file_type = "ks" if path.suffix == ".ks" else "json"
        self._save_to_file(path.stem, file_type)

    def action_open(self) -> None:
        """Open file."""
        self._ensure_chats_dir()
        self.push_screen(
            OpenDialog(
                title="Open",
                initial_path=str(DEFAULT_CHATS_DIR),
                file_type=None,  # Allow all files
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

        if path.suffix == ".ks":
            # Open as KScript file
            with open(path, "r") as f:
                script = f.read()
            chat_region = self.query_one(ChatRegion)
            chat_region.set_script(script)
            self._current_file_type = "ks"
        else:
            # Open as JSON chat file
            with open(path, "r") as f:
                data = json.load(f)

            # Load script if present
            chat_region = self.query_one(ChatRegion)
            if "script" in data:
                chat_region.set_script(data["script"])

            # Load history if present
            self._history = data.get("history", data.get("chats", []))
            self._current_file_type = "json"

        self._current_file = path.stem
        self._update_history_view()

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
        elif event.button.id == "run-script-btn":
            self._handle_run_script()
        elif event.button.id == "clear-script-btn":
            self.action_clear_script()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (Enter key)."""
        if event.input.id == "chat-input":
            self._handle_send()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection in history list."""
        if event.list_view.id == "history-list":
            history_region = self.query_one(ChatHistoryRegion)
            index = history_region.get_selected_index()
            entry = history_region.get_entry_at(index)
            if entry:
                # If it's a script entry, load the script into the editor
                if entry.get("type") == "script":
                    chat_region = self.query_one(ChatRegion)
                    chat_region.set_script(entry.get("script", ""))

    def _update_history_view(self) -> None:
        """Update the history region with current entries."""
        history_region = self.query_one(ChatHistoryRegion)
        history_region.update_history(self._history)

    def _handle_send(self) -> None:
        """Handle sending a direct chat message."""
        chat_region = self.query_one(ChatRegion)

        user_input = chat_region.get_input()
        if not user_input.strip():
            return

        chat_region.clear_input()

        # Generate response using loaded model
        if self._kalvin:
            try:
                result = self._kalvin.encode(user_input)
                if result:
                    signature = result.signature
                    if signature:
                        response = self._kalvin.decode(signature)
                        if not response:
                            response = f"[Encoded: signature={signature}]"
                    else:
                        response = "[No signature in result]"
                else:
                    response = "[No result from encode]"
            except Exception as e:
                response = f"[Error: {e}]"
        elif self._model_error:
            response = f"[Model Error: {self._model_error}]"
        else:
            response = "[Model not loaded]"

        # Add to history
        entry = {
            "type": "chat",
            "chat": user_input,
            "response": response,
        }
        self._history.append(entry)
        self._update_history_view()

    def _handle_run_script(self) -> None:
        """Handle running the KScript."""
        chat_region = self.query_one(ChatRegion)
        script = chat_region.get_script()

        if not script.strip():
            return

        # Compile the script using Kalvin agent
        try:
            result = compile_script(script, agent=self._kalvin)

            # Build output summary
            output_lines = [
                f"Compiled {len(result.model)} KLines",
                f"Symbol table: {len(result.symbol_table)} entries",
            ]

            if result.load_paths:
                output_lines.append(f"Load paths: {result.load_paths}")
            if result.save_path:
                output_lines.append(f"Save path: {result.save_path}")
            if result.attention_klines:
                output_lines.append(f"Attention KLines: {len(result.attention_klines)}")

            # Show some symbol table entries
            if result.symbol_table:
                symbols = list(result.symbol_table.keys())[:5]
                output_lines.append("Symbols: " + ", ".join(symbols))
                if len(result.symbol_table) > 5:
                    output_lines.append(f"  ... and {len(result.symbol_table) - 5} more")

            output = "\n".join(output_lines)

            # KLines are already integrated via the Kalvin agent
            if self._kalvin:
                output += f"\n\nModel now has {len(self._kalvin.model)} KLines"

        except Exception as e:
            output = f"Compilation error:\n{e}"

        # Add to history
        entry = {
            "type": "script",
            "script": script,
            "output": output,
        }
        self._history.append(entry)
        self._update_history_view()

    def action_clear_script(self) -> None:
        """Clear the script area."""
        chat_region = self.query_one(ChatRegion)
        chat_region.clear_script()

    def action_run_script(self) -> None:
        """Run the current script."""
        self._handle_run_script()

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
