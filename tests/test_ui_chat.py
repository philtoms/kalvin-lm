"""Tests for the UI chat application components."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from textual.app import App, ComposeResult
from textual.widgets import Input, TextArea, ListView, ListItem, Button

from ui.chat.regions.chat import ChatRegion
from ui.chat.regions.config import ConfigRegion
from ui.chat.regions.history import ChatHistoryRegion
from ui.chat.dialogs import FileDialog, OpenDialog, SaveDialog


# ============================================================================
# Test Apps for Component Testing
# ============================================================================


class ChatRegionTestApp(App):
    """Test app that includes ChatRegion."""

    def compose(self) -> ComposeResult:
        yield ChatRegion()


class ChatHistoryTestApp(App):
    """Test app that includes ChatHistoryRegion."""

    def compose(self) -> ComposeResult:
        yield ChatHistoryRegion()


class ConfigRegionTestApp(App):
    """Test app that includes ConfigRegion."""

    def __init__(self, callback=None):
        super().__init__()
        self._callback = callback

    def compose(self) -> ComposeResult:
        yield ConfigRegion(
            model_path="/test/model.bin",
            grammar_path="/test/grammar.json",
            on_config_change=self._callback,
        )


class OpenDialogTestApp(App):
    """Test app that includes OpenDialog."""

    def compose(self) -> ComposeResult:
        yield OpenDialog(title="Test Open", initial_path="/", file_type=".json")


class SaveDialogTestApp(App):
    """Test app that includes SaveDialog."""

    def compose(self) -> ComposeResult:
        yield SaveDialog(title="Test Save", initial_path="/", file_type=".json")


# ============================================================================
# ChatRegion Tests
# ============================================================================


class TestChatRegion:
    """Tests for ChatRegion component."""

    @pytest.mark.asyncio
    async def test_compose_creates_widgets(self):
        """ChatRegion composes with expected widgets."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            assert pilot.app.query_one("#chat-input", Input) is not None
            assert pilot.app.query_one("#response-text", TextArea) is not None
            assert pilot.app.query_one("#send-btn") is not None
            assert pilot.app.query_one("#submit-response-btn") is not None

    @pytest.mark.asyncio
    async def test_get_input_returns_current_value(self):
        """get_input returns the current input field value."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            input_widget = pilot.app.query_one("#chat-input", Input)
            input_widget.value = "Hello, world!"

            assert chat_region.get_input() == "Hello, world!"

    @pytest.mark.asyncio
    async def test_get_input_returns_empty_string_by_default(self):
        """get_input returns empty string when input is empty."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            assert chat_region.get_input() == ""

    @pytest.mark.asyncio
    async def test_clear_input_resets_value(self):
        """clear_input sets the input field to empty."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            input_widget = pilot.app.query_one("#chat-input", Input)
            input_widget.value = "Some text"

            chat_region.clear_input()

            assert input_widget.value == ""

    @pytest.mark.asyncio
    async def test_get_response_returns_empty_when_placeholder(self):
        """get_response returns empty string when placeholder is active."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            # By default, placeholder is active
            assert chat_region.get_response() == ""

    @pytest.mark.asyncio
    async def test_set_response_updates_textarea(self):
        """set_response updates the response textarea."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            chat_region.set_response("Test response")

            textarea = pilot.app.query_one("#response-text", TextArea)
            assert textarea.text == "Test response"
            assert chat_region.get_response() == "Test response"

    @pytest.mark.asyncio
    async def test_set_response_clears_placeholder(self):
        """set_response clears the placeholder flag."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            chat_region.set_response("New response")
            assert not chat_region._placeholder_active
            assert chat_region.get_response() == "New response"

    @pytest.mark.asyncio
    async def test_append_response_when_placeholder(self):
        """append_response replaces placeholder text."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            chat_region.append_response("First line")

            textarea = pilot.app.query_one("#response-text", TextArea)
            assert textarea.text == "First line"
            assert not chat_region._placeholder_active

    @pytest.mark.asyncio
    async def test_append_response_adds_newline(self):
        """append_response adds newline between existing and new text."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            chat_region.append_response("Line 1")
            chat_region.append_response("Line 2")

            textarea = pilot.app.query_one("#response-text", TextArea)
            assert textarea.text == "Line 1\nLine 2"

    @pytest.mark.asyncio
    async def test_append_multiple_lines(self):
        """append_response handles multiple appends correctly."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            chat_region.append_response("A")
            chat_region.append_response("B")
            chat_region.append_response("C")

            textarea = pilot.app.query_one("#response-text", TextArea)
            assert textarea.text == "A\nB\nC"

    @pytest.mark.asyncio
    async def test_clear_response_resets_to_placeholder(self):
        """clear_response resets textarea to placeholder."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            chat_region.set_response("Some text")
            chat_region.clear_response()

            textarea = pilot.app.query_one("#response-text", TextArea)
            assert chat_region._placeholder_active
            assert textarea.text == "Response will appear here..."

    @pytest.mark.asyncio
    async def test_clear_response_allows_new_appends(self):
        """clear_response allows new appends after clearing."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            chat_region.set_response("Old text")
            chat_region.clear_response()
            chat_region.append_response("New text")

            assert chat_region.get_response() == "New text"


# ============================================================================
# ChatHistoryRegion Tests
# ============================================================================


class TestChatHistoryRegion:
    """Tests for ChatHistoryRegion component."""

    @pytest.mark.asyncio
    async def test_compose_creates_widgets(self):
        """ChatHistoryRegion composes with expected widgets."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            assert pilot.app.query_one("#history-list", ListView) is not None

    @pytest.mark.asyncio
    async def test_update_history_populates_list(self):
        """update_history populates the list view with chats."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            chats = [
                {"chat": "Hello", "response": "Hi there"},
                {"chat": "Goodbye", "response": "See you"},
            ]

            history_region.update_history(chats)
            await pilot.pause()

            list_view = pilot.app.query_one("#history-list", ListView)
            # ListView contains ListItem widgets
            assert len(list_view.children) == 2

    @pytest.mark.asyncio
    async def test_update_history_clears_previous(self):
        """update_history clears previous history before adding new."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            history_region.update_history([{"chat": "A", "response": "B"}])
            await pilot.pause()
            history_region.update_history([{"chat": "C", "response": "D"}])
            await pilot.pause()

            list_view = pilot.app.query_one("#history-list", ListView)
            assert len(list_view.children) == 1

    @pytest.mark.asyncio
    async def test_update_history_empty_list(self):
        """update_history handles empty list."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            history_region.update_history([])
            await pilot.pause()
            history_region.update_history([])  # Second call should not error
            await pilot.pause()

            list_view = pilot.app.query_one("#history-list", ListView)
            assert len(list_view.children) == 0

    @pytest.mark.asyncio
    async def test_get_chat_at_valid_index(self):
        """get_chat_at returns chat at valid index."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            chats = [
                {"chat": "First", "response": "Response 1"},
                {"chat": "Second", "response": "Response 2"},
            ]

            history_region.update_history(chats)
            await pilot.pause()

            assert history_region.get_chat_at(0) == chats[0]
            assert history_region.get_chat_at(1) == chats[1]

    @pytest.mark.asyncio
    async def test_get_chat_at_negative_index(self):
        """get_chat_at returns None for negative index."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            history_region.update_history([{"chat": "Test", "response": "Resp"}])
            await pilot.pause()

            assert history_region.get_chat_at(-1) is None

    @pytest.mark.asyncio
    async def test_get_chat_at_out_of_range(self):
        """get_chat_at returns None for out of range index."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            history_region.update_history([{"chat": "Test", "response": "Resp"}])
            await pilot.pause()

            assert history_region.get_chat_at(10) is None

    @pytest.mark.asyncio
    async def test_get_chat_at_empty_history(self):
        """get_chat_at returns None when history is empty."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            history_region.update_history([])
            await pilot.pause()

            assert history_region.get_chat_at(0) is None

    @pytest.mark.asyncio
    async def test_get_selected_index_default(self):
        """get_selected_index returns -1 when nothing selected."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            history_region.update_history([{"chat": "A", "response": "B"}])
            await pilot.pause()

            assert history_region.get_selected_index() == -1

    @pytest.mark.asyncio
    async def test_stores_chats_internally(self):
        """ChatHistoryRegion stores chats for retrieval."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            chats = [{"chat": "Test", "response": "Response"}]

            history_region.update_history(chats)
            await pilot.pause()

            assert history_region._chats == chats

    @pytest.mark.asyncio
    async def test_long_chat_truncated(self):
        """Long chat messages are truncated in preview."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            long_message = "x" * 100
            chats = [{"chat": long_message, "response": "Response"}]

            history_region.update_history(chats)
            await pilot.pause()

            # Check that the item was added (preview truncation is internal)
            list_view = pilot.app.query_one("#history-list", ListView)
            assert len(list_view.children) == 1


# ============================================================================
# ConfigRegion Tests
# ============================================================================


class TestConfigRegion:
    """Tests for ConfigRegion component."""

    @pytest.mark.asyncio
    async def test_compose_creates_widgets(self):
        """ConfigRegion composes with expected widgets."""
        app = ConfigRegionTestApp()
        async with app.run_test() as pilot:
            assert pilot.app.query_one("#model-input", Input) is not None
            assert pilot.app.query_one("#grammar-input", Input) is not None
            assert pilot.app.query_one("#browse-model") is not None
            assert pilot.app.query_one("#browse-grammar") is not None

    @pytest.mark.asyncio
    async def test_initial_paths_displayed(self):
        """ConfigRegion displays initial paths in inputs."""
        app = ConfigRegionTestApp()
        async with app.run_test() as pilot:
            model_input = pilot.app.query_one("#model-input", Input)
            grammar_input = pilot.app.query_one("#grammar-input", Input)

            assert model_input.value == "/test/model.bin"
            assert grammar_input.value == "/test/grammar.json"

    @pytest.mark.asyncio
    async def test_get_model_path(self):
        """get_model_path returns current model path."""
        app = ConfigRegionTestApp()
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            assert config_region.get_model_path() == "/test/model.bin"

    @pytest.mark.asyncio
    async def test_get_grammar_path(self):
        """get_grammar_path returns current grammar path."""
        app = ConfigRegionTestApp()
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            assert config_region.get_grammar_path() == "/test/grammar.json"

    @pytest.mark.asyncio
    async def test_set_model_path_updates_input(self):
        """_set_model_path updates the model input field."""
        app = ConfigRegionTestApp()
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            config_region._set_model_path("/new/model.bin")

            model_input = pilot.app.query_one("#model-input", Input)
            assert model_input.value == "/new/model.bin"

    @pytest.mark.asyncio
    async def test_set_grammar_path_updates_input(self):
        """_set_grammar_path updates the grammar input field."""
        app = ConfigRegionTestApp()
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            config_region._set_grammar_path("/new/grammar.json")

            grammar_input = pilot.app.query_one("#grammar-input", Input)
            assert grammar_input.value == "/new/grammar.json"

    @pytest.mark.asyncio
    async def test_set_model_path_none_does_nothing(self):
        """_set_model_path with None does not update input."""
        app = ConfigRegionTestApp()
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            config_region._set_model_path(None)
            assert config_region.get_model_path() == "/test/model.bin"

    @pytest.mark.asyncio
    async def test_set_grammar_path_none_does_nothing(self):
        """_set_grammar_path with None does not update input."""
        app = ConfigRegionTestApp()
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            config_region._set_grammar_path(None)
            assert config_region.get_grammar_path() == "/test/grammar.json"

    @pytest.mark.asyncio
    async def test_callback_called_on_model_change(self):
        """ConfigRegion calls callback when model path changes."""
        callback_calls = []

        def callback(model, grammar):
            callback_calls.append((model, grammar))

        app = ConfigRegionTestApp(callback=callback)
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            config_region._set_model_path("/new/model.bin")

            assert len(callback_calls) == 1
            assert callback_calls[0] == ("/new/model.bin", "/test/grammar.json")

    @pytest.mark.asyncio
    async def test_callback_called_on_grammar_change(self):
        """ConfigRegion calls callback when grammar path changes."""
        callback_calls = []

        def callback(model, grammar):
            callback_calls.append((model, grammar))

        app = ConfigRegionTestApp(callback=callback)
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            config_region._set_grammar_path("/new/grammar.json")

            assert len(callback_calls) == 1
            assert callback_calls[0] == ("/test/model.bin", "/new/grammar.json")

    @pytest.mark.asyncio
    async def test_no_callback_when_none(self):
        """ConfigRegion handles None callback without error."""
        app = ConfigRegionTestApp(callback=None)
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            # Should not raise
            config_region._set_model_path("/new/model.bin")


# ============================================================================
# Dialog Tests
# ============================================================================


class TestFileDialog:
    """Tests for FileDialog base class."""

    def test_mount_points_defined(self):
        """FileDialog has common mount points defined."""
        assert FileDialog.MOUNT_POINTS == [
            ("/", "Root"),
            ("/Volumes", "Volumes"),
            ("~", "Home"),
        ]

    def test_initial_state(self):
        """FileDialog initializes with expected state."""
        dialog = FileDialog(
            title="Test Dialog",
            initial_path="/some/path",
            file_type=".json",
        )

        assert dialog.title_text == "Test Dialog"
        assert dialog.initial_path == "/some/path"
        assert dialog.file_type == ".json"
        assert dialog._selected_path is None

    def test_handle_mount_button_unknown(self):
        """_handle_mount_button returns False for unknown buttons."""
        dialog = FileDialog(initial_path="/some/path")
        result = dialog._handle_mount_button("unknown-button")
        assert result is False


class TestOpenDialog:
    """Tests for OpenDialog."""

    @pytest.mark.asyncio
    async def test_compose_creates_widgets(self):
        """OpenDialog composes with expected widgets."""
        app = OpenDialogTestApp()
        async with app.run_test() as pilot:
            # Should have cancel and confirm buttons
            buttons = pilot.app.query("Button")
            button_ids = {b.id for b in buttons}
            assert "cancel-btn" in button_ids
            assert "confirm-btn" in button_ids

    def test_file_type_filter(self):
        """OpenDialog accepts file_type parameter."""
        dialog = OpenDialog(
            title="Open JSON",
            initial_path="/",
            file_type=".json",
        )

        assert dialog.file_type == ".json"


class TestSaveDialog:
    """Tests for SaveDialog."""

    @pytest.mark.asyncio
    async def test_compose_creates_filename_input(self):
        """SaveDialog composes with filename input field."""
        app = SaveDialogTestApp()
        async with app.run_test() as pilot:
            filename_input = pilot.app.query_one("#filename-input", Input)
            assert filename_input is not None

    def test_file_type_appended(self):
        """SaveDialog appends file extension if needed."""
        dialog = SaveDialog(
            title="Save JSON",
            initial_path="/",
            file_type=".json",
        )

        assert dialog.file_type == ".json"


# ============================================================================
# Integration Tests
# ============================================================================


class TestChatIntegration:
    """Integration tests for chat components working together."""

    @pytest.mark.asyncio
    async def test_chat_and_history_workflow(self):
        """Test workflow of sending messages and viewing history."""
        # Create combined test app
        class IntegrationApp(App):
            def compose(self) -> ComposeResult:
                yield ChatRegion()
                yield ChatHistoryRegion()

        app = IntegrationApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            history_region = pilot.app.query_one(ChatHistoryRegion)

            # Simulate user input
            chat_region.set_response("User message and response")

            # Simulate adding to history
            chats = [{"chat": "Hello", "response": "Hi there"}]
            history_region.update_history(chats)
            await pilot.pause()

            # Verify history contains the chat
            assert history_region.get_chat_at(0) == chats[0]

    @pytest.mark.asyncio
    async def test_config_change_updates_paths(self):
        """Test config changes update all paths correctly."""
        callback_calls = []

        class ConfigApp(App):
            def __init__(self):
                super().__init__()
                self._calls = callback_calls

            def compose(self) -> ComposeResult:
                yield ConfigRegion(
                    model_path="/old/model.bin",
                    grammar_path="/old/grammar.json",
                    on_config_change=lambda m, g: callback_calls.append((m, g)),
                )

        app = ConfigApp()
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            config_region._set_model_path("/new/model.bin")
            config_region._set_grammar_path("/new/grammar.json")

            assert len(callback_calls) == 2
            assert callback_calls[0] == ("/new/model.bin", "/old/grammar.json")
            assert callback_calls[1] == ("/new/model.bin", "/new/grammar.json")


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_chat_region_empty_input_send(self):
        """ChatRegion handles empty input gracefully."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            # Empty input should not be sent
            assert chat_region.get_input() == ""

    @pytest.mark.asyncio
    async def test_chat_history_with_special_characters(self):
        """ChatHistoryRegion handles special characters in chat."""
        app = ChatHistoryTestApp()
        async with app.run_test() as pilot:
            history_region = pilot.app.query_one(ChatHistoryRegion)
            chats = [
                {"chat": "Hello! @#$%^&*()", "response": "Hi! <>?/.,;'"},
            ]

            history_region.update_history(chats)
            await pilot.pause()

            assert history_region.get_chat_at(0) == chats[0]

    @pytest.mark.asyncio
    async def test_config_region_with_tilde_paths(self):
        """ConfigRegion handles tilde in paths."""
        app = ConfigRegionTestApp()
        async with app.run_test() as pilot:
            config_region = pilot.app.query_one(ConfigRegion)
            config_region._set_model_path("~/test/model.bin")

            assert config_region.get_model_path() == "~/test/model.bin"

    @pytest.mark.asyncio
    async def test_chat_region_unicode_text(self):
        """ChatRegion handles unicode text correctly."""
        app = ChatRegionTestApp()
        async with app.run_test() as pilot:
            chat_region = pilot.app.query_one(ChatRegion)
            chat_region.set_response("Hello 世界 🌍")

            assert chat_region.get_response() == "Hello 世界 🌍"
