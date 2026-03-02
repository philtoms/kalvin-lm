"""Dialogs for the Kalvin chat application."""

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Label


class FileDialog(ModalScreen):
    """Base class for file dialogs with folder navigation."""

    CSS = """
    FileDialog, OpenDialog, SaveDialog {
        align: center middle;
    }

    FileDialog > Container, OpenDialog > Container, SaveDialog > Container {
        width: 70;
        height: 28;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }

    FileDialog .dialog-title, OpenDialog .dialog-title, SaveDialog .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    FileDialog .mount-points, OpenDialog .mount-points, SaveDialog .mount-points {
        height: auto;
        margin-bottom: 1;
    }

    FileDialog DirectoryTree, OpenDialog DirectoryTree, SaveDialog DirectoryTree {
        height: 1fr;
        border: solid $panel;
        margin-bottom: 1;
    }

    FileDialog .dialog-buttons, OpenDialog .dialog-buttons, SaveDialog .dialog-buttons {
        align: center middle;
        height: 3;
    }

    FileDialog #current-path, OpenDialog #current-path, SaveDialog #current-path {
        height: 3;
        margin-bottom: 1;
        border: solid $panel;
        padding: 0 1;
    }

    FileDialog .filename-row, SaveDialog .filename-row {
        height: auto;
        margin-bottom: 1;
    }

    FileDialog .filename-input, SaveDialog .filename-input {
        width: 1fr;
    }
    """

    # Common mount points
    MOUNT_POINTS = [
        ("/", "Root"),
        ("/Volumes", "Volumes"),
        ("~", "Home"),
    ]

    def __init__(
        self,
        title: str = "Select File",
        initial_path: str = "/",
        file_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.title_text = title
        self.initial_path = initial_path
        self.file_type = file_type
        self._selected_path: Optional[Path] = None
        self._current_dir = initial_path
        self._target_file: Optional[Path] = None

    def _compose_header(self) -> ComposeResult:
        """Compose the common header elements."""
        yield Label(self.title_text, classes="dialog-title")
        with Horizontal(classes="mount-points"):
            for _, label in self.MOUNT_POINTS:
                yield Button(label, id=f"mount-{label.lower()}")

    def _compose_tree(self) -> ComposeResult:
        """Compose the directory tree and path display."""
        yield Label(self.initial_path, id="current-path")
        yield DirectoryTree(self.initial_path, id="file-tree")

    def _compose_footer(self, confirm_label: str = "Select") -> ComposeResult:
        """Compose the dialog footer with buttons."""
        with Horizontal(classes="dialog-buttons"):
            yield Button("Cancel", id="cancel-btn")
            yield Button(confirm_label, id="confirm-btn", variant="primary")

    def on_mount(self) -> None:
        """Set up initial state."""
        self._update_tree(self.initial_path, focus_file=True)

    def _update_tree(self, path: str, focus_file: bool = False) -> None:
        """Update the directory tree to show a new path."""
        expanded_path = Path(path).expanduser()
        self._target_file = None

        if expanded_path.exists():
            if expanded_path.is_file():
                self._target_file = expanded_path if focus_file else None
                self._current_dir = str(expanded_path.parent)
            else:
                self._current_dir = str(expanded_path)
        else:
            self._current_dir = "/"

        tree = self.query_one("#file-tree", DirectoryTree)
        tree.path = Path(self._current_dir)
        self.query_one("#current-path", Label).update(self._current_dir)

        if self._target_file:
            self.call_after_refresh(self._focus_target_file)

    def _focus_target_file(self) -> None:
        """Focus and select the target file in the tree."""
        if not self._target_file:
            return

        tree = self.query_one("#file-tree", DirectoryTree)

        def find_and_select_node(node) -> bool:
            """Recursively find and select the target file node."""
            if hasattr(node, 'data') and node.data and hasattr(node.data, 'path'):
                if node.data.path == self._target_file:
                    tree.select_node(node)
                    self._selected_path = self._target_file
                    tree.scroll_to_node(node, animate=False)
                    self.query_one("#current-path", Label).update(str(self._target_file))
                    return True

            if node.is_expanded:
                for child in node.children:
                    if find_and_select_node(child):
                        return True
            return False

        if tree.root and not find_and_select_node(tree.root):
            tree.root.expand()
            self.call_after_refresh(lambda: find_and_select_node(tree.root) if tree.root else None)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection in the directory tree."""
        # Filter by file type if specified
        if self.file_type and not str(event.path).endswith(self.file_type):
            return
        self._selected_path = event.path
        self.query_one("#current-path", Label).update(str(event.path))

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Handle directory selection - update the path display."""
        self._current_dir = str(event.path)
        self.query_one("#current-path", Label).update(str(event.path) + "/")

    def _handle_mount_button(self, button_id: str) -> bool:
        """Handle mount button press. Returns True if handled."""
        if button_id and button_id.startswith("mount-"):
            for path, label in self.MOUNT_POINTS:
                if button_id == f"mount-{label.lower()}":
                    self._update_tree(path)
                    return True
        return False


class OpenDialog(FileDialog):
    """Modal dialog for opening files."""

    def compose(self) -> ComposeResult:
        with Container():
            yield from self._compose_header()
            yield from self._compose_tree()
            yield from self._compose_footer("Open")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "cancel-btn":
            self.dismiss(None)
        elif event.button.id == "confirm-btn" and self._selected_path:
            self.dismiss(str(self._selected_path))
        else:
            self._handle_mount_button(event.button.id or "")


class SaveDialog(FileDialog):
    """Modal dialog for saving files with filename input."""

    def __init__(
        self,
        title: str = "Save File",
        initial_path: str = "/",
        file_type: Optional[str] = None,
    ) -> None:
        super().__init__(title, initial_path, file_type)

    def compose(self) -> ComposeResult:
        with Container():
            yield from self._compose_header()
            with Horizontal(classes="filename-row"):
                yield Input(placeholder="filename", id="filename-input", classes="filename-input")
            yield from self._compose_tree()
            yield from self._compose_footer("Save")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection - populate filename input."""
        super().on_directory_tree_file_selected(event)
        if self._selected_path:
            self.query_one("#filename-input", Input).value = self._selected_path.name

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Handle directory selection - update current dir for save target."""
        super().on_directory_tree_directory_selected(event)
        self._selected_path = None  # Clear selection when navigating directories

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "cancel-btn":
            self.dismiss(None)
        elif event.button.id == "confirm-btn":
            filename = self.query_one("#filename-input", Input).value.strip()
            if filename:
                # Add file extension if not present and file_type specified
                if self.file_type and not filename.endswith(self.file_type):
                    filename += self.file_type
                save_path = Path(self._current_dir) / filename
                self.dismiss(str(save_path))
        else:
            self._handle_mount_button(event.button.id or "")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle filename input submission."""
        if event.input.id == "filename-input":
            filename = event.input.value.strip()
            if filename:
                if self.file_type and not filename.endswith(self.file_type):
                    filename += self.file_type
                save_path = Path(self._current_dir) / filename
                self.dismiss(str(save_path))
