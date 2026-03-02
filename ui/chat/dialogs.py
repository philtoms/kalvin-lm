"""File selection dialogs for the Kalvin chat application."""

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Label


class FileLoadDialog(ModalScreen):
    """Modal file selection dialog."""

    CSS = """
    FileLoadDialog {
        align: center middle;
    }

    FileLoadDialog > Container {
        width: 70;
        height: 28;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }

    FileLoadDialog .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    FileLoadDialog .mount-points {
        height: auto;
        margin-bottom: 1;
    }

    FileLoadDialog .mount-btn {
        min-width: 10;
        margin-right: 1;
    }

    FileLoadDialog DirectoryTree {
        height: 1fr;
        border: solid $panel;
        margin-bottom: 1;
    }

    FileLoadDialog .dialog-buttons {
        align: center middle;
        height: 3;
    }

    FileLoadDialog Button {
        width: auto;
        margin: 0 1;
    }

    FileLoadDialog #current-path {
        height: 3;
        margin-bottom: 1;
        border: solid $panel;
        padding: 0 1;
    }
    """

    # Common mount points
    MOUNT_POINTS = [
        ("/", "Root"),
        ("/Volumes", "Volumes"),
        ("~", "Home"),
    ]

    def __init__(self, title: str = "Select File", initial_path: str = "/") -> None:
        super().__init__()
        self.title_text = title
        self.initial_path = initial_path
        self._selected_path: Optional[Path] = None
        self._current_dir = initial_path
        self._target_file: Optional[Path] = None

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(self.title_text, classes="dialog-title")
            with Horizontal(classes="mount-points"):
                for _, label in self.MOUNT_POINTS:
                    yield Button(label, id=f"mount-{label.lower()}", classes="mount-btn")
            yield Label(self.initial_path, id="current-path")
            yield DirectoryTree(self.initial_path, id="file-tree")
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel-btn", variant="default")
                yield Button("Select", id="select-btn", variant="primary")

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
        self._selected_path = event.path
        self.query_one("#current-path", Label).update(str(event.path))

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Handle directory selection - update the path display."""
        self._current_dir = str(event.path)
        self.query_one("#current-path", Label).update(str(event.path) + "/")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "cancel-btn":
            self.dismiss(None)
        elif event.button.id == "select-btn" and self._selected_path:
            self.dismiss(str(self._selected_path))
        elif event.button.id and event.button.id.startswith("mount-"):
            for path, label in self.MOUNT_POINTS:
                if event.button.id == f"mount-{label.lower()}":
                    self._update_tree(path)
                    break
