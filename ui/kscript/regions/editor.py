"""Editor region for KScript source editing."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import TextArea


class EditorRegion(Vertical):
    """Editor region for KScript source code."""

    DEFAULT_CSS = """
    EditorRegion {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }

    EditorRegion TextArea {
        width: 100%;
        height: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield TextArea(id="script-editor", language=None)

    def get_script(self) -> str:
        """Get the current script content.

        Returns:
            Current editor content as string.
        """
        editor = self.query_one("#script-editor", TextArea)
        return editor.text

    def set_script(self, content: str) -> None:
        """Set the editor content.

        Args:
            content: Script content to display.
        """
        editor = self.query_one("#script-editor", TextArea)
        editor.load_text(content)

    def append_to_script(self, content: str) -> None:
        """Append content to the editor.

        Args:
            content: Content to append.
        """
        editor = self.query_one("#script-editor", TextArea)
        current = editor.text
        if current and not current.endswith("\n"):
            current += "\n"
        editor.load_text(current + content)
