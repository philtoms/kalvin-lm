"""Hot reload module for KScript TUI development mode.

Provides file watching, state serialization, and graceful restart
capabilities for rapid development iteration.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

# State file paths
STATE_DIR = Path(__file__).parent.parent.parent.parent  # Project root
MODEL_STATE_FILE = STATE_DIR / ".tmp.bin"
UI_STATE_FILE = STATE_DIR / ".tmp.json"

# Watched directories
WATCHED_PATHS = [
    Path(__file__).parent.parent / "kscript",  # ui/kscript
    Path(__file__).parent.parent.parent / "src",  # src/
]


class HotReloadHandler(FileSystemEventHandler):
    """File system event handler with debouncing for hot reload triggers."""

    def __init__(self, callback: Callable[[], None], debounce_ms: int = 200):
        """Initialize the handler.

        Args:
            callback: Function to call when a .py file is modified.
            debounce_ms: Minimum time between callbacks in milliseconds.
        """
        self._callback = callback
        self._debounce_ms = debounce_ms
        self._last_trigger: float = 0

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        if not event.src_path.endswith(".py"):
            return

        # Ignore __pycache__ files
        if "__pycache__" in event.src_path:
            return

        now = time.time() * 1000
        if now - self._last_trigger > self._debounce_ms:
            self._last_trigger = now
            logger.info(f"Hot reload triggered by: {event.src_path}")
            self._callback()


def restart_app() -> None:
    """Restart the current process with the same arguments.

    Uses os.execv() to replace the current process, preserving PID
    and terminal attachment.
    """
    logger.info("Restarting application...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


class HotReloadManager:
    """Manages file watching and coordinates hot reload operations."""

    def __init__(
        self,
        on_reload: Callable[[], None],
        watched_paths: Optional[list[Path]] = None,
    ):
        """Initialize the hot reload manager.

        Args:
            on_reload: Callback to invoke when files change (typically save state + restart).
            watched_paths: Directories to watch. Defaults to WATCHED_PATHS.
        """
        self._on_reload = on_reload
        self._watched_paths = watched_paths or WATCHED_PATHS
        self._observer: Optional[Observer] = None
        self._handler: Optional[HotReloadHandler] = None

    def start(self) -> None:
        """Start watching for file changes."""
        if self._observer is not None:
            logger.warning("Hot reload already running")
            return

        self._handler = HotReloadHandler(self._on_reload)
        self._observer = Observer()

        for path in self._watched_paths:
            if path.exists():
                self._observer.schedule(self._handler, str(path), recursive=True)
                logger.info(f"Watching: {path}")
            else:
                logger.warning(f"Watch path does not exist: {path}")

        self._observer.start()
        logger.info("Hot reload manager started")

    def stop(self) -> None:
        """Stop watching for file changes."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            self._handler = None
            logger.info("Hot reload manager stopped")

    def is_active(self) -> bool:
        """Check if hot reload is currently active."""
        return self._observer is not None and self._observer.is_alive()


# === State Serialization Functions ===


def save_state(kalvin_model: Any) -> Path:
    """Save Kalvin model state to temporary file.

    Args:
        kalvin_model: Kalvin instance with save() method.

    Returns:
        Path to the saved state file.
    """
    kalvin_model.save(MODEL_STATE_FILE)
    logger.info(f"Saved model state to {MODEL_STATE_FILE}")
    return MODEL_STATE_FILE


def save_ui_state(state: dict[str, Any]) -> Path:
    """Save UI state to temporary JSON file.

    Args:
        state: Dictionary containing UI state (editor content, cursor, etc.).

    Returns:
        Path to the saved state file.
    """
    with open(UI_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved UI state to {UI_STATE_FILE}")
    return UI_STATE_FILE


def load_state() -> Optional[Any]:
    """Load Kalvin model state from temporary file.

    Returns:
        Restored Kalvin instance, or None if no state file exists.
    """
    if not MODEL_STATE_FILE.exists():
        logger.info("No model state file found, starting fresh")
        return None

    try:
        # Import here to avoid circular dependency.
        from kalvin import Kalvin

        model = Kalvin.load(MODEL_STATE_FILE)
        logger.info(f"Loaded model state from {MODEL_STATE_FILE}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load model state: {e}")
        return None


def load_ui_state() -> Optional[dict[str, Any]]:
    """Load UI state from temporary JSON file.

    Returns:
        Dictionary containing UI state, or None if no state file exists
        or if the file is corrupted.
    """
    if not UI_STATE_FILE.exists():
        logger.info("No UI state file found, starting fresh")
        return None

    try:
        with open(UI_STATE_FILE, "r") as f:
            state = json.load(f)
        logger.info(f"Loaded UI state from {UI_STATE_FILE}")
        return state
    except json.JSONDecodeError as e:
        logger.warning(f"Corrupted UI state file: {e}")
        # Remove corrupted file
        UI_STATE_FILE.unlink(missing_ok=True)
        return None
    except Exception as e:
        logger.warning(f"Failed to load UI state: {e}")
        return None


def cleanup_state_files() -> None:
    """Remove temporary state files after successful restore."""
    MODEL_STATE_FILE.unlink(missing_ok=True)
    UI_STATE_FILE.unlink(missing_ok=True)
    logger.info("Cleaned up state files")
