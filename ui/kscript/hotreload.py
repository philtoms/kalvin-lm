"""Process harness with auto-restart and source file watching.

Runs a child process (default: the KScript TUI), passing through all args.
Restarts on file changes or if the child exits unexpectedly.

Usage:
    python -m ui.kscript.hotreload [any args forwarded to child]
"""

import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent  # project root
WATCH_DIRS = [_ROOT / "src", _ROOT / "ui"]
DEBOUNCE_MS = 300
SHUTDOWN_TIMEOUT = 5.0
RESTART_DELAY = 0.5


class _Handler(FileSystemEventHandler):
    def __init__(self, callback, debounce_ms: int = DEBOUNCE_MS):
        self._cb = callback
        self._ms = debounce_ms
        self._last = 0.0

    def on_modified(self, event: FileModifiedEvent) -> None:
        if event.is_directory or not event.src_path.endswith(".py"):
            return
        if "__pycache__" in event.src_path:
            return
        now = time.time() * 1000
        if now - self._last > self._ms:
            self._last = now
            logger.info("changed: %s", event.src_path)
            self._cb()


class Harness:
    def __init__(self, argv: list[str]):
        self._cmd = argv
        self._proc = None
        self._restart_requested = False
        self._observer = Observer()
        self._handler = _Handler(self._request_restart)

    # --- lifecycle ---

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._forward)
        signal.signal(signal.SIGINT, self._forward)

        for d in WATCH_DIRS:
            if d.exists():
                self._observer.schedule(self._handler, str(d), recursive=True)
                logger.info("watching: %s", d)
        self._observer.start()

        while True:
            self._start()
            self._wait()
            if self._restart_requested:
                self._restart_requested = False
                logger.info("restarting (file change)")
                continue
            # child exited on its own
            if self._proc.returncode == 0:
                logger.info("child exited cleanly, harness shutting down")
                break
            logger.info("child crashed (code %s), restarting in %.1fs",
                        self._proc.returncode, RESTART_DELAY)
            time.sleep(RESTART_DELAY)

    # --- internals ---

    def _start(self) -> None:
        logger.info("launch: %s", " ".join(self._cmd))
        self._proc = subprocess.Popen(self._cmd)

    def _wait(self) -> None:
        """Block until child exits or a restart is requested."""
        while True:
            rc = self._proc.poll()
            if rc is not None:
                return
            if self._restart_requested:
                self._stop_child()
                return
            time.sleep(0.1)

    def _stop_child(self) -> None:
        if self._proc.poll() is not None:
            return
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

    def _request_restart(self) -> None:
        self._restart_requested = True

    def _forward(self, signum, _frame) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signum)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    child_argv = [sys.executable, "-m", "ui.kscript", "--dev"] + sys.argv[1:]
    Harness(child_argv).run()


if __name__ == "__main__":
    main()
