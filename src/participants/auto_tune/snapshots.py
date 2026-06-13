"""Snapshot, restore, and reset operations for auto-tune sessions.

Provides :func:`snapshot` and :func:`restore` so that an auto-tune session
can capture and reinstate its complete training state — curriculum state
file, event log, Kalvin model, and git metadata.  Also provides
:func:`reset` to clear session state for a fresh start.

Spec ref: specs/auto-tune.md §Snapshot and Restore (rules 32–38), §Reset (rules 39–42)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from participants.auto_tune.session import SessionConfig, SessionDir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_state_path(curriculum: str, root: Path) -> Path:
    """Derive the curriculum state file path from the curriculum markdown path.

    Mirrors the logic in ``src/harness/__main__.py`` lines 202–206:
    ``curricula/first-steps.md`` → ``curricula/first-steps.json``.

    The path is resolved against *root* (the project root directory).
    """
    return root / Path(curriculum).with_suffix(".json")


def _git_info() -> dict[str, Any]:
    """Capture git HEAD, branch, and dirty status."""
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        git_head = ""

    try:
        git_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        git_branch = ""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_dirty = result.stdout.strip() != ""
    except subprocess.CalledProcessError:
        git_dirty = False

    return {
        "git_head": git_head,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
    }


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


def snapshot(session_dir: SessionDir) -> int:
    """Capture a snapshot of the current auto-tune session state.

    1. Increments the run counter in ``config.json``.
    2. Creates ``runs/<n>/`` with copies of the curriculum state file,
       event log, Kalvin model (if it exists), and git metadata.

    Args:
        session_dir: Bound :class:`SessionDir` for the session.

    Returns:
        The new run number (int).
    """
    # 1. Load config, increment run counter, persist
    config_data = json.loads(session_dir.config_path.read_text(encoding="utf-8"))
    cfg = SessionConfig.from_dict(config_data)
    cfg.run_counter += 1
    run_number = cfg.run_counter

    session_dir.config_path.write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    # 2. Create run directory
    run_dir = session_dir.runs_dir / f"{run_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 3. Copy curriculum state file (if it exists)
    state_path = _derive_state_path(cfg.curriculum, session_dir._root)
    if state_path.exists():
        shutil.copy2(state_path, run_dir / "state.json")

    # 4. Copy events.jsonl
    if session_dir.events_path.exists():
        shutil.copy2(session_dir.events_path, run_dir / "events.jsonl")

    # 5. Copy model file (if configured and exists)
    model_file = session_dir._root / cfg.model_path if cfg.model_path else None
    if model_file is not None and model_file.exists():
        shutil.copy2(model_file, run_dir / "model.bin")

    # 6. Write meta.json
    git = _git_info()
    meta: dict[str, Any] = {
        "run": run_number,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_head": git["git_head"],
        "git_branch": git["git_branch"],
        "git_dirty": git["git_dirty"],
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )

    return run_number


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------


def restore(session_dir: SessionDir, run_number: int) -> None:
    """Restore session state from a previously captured snapshot.

    Copies the curriculum state file and Kalvin model from the specified
    run back to their working locations.

    Raises :class:`RuntimeError` if a harness or supervisor process is
    still running.  Raises :class:`FileNotFoundError` if the run directory
    does not exist.

    Args:
        session_dir: Bound :class:`SessionDir` for the session.
        run_number: The run number to restore from.
    """
    # 1. Derive run directory and verify it exists
    run_dir = session_dir.runs_dir / f"{run_number:03d}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # 2. Guard: verify no harness or supervisor is still running
    _assert_no_running_processes(session_dir)

    # 3. Load config to derive paths
    config_data = json.loads(session_dir.config_path.read_text(encoding="utf-8"))
    cfg = SessionConfig.from_dict(config_data)

    # 4. Restore curriculum state file
    snapshot_state = run_dir / "state.json"
    if snapshot_state.exists():
        state_path = _derive_state_path(cfg.curriculum, session_dir._root)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot_state, state_path)

    # 5. Restore model file
    snapshot_model = run_dir / "model.bin"
    if snapshot_model.exists():
        if cfg.model_path is not None:
            model_path = session_dir._root / cfg.model_path
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot_model, model_path)


def _assert_no_running_processes(session_dir: SessionDir) -> None:
    """Raise RuntimeError if status.json indicates a running process.

    Checks the ``pid`` field in ``status.json``.  If the file does not
    exist or has no active PID, the check passes silently.
    """
    status_path = session_dir.status_path
    if not status_path.exists():
        return

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    pid = status.get("pid")
    if pid and pid != 0:
        # Check if the process is actually running
        try:
            import os

            os.kill(pid, 0)
            raise RuntimeError(f"Cannot restore: supervisor is still running (pid={pid})")
        except ProcessLookupError:
            # Process no longer exists — safe to proceed
            pass
        except PermissionError:
            # Process exists but we can't signal it — treat as running
            raise RuntimeError(f"Cannot restore: supervisor is still running (pid={pid})")


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def reset(session_dir: SessionDir, *, fresh_model: bool = False) -> None:
    """Clear auto-tune session state without destroying run history.

    Deletes the curriculum state file (derived from the curriculum path by
    swapping the extension to ``.json``), truncates ``events.jsonl`` to empty,
    and optionally deletes the Kalvin model file.

    Safe to call multiple times — missing files are silently skipped.
    Does **not** modify ``config.json``, the run counter, or any files
    under ``runs/``.

    Spec ref: rules 39–42.

    Args:
        session_dir: Bound session directory with loaded config.
        fresh_model: If ``True``, also delete the Kalvin model file
            referenced by ``config.model_path``.
    """
    config = session_dir.config
    root = session_dir._root

    # 1. Delete curriculum state file (e.g. curricula/first-steps.md → curricula/first-steps.json)
    curriculum_state = root / Path(config.curriculum).with_suffix(".json")
    if curriculum_state.exists():
        curriculum_state.unlink()

    # 2. Truncate events.jsonl
    events_path = session_dir.events_path
    events_path.write_text("", encoding="utf-8")

    # 3. Delete stale cmd.json (prevents immediate shutdown on next supervisor start)
    cmd_path = session_dir.cmd_path
    if cmd_path.exists():
        cmd_path.unlink()

    # 4. Optionally delete model file
    if fresh_model and config.model_path:
        model = root / config.model_path
        if model.exists():
            model.unlink()
