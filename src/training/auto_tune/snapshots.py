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

from training.auto_tune.session import SessionConfig, SessionDir

# Helpers


def _derive_state_path(curriculum: str, root: Path) -> Path:
    """Derive the curriculum state file path from the curriculum markdown path.

    Mirrors the logic in ``src/training/harness/__main__.py`` lines 202–206:
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


# snapshot


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
    config_data = json.loads(session_dir.config_path.read_text(encoding="utf-8"))
    cfg = SessionConfig.from_dict(config_data)
    cfg.run_counter += 1
    run_number = cfg.run_counter

    session_dir.config_path.write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    run_dir = session_dir.runs_dir / f"{run_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    state_path = _derive_state_path(cfg.curriculum, session_dir._root)
    if state_path.exists():
        shutil.copy2(state_path, run_dir / "state.json")

    if session_dir.events_path.exists():
        shutil.copy2(session_dir.events_path, run_dir / "events.jsonl")

    model_file = session_dir._root / cfg.model_path if cfg.model_path else None
    if model_file is not None and model_file.exists():
        shutil.copy2(model_file, run_dir / "model.bin")

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


# restore


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
    run_dir = session_dir.runs_dir / f"{run_number:03d}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    _assert_no_running_processes(session_dir)

    config_data = json.loads(session_dir.config_path.read_text(encoding="utf-8"))
    cfg = SessionConfig.from_dict(config_data)

    snapshot_state = run_dir / "state.json"
    if snapshot_state.exists():
        state_path = _derive_state_path(cfg.curriculum, session_dir._root)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot_state, state_path)

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
        import os

        try:
            os.kill(pid, 0)
            raise RuntimeError(f"Cannot restore: supervisor is still running (pid={pid})")
        except ProcessLookupError:
            pass  # process gone — safe to proceed
        except PermissionError:
            raise RuntimeError(f"Cannot restore: supervisor is still running (pid={pid})")


# reset


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

    curriculum_state = root / Path(config.curriculum).with_suffix(".json")
    if curriculum_state.exists():
        curriculum_state.unlink()

    session_dir.events_path.write_text("", encoding="utf-8")

    # Delete stale cmd.json to avoid an immediate shutdown on next start.
    cmd_path = session_dir.cmd_path
    if cmd_path.exists():
        cmd_path.unlink()

    if fresh_model and config.model_path:
        model = root / config.model_path
        if model.exists():
            model.unlink()
