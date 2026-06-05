"""Snapshot, restore, and reset operations for auto-tune sessions.

Spec ref: specs/auto-tune.md §Snapshot and Restore (rules 32–38), §Reset (rules 39–42)

Functions:
    reset — clear auto-tune session state for a fresh start
"""

from __future__ import annotations

from pathlib import Path

from participants.auto_tune.session import SessionDir


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

    # 1. Delete curriculum state file (e.g. curricula/first-steps.md → curricula/first-steps.json)
    curriculum_state = Path(config.curriculum).with_suffix(".json")
    if curriculum_state.exists():
        curriculum_state.unlink()

    # 2. Truncate events.jsonl
    events_path = session_dir.events_path
    events_path.write_text("", encoding="utf-8")

    # 3. Optionally delete model file
    if fresh_model and config.model_path:
        model = Path(config.model_path)
        if model.exists():
            model.unlink()
