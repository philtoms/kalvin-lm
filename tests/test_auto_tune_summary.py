"""Tests for the auto-tune run summary — the arbiter signal.

``summarize`` aggregates a run's event stream and trainer state into a
verdict (``run-summary.json``) that tells the agent whether the goal was
met and, if not, where to look. It replaces per-turn supervision prose
with a machine-readable signal (see SKILL.md §Pi-in-the-Loop Model).

Spec ref: specs/auto-tune.md §Run Summary.
"""

from __future__ import annotations

import json
from pathlib import Path

from training.auto_tune import orchestrate
from training.auto_tune.session import SessionDir


# ── Helpers ───────────────────────────────────────────────────────────


def _session_dir(tmp_path: Path) -> SessionDir:
    """Bind a SessionDir to *tmp_path* with a minimal curriculum pointer."""
    (tmp_path / "config.json").write_text(json.dumps({"curriculum": "snap.md"}))
    (tmp_path / "snap.json").write_text("{}")
    return SessionDir(root=tmp_path, base_dir=".", _session="")


def _write_events(tmp_path: Path, events: list[dict]) -> None:
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )


def _write_status(tmp_path: Path, status: dict) -> None:
    (tmp_path / "status.json").write_text(json.dumps(status))


def _write_state(tmp_path: Path, submitted: int, satisfied: int) -> None:
    (tmp_path / "snap.json").write_text(
        json.dumps(
            {
                "submitted": [[1, []]] * submitted,
                "satisfied": [[1, []]] * satisfied,
                "lesson_satisfied": [],
            }
        )
    )


# ── Outcome classification ────────────────────────────────────────────


def test_summary_crashed_when_error_event_present(tmp_path: Path) -> None:
    sd = _session_dir(tmp_path)
    _write_events(
        tmp_path,
        [{"seq": 1, "type": "connected"}, {"seq": 2, "type": "error", "message": "x"}],
    )
    _write_status(tmp_path, {"connected": False, "state": "errored", "last_event_seq": 2})

    summary = orchestrate.summarize(sd, write=False)

    assert summary["outcome"] == "crashed"
    assert "error surfaced" in summary["diagnosis"]


def test_summary_supervisor_stalled_when_ratify_unanswered(tmp_path: Path) -> None:
    sd = _session_dir(tmp_path)
    _write_events(
        tmp_path,
        [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "ratify_request", "significance": {"raw": 127, "level": "S3"}},
        ],
    )
    # Supervisor still connected and waiting for a command that never came.
    _write_status(
        tmp_path,
        {"connected": True, "state": "waiting_for_command", "last_event_seq": 2},
    )

    summary = orchestrate.summarize(sd, write=False)

    assert summary["outcome"] == "supervisor-stalled"
    assert summary["ratify_requests"] == 1
    assert "did not enact" in summary["diagnosis"]


def test_summary_incomplete_for_live_run(tmp_path: Path) -> None:
    sd = _session_dir(tmp_path)
    _write_events(
        tmp_path,
        [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "rationalise", "kind": "ground", "significance": {"raw": 255, "level": "S1"}},
        ],
    )
    _write_status(
        tmp_path,
        {"connected": True, "state": "waiting_for_event", "last_event_seq": 2},
    )

    summary = orchestrate.summarize(sd, write=False)

    assert summary["outcome"] == "incomplete"


def test_summary_completed_when_terminal_complete_event(tmp_path: Path) -> None:
    sd = _session_dir(tmp_path)
    _write_events(
        tmp_path,
        [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "progress", "status": "complete", "lessons_completed": 5},
            {"seq": 3, "type": "disconnected"},
        ],
    )
    _write_status(tmp_path, {"connected": False, "state": "run_complete", "last_event_seq": 3})
    _write_state(tmp_path, submitted=5, satisfied=5)

    summary = orchestrate.summarize(sd, write=False)

    assert summary["outcome"] == "completed"
    assert "5 lesson(s) satisfied" in summary["diagnosis"]
    assert summary["entries_satisfied"] == 5
    assert summary["entries_deadlocked"] == 0


def test_summary_deadlocked_when_entries_unsatisfied_and_no_completion(
    tmp_path: Path,
) -> None:
    sd = _session_dir(tmp_path)
    _write_events(
        tmp_path,
        [
            {"seq": 1, "type": "connected"},
            # Lesson 3's misfits: S3 frames, no lesson_complete, run ends idle.
            {"seq": 2, "type": "rationalise", "kind": "frame", "significance": {"raw": 127, "level": "S3"}},
            {"seq": 3, "type": "disconnected"},
        ],
    )
    _write_status(
        tmp_path,
        {"connected": False, "state": "waiting_for_event", "last_event_seq": 3},
    )
    _write_state(tmp_path, submitted=30, satisfied=12)

    summary = orchestrate.summarize(sd, write=False)

    assert summary["outcome"] == "deadlocked"
    assert summary["entries_total"] == 30
    assert summary["entries_satisfied"] == 12
    assert summary["entries_deadlocked"] == 18
    assert "reactor.py" in summary["diagnosis"]


# ── Significance profile ──────────────────────────────────────────────


def test_summary_significance_profile_aggregates_bands(tmp_path: Path) -> None:
    sd = _session_dir(tmp_path)
    _write_events(
        tmp_path,
        [
            {"seq": 1, "type": "connected"},
            # grounds (S1)
            {"seq": 2, "type": "rationalise", "kind": "ground", "significance": {"raw": 255, "level": "S1"}},
            {"seq": 3, "type": "rationalise", "kind": "ground", "significance": {"raw": 0xE0, "level": "S1"}},
            # S3 frames (proposals)
            {"seq": 4, "type": "rationalise", "kind": "frame", "significance": {"raw": 127, "level": "S3"}},
            {"seq": 5, "type": "rationalise", "kind": "frame", "significance": {"raw": 0x7F, "level": "S3"}},
            # a ratify_request is not a rationalise event
            {"seq": 6, "type": "ratify_request", "significance": {"raw": 127, "level": "S3"}},
        ],
    )
    _write_status(
        tmp_path,
        {"connected": False, "state": "run_complete", "last_event_seq": 6},
    )

    summary = orchestrate.summarize(sd, write=False)

    assert summary["significance"] == {"S1": 2, "S2": 0, "S3": 2, "S4": 0, "ground": 2, "frame": 2}
    assert summary["proposals_emitted"] == 2  # frames only
    assert summary["ratify_requests"] == 1


# ── Output ────────────────────────────────────────────────────────────


def test_summary_writes_run_summary_json(tmp_path: Path) -> None:
    sd = _session_dir(tmp_path)
    _write_events(tmp_path, [{"seq": 1, "type": "connected"}])
    _write_status(tmp_path, {"connected": True, "state": "waiting_for_event", "last_event_seq": 1})

    orchestrate.summarize(sd, write=True)

    written = json.loads((tmp_path / "run-summary.json").read_text())
    assert written["outcome"] == "incomplete"


def test_summary_degrades_gracefully_without_trainer_state(tmp_path: Path) -> None:
    """When the curriculum state file is absent, entry counts are null but
    outcome is still inferred from the event stream."""
    sd = _session_dir(tmp_path)
    (tmp_path / "snap.json").unlink()  # no trainer state
    _write_events(
        tmp_path,
        [
            {"seq": 1, "type": "connected"},
            {"seq": 2, "type": "progress", "status": "complete", "lessons_completed": 3},
        ],
    )
    _write_status(tmp_path, {"connected": False, "state": "run_complete", "last_event_seq": 2})

    summary = orchestrate.summarize(sd, write=False)

    assert summary["outcome"] == "completed"
    assert summary["entries_total"] is None
    assert summary["entries_deadlocked"] is None
