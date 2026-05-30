"""Tests for Curriculum and CurriculumState — HRNS-15, HRNS-16."""

from __future__ import annotations

from pathlib import Path

import pytest

from trainer.curriculum import Curriculum, CurriculumState, EntryKey

# ── Curriculum unit tests ─────────────────────────────────────────────


class TestCurriculumAdvance:
    """Verify Curriculum.advance() moves position and current() returns next lesson."""

    def test_advance_moves_position(self) -> None:
        c = Curriculum(["lesson1", "lesson2", "lesson3"])
        assert c.current() == "lesson1"
        c.advance()
        assert c.current() == "lesson2"
        assert c.remaining() == 2

    def test_advance_to_last(self) -> None:
        c = Curriculum(["a", "b"])
        c.advance()
        assert c.current() == "b"
        assert c.remaining() == 1
        assert not c.is_complete()

    def test_total_and_remaining(self) -> None:
        c = Curriculum(["a", "b", "c"])
        assert c.total() == 3
        assert c.remaining() == 3
        c.advance()
        assert c.remaining() == 2


class TestCurriculumComplete:
    """Verify is_complete() and current() when all lessons consumed."""

    def test_complete_returns_none(self) -> None:
        c = Curriculum(["only"])
        assert c.current() == "only"
        c.advance()
        assert c.current() is None
        assert c.is_complete()

    def test_empty_curriculum_is_complete(self) -> None:
        c = Curriculum([])
        assert c.is_complete()
        assert c.current() is None
        assert c.remaining() == 0


# ── CurriculumState set operations ────────────────────────────────────


class TestStateMarkSubmittedSatisfied:
    """Verify set operations: submitted → satisfied, pending → submitted."""

    def test_mark_submitted_adds_to_set(self) -> None:
        state = CurriculumState(Curriculum(["a"]))
        key: EntryKey = (42, (1, 2, 3))
        state.mark_submitted(key)
        assert state.is_submitted(key)
        assert not state.is_satisfied(key)

    def test_mark_satisfied(self) -> None:
        state = CurriculumState(Curriculum(["a"]))
        key: EntryKey = (42, (1, 2, 3))
        state.mark_submitted(key)
        state.mark_satisfied(key)
        assert state.is_satisfied(key)
        assert state.is_submitted(key)

    def test_mark_submitted_removes_from_pending(self) -> None:
        state = CurriculumState(Curriculum(["a"]))
        key: EntryKey = (42, (1, 2, 3))
        state.pending.add(key)
        state.mark_submitted(key)
        assert key not in state.pending
        assert key in state.submitted

    def test_pending_to_submitted_transition(self) -> None:
        state = CurriculumState(Curriculum(["a"]))
        key1: EntryKey = (100, (5,))
        key2: EntryKey = (200, (6, 7))
        state.pending.add(key1)
        state.pending.add(key2)
        state.mark_submitted(key1)
        assert key1 in state.submitted
        assert key1 not in state.pending
        assert key2 in state.pending


class TestStateResetSession:
    """Verify reset_session() clears sets but preserves position and event_log."""

    def test_reset_clears_sets(self) -> None:
        state = CurriculumState(Curriculum(["a", "b"]))
        state.mark_submitted((42, (1,)))
        state.mark_satisfied((42, (1,)))
        state.pending.add((99, (2,)))
        state.curriculum.advance()

        state.reset_session()

        assert len(state.submitted) == 0
        assert len(state.satisfied) == 0
        assert len(state.pending) == 0

    def test_reset_preserves_position(self) -> None:
        state = CurriculumState(Curriculum(["a", "b", "c"]))
        state.curriculum.advance()
        assert state.curriculum.position == 1
        state.reset_session()
        assert state.curriculum.position == 1

    def test_reset_preserves_event_log(self) -> None:
        state = CurriculumState(Curriculum(["a"]))
        state.log_event("test", {"detail": "hello"})
        state.reset_session()
        assert len(state.event_log) == 1
        assert state.event_log[0]["type"] == "test"


# ── HRNS-15: State persistence across restart ────────────────────────


class TestStatePersistenceAcrossRestart:
    """HRNS-15: Curriculum state round-trips through JSON persistence."""

    def test_state_persistence_across_restart(self, tmp_path: Path) -> None:
        save_file = tmp_path / "state.json"

        # Build state with 3 lessons
        curriculum = Curriculum(["lesson A", "lesson B", "lesson C"])
        state = CurriculumState(curriculum, save_path=save_file)

        # Mark some entries submitted and satisfied
        key1: EntryKey = (100, (1, 2))
        key2: EntryKey = (200, (3,))
        key3: EntryKey = (300, (4, 5, 6))

        state.mark_submitted(key1)
        state.mark_submitted(key2)
        state.mark_satisfied(key1)
        state.pending.add(key3)

        # Advance position
        state.curriculum.advance()

        # Add event log entries
        state.log_event("session_start", {"goal": "test"})
        state.log_event("lesson_complete", {"position": 0})

        # Save
        state.save()

        # Verify file was written
        assert save_file.exists()

        # Load back
        loaded = CurriculumState.load(save_file)

        # Verify all state is identical
        assert loaded.curriculum.position == 1
        assert loaded.curriculum.lessons == ["lesson A", "lesson B", "lesson C"]
        assert loaded.curriculum.current() == "lesson B"

        # Verify sets round-trip correctly (tuples from JSON lists)
        assert loaded.submitted == {(100, (1, 2)), (200, (3,))}
        assert loaded.satisfied == {(100, (1, 2))}
        assert loaded.pending == {(300, (4, 5, 6))}

        # Verify EntryKey type is correct (tuple, not list)
        for key in loaded.submitted:
            assert isinstance(key, tuple)
            assert isinstance(key[0], int)
            assert isinstance(key[1], tuple)

        # Verify event log
        assert len(loaded.event_log) == 2
        assert loaded.event_log[0]["type"] == "session_start"
        assert loaded.event_log[0]["data"]["goal"] == "test"
        assert loaded.event_log[1]["type"] == "lesson_complete"
        assert "timestamp" in loaded.event_log[0]

    def test_save_requires_path(self) -> None:
        state = CurriculumState(Curriculum(["a"]))
        with pytest.raises(ValueError, match="No save path"):
            state.save()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "state.json"
        state = CurriculumState(Curriculum(["a"]), save_path=nested)
        state.save()
        assert nested.exists()
