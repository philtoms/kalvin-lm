"""Tests for Curriculum and CurriculumState — HRNS-15, HRNS-16, CRS-24..CRS-31."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from trainer.curriculum import Curriculum, CurriculumState, EntryKey
from trainer.curriculum_document import CurriculumDocument

# ── Fixtures ──────────────────────────────────────────────────────────

SAMPLE_DOCUMENT = CurriculumDocument.from_string(
    textwrap.dedent("""\
    ## Objective

    Teach Kalvin basic structure.

    ## Approach

    Step by step.

    ## Lessons

    ### 1

    First lesson.

    ```
    A = B
    ```

    ### 2

    Second lesson.

    ```
    C = D
    ```

    ### 2a

    Bridge lesson.

    ```
    E = F
    ```

    ### 3

    Third lesson.

    ```
    G = H
    ```
""")
)

MULTI_BLOCK_DOCUMENT = CurriculumDocument.from_string(
    textwrap.dedent("""\
    ## Objective

    Multi-block test.

    ## Approach

    Multiple blocks per lesson.

    ## Lessons

    ### 1

    First lesson with two blocks.

    ```
    A = B
    ```

    Intermediate prose.

    ```
    C = D
    ```

    ### 2

    Second lesson.

    ```
    E = F
    ```
""")
)


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


class TestCurriculumDocumentMode:
    """Verify Curriculum with CurriculumDocument input."""

    def test_document_mode_extracts_kscript(self) -> None:
        c = Curriculum(SAMPLE_DOCUMENT)
        # lessons property returns flat list of all kscript blocks
        assert len(c.lessons) == 4
        assert c.lessons[0] == "A = B"
        assert c.lessons[3] == "G = H"

    def test_document_mode_position(self) -> None:
        c = Curriculum(SAMPLE_DOCUMENT)
        assert c.position == 0
        assert c.current() == "A = B"

    def test_document_mode_advance(self) -> None:
        c = Curriculum(SAMPLE_DOCUMENT)
        c.advance()
        assert c.current() == "C = D"

    def test_document_mode_total(self) -> None:
        c = Curriculum(SAMPLE_DOCUMENT)
        assert c.total() == 4

    def test_current_label(self) -> None:
        c = Curriculum(SAMPLE_DOCUMENT)
        assert c.current_label() == "1"
        c.advance()
        assert c.current_label() == "2"

    def test_label_at_position(self) -> None:
        c = Curriculum(SAMPLE_DOCUMENT)
        assert c.label_at_position(0) == "1"
        assert c.label_at_position(2) == "2a"
        assert c.label_at_position(99) is None

    def test_position_of_label(self) -> None:
        c = Curriculum(SAMPLE_DOCUMENT)
        assert c.position_of_label("1") == 0
        assert c.position_of_label("2a") == 2
        assert c.position_of_label("99") is None

    def test_current_lesson(self) -> None:
        c = Curriculum(SAMPLE_DOCUMENT)
        lesson = c.current_lesson()
        assert lesson is not None
        assert lesson.label == "1"
        assert "First lesson" in lesson.prose

    def test_synthetic_document_for_legacy(self) -> None:
        c = Curriculum(["a", "b"])
        doc = c.document
        assert doc.objective == "(auto-generated)"
        assert len(doc.lessons) == 2
        assert doc.lessons[0].label == "1"
        assert doc.lessons[1].label == "2"


class TestCurriculumMultiBlock:
    """Verify Curriculum handles multi-block lessons correctly.

    Position indexes into lessons (not kscript blocks), so current()
    returns the joined kscript for the current lesson.
    """

    def test_multi_block_current(self) -> None:
        c = Curriculum(MULTI_BLOCK_DOCUMENT)
        # Lesson 1 has two blocks: "A = B" and "C = D"
        assert c.current() == "A = B\nC = D"

    def test_multi_block_advance(self) -> None:
        c = Curriculum(MULTI_BLOCK_DOCUMENT)
        c.advance()
        assert c.current() == "E = F"

    def test_multi_block_total(self) -> None:
        c = Curriculum(MULTI_BLOCK_DOCUMENT)
        assert c.total() == 2

    def test_multi_block_lessons_property(self) -> None:
        c = Curriculum(MULTI_BLOCK_DOCUMENT)
        # Flat lessons list has 3 blocks total
        assert c.lessons == ["A = B", "C = D", "E = F"]

    def test_multi_block_position_consistent(self) -> None:
        """position indexes lessons, not blocks."""
        c = Curriculum(MULTI_BLOCK_DOCUMENT)
        assert c.position == 0
        assert c.current_label() == "1"
        c.advance()
        assert c.position == 1
        assert c.current_label() == "2"


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


# ── CRS-24..CRS-27: Label-based tracking ─────────────────────────────


class TestLabelTracking:
    """CRS-24..CRS-27: Label-based tracking in CurriculumState."""

    def test_current_label_first_unsatisfied(self) -> None:
        """CRS-24: current_label returns first unsatisfied label."""
        state = CurriculumState(Curriculum(SAMPLE_DOCUMENT))
        assert state.current_label == "1"

    def test_current_label_advances_after_satisfied(self) -> None:
        """CRS-24: current_label skips satisfied labels."""
        state = CurriculumState(Curriculum(SAMPLE_DOCUMENT))
        state.mark_lesson_satisfied("1")
        assert state.current_label == "2"

    def test_current_label_none_when_all_satisfied(self) -> None:
        """CRS-24: current_label is None when all lessons satisfied."""
        state = CurriculumState(Curriculum(SAMPLE_DOCUMENT))
        state.mark_lesson_satisfied("1")
        state.mark_lesson_satisfied("2")
        state.mark_lesson_satisfied("2a")
        state.mark_lesson_satisfied("3")
        assert state.current_label is None

    def test_mark_lesson_submitted(self) -> None:
        """CRS-25: mark_lesson_submitted adds label to set."""
        state = CurriculumState(Curriculum(SAMPLE_DOCUMENT))
        state.mark_lesson_submitted("1")
        assert state.is_lesson_submitted("1")
        assert not state.is_lesson_submitted("2")

    def test_mark_lesson_satisfied_advances(self) -> None:
        """CRS-26: mark_lesson_satisfied advances curriculum position."""
        state = CurriculumState(Curriculum(SAMPLE_DOCUMENT))
        assert state.curriculum.position == 0
        state.mark_lesson_satisfied("1")
        assert state.curriculum.position == 1  # advanced to "2"
        assert state.is_lesson_satisfied("1")

    def test_is_lesson_submitted_and_satisfied(self) -> None:
        """CRS-27: is_lesson_submitted and is_lesson_satisfied check membership."""
        state = CurriculumState(Curriculum(SAMPLE_DOCUMENT))
        assert not state.is_lesson_submitted("1")
        assert not state.is_lesson_satisfied("1")
        state.mark_lesson_submitted("1")
        assert state.is_lesson_submitted("1")
        state.mark_lesson_satisfied("1")
        assert state.is_lesson_satisfied("1")

    def test_label_and_entry_tracking_coexist(self) -> None:
        """CRS-28: Label and EntryKey tracking are independent."""
        state = CurriculumState(Curriculum(SAMPLE_DOCUMENT))

        # Label tracking
        state.mark_lesson_submitted("1")
        state.mark_lesson_satisfied("1")

        # EntryKey tracking
        key: EntryKey = (42, (1, 2))
        state.mark_submitted(key)
        state.mark_satisfied(key)

        # Both work independently
        assert state.is_lesson_submitted("1")
        assert state.is_lesson_satisfied("1")
        assert state.is_submitted(key)
        assert state.is_satisfied(key)

        # Reset clears both
        state.reset_session()
        assert not state.is_lesson_submitted("1")
        assert not state.is_lesson_satisfied("1")
        assert not state.is_submitted(key)
        assert not state.is_satisfied(key)

    def test_current_label_with_sub_labels(self) -> None:
        """Verify sub-labels like '2a' work correctly."""
        state = CurriculumState(Curriculum(SAMPLE_DOCUMENT))
        state.mark_lesson_satisfied("1")
        state.mark_lesson_satisfied("2")
        assert state.current_label == "2a"
        state.mark_lesson_satisfied("2a")
        assert state.current_label == "3"


# ── CRS-29..CRS-31: Persistence ──────────────────────────────────────


class TestLabelPersistence:
    """CRS-29..CRS-31: Label-based state persistence."""

    def test_save_includes_label_state(self, tmp_path: Path) -> None:
        """CRS-29: save() includes curriculum_file, labels, and label-based state."""
        save_file = tmp_path / "state.json"
        state = CurriculumState(
            Curriculum(SAMPLE_DOCUMENT),
            save_path=save_file,
            curriculum_file="/path/to/curriculum.md",
        )
        state.mark_lesson_submitted("1")
        state.mark_lesson_satisfied("1")
        state.mark_lesson_submitted("2")
        state.save()

        data = json.loads(save_file.read_text())
        assert data["curriculum_file"] == "/path/to/curriculum.md"
        assert data["current_label"] == "2"
        assert "1" in data["lesson_submitted"]
        assert "2" in data["lesson_submitted"]
        assert "1" in data["lesson_satisfied"]
        assert data["version"] == 2

    def test_load_new_format_with_file(self, tmp_path: Path) -> None:
        """CRS-30: load() reconstructs state from file when curriculum_file exists."""
        # Write the curriculum document to disk
        curriculum_path = tmp_path / "curricula" / "test.md"
        curriculum_path.parent.mkdir(parents=True)
        curriculum_path.write_text(
            textwrap.dedent("""\
            ## Objective

            Teach Kalvin basic structure.

            ## Approach

            Step by step.

            ## Lessons

            ### 1

            First lesson.

            ```
            A = B
            ```

            ### 2

            Second lesson.

            ```
            C = D
            ```

            ### 2a

            Bridge lesson.

            ```
            E = F
            ```

            ### 3

            Third lesson.

            ```
            G = H
            ```
        """)
        )

        save_file = tmp_path / "state.json"
        data = {
            "version": 2,
            "position": 2,
            "lessons": ["A = B", "C = D", "E = F", "G = H"],
            "curriculum_file": str(curriculum_path),
            "current_label": "2a",
            "lesson_submitted": ["1", "2", "2a"],
            "lesson_satisfied": ["1", "2"],
            "submitted": [],
            "satisfied": [],
            "pending": [],
            "event_log": [],
        }
        save_file.write_text(json.dumps(data))

        state = CurriculumState.load(save_file)
        assert state.curriculum_file == str(curriculum_path)
        # Document loaded from file preserves labels
        assert state.curriculum.document.lessons[2].label == "2a"
        assert state.is_lesson_submitted("1")
        assert state.is_lesson_submitted("2a")
        assert state.is_lesson_satisfied("1")
        assert state.is_lesson_satisfied("2")
        assert not state.is_lesson_satisfied("2a")
        # current_label derives correctly from original labels
        assert state.current_label == "2a"

    def test_load_new_format_file_missing_falls_back(self, tmp_path: Path) -> None:
        """load() falls back to synthetic when curriculum_file doesn't exist."""
        save_file = tmp_path / "state.json"
        data = {
            "version": 2,
            "position": 1,
            "lessons": ["A = B", "C = D", "E = F"],
            "curriculum_file": "/nonexistent/path.md",
            "current_label": "2",
            "lesson_submitted": ["1"],
            "lesson_satisfied": ["1"],
            "submitted": [],
            "satisfied": [],
            "pending": [],
            "event_log": [],
        }
        save_file.write_text(json.dumps(data))

        state = CurriculumState.load(save_file)
        # Falls back to synthetic document with labels "1", "2", "3"
        assert state.curriculum.document.objective == "(auto-generated)"
        assert state.current_label == "2"

    def test_load_legacy_format(self, tmp_path: Path) -> None:
        """CRS-31: load() handles legacy format (flat lessons + position)."""
        save_file = tmp_path / "state.json"
        data = {
            "position": 1,
            "lessons": ["lesson A", "lesson B", "lesson C"],
            "submitted": [[100, [1, 2]]],
            "satisfied": [[100, [1, 2]]],
            "pending": [[300, [4, 5, 6]]],
            "event_log": [{"type": "test", "data": {}}],
        }
        save_file.write_text(json.dumps(data))

        state = CurriculumState.load(save_file)
        assert state.curriculum.position == 1
        assert state.curriculum.lessons == ["lesson A", "lesson B", "lesson C"]
        assert state.curriculum.current() == "lesson B"
        assert state.submitted == {(100, (1, 2))}
        assert state.satisfied == {(100, (1, 2))}
        assert state.pending == {(300, (4, 5, 6))}
        # No curriculum_file in legacy format
        assert state.curriculum_file is None

    def test_round_trip_with_document_file(self, tmp_path: Path) -> None:
        """Full round-trip: save with document + file, load back correctly."""
        # Write curriculum to file
        curriculum_path = tmp_path / "curricula" / "test.md"
        curriculum_path.parent.mkdir(parents=True)
        curriculum_path.write_text(
            textwrap.dedent("""\
            ## Objective

            Teach Kalvin basic structure.

            ## Approach

            Step by step.

            ## Lessons

            ### 1

            First lesson.

            ```
            A = B
            ```

            ### 2

            Second lesson.

            ```
            C = D
            ```

            ### 2a

            Bridge lesson.

            ```
            E = F
            ```
        """)
        )

        doc = CurriculumDocument.from_file(curriculum_path)
        save_file = tmp_path / "state.json"
        state = CurriculumState(
            Curriculum(doc),
            save_path=save_file,
            curriculum_file=str(curriculum_path),
        )
        state.mark_lesson_submitted("1")
        state.mark_lesson_satisfied("1")
        state.mark_lesson_submitted("2")

        key: EntryKey = (42, (1, 2))
        state.mark_submitted(key)
        state.mark_satisfied(key)

        state.save()
        loaded = CurriculumState.load(save_file)

        # Document loaded from file, not synthetic
        assert loaded.curriculum.document.objective == "Teach Kalvin basic structure."
        assert loaded.curriculum_file == str(curriculum_path)
        assert loaded.is_lesson_submitted("1")
        assert loaded.is_lesson_satisfied("1")
        assert loaded.is_lesson_submitted("2")
        assert loaded.is_submitted(key)
        assert loaded.is_satisfied(key)
        # current_label correctly reflects "2" (first unsatisfied after "1")
        assert loaded.current_label == "2"


# ── HRNS-15: State persistence across restart (legacy) ───────────────


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
