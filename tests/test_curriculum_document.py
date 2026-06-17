"""Tests for CurriculumDocument markdown parser and amendment.

Covers CRS-1 through CRS-23 from the curriculum spec.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from training.trainer.curriculum_document import (
    CurriculumDocument,
    CurriculumParseError,
    Lesson,
)

# ── Fixtures ──────────────────────────────────────────────────────────

VALID_CURRICULUM = textwrap.dedent("""\
    ## Objective

    Teach Kalvin the SVO structure of "Mary had a little lamb".

    ## Approach

    Introduce components one at a time: subject, verb, object.

    ## Lessons

    ### 1

    This lesson introduces the subject.

    ```
    M = S
    ```

    ### 2

    This lesson introduces the verb.

    ```
    H = V
    ```

    ### 2a

    A refinement combining subject and verb.

    ```
    MH = SV
    ```

    ### 3

    This lesson introduces the full SVO structure.

    ```
    MHALL = SVO
    ```
""")

VALID_TWO_LESSONS = textwrap.dedent("""\
    ## Objective

    A simple curriculum.

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
""")


# ── CRS-1: from_file parses valid curriculum ──────────────────────────


class TestFromFile:
    """CRS-1: from_file parses a valid curriculum file."""

    def test_from_file_parses_valid_document(self, tmp_path: Path) -> None:
        path = tmp_path / "test.md"
        path.write_text(VALID_CURRICULUM)
        doc = CurriculumDocument.from_file(path)
        assert doc.objective == 'Teach Kalvin the SVO structure of "Mary had a little lamb".'
        assert doc.approach == "Introduce components one at a time: subject, verb, object."
        assert len(doc.lessons) == 4
        assert doc.source_path == path


# ── CRS-2: from_string parses valid curriculum ────────────────────────


class TestFromString:
    """CRS-2: from_string parses a valid curriculum from string."""

    def test_from_string_parses_valid_document(self) -> None:
        doc = CurriculumDocument.from_string(VALID_CURRICULUM)
        assert doc.objective == 'Teach Kalvin the SVO structure of "Mary had a little lamb".'
        assert doc.source_path is None

    def test_from_string_two_lessons(self) -> None:
        doc = CurriculumDocument.from_string(VALID_TWO_LESSONS)
        assert len(doc.lessons) == 2


# ── CRS-3..CRS-5: Missing sections ───────────────────────────────────


class TestMissingSections:
    """CRS-3..CRS-5: Parsing rejects documents with missing sections."""

    def test_rejects_missing_objective(self) -> None:
        text = textwrap.dedent("""\
            ## Approach

            Step by step.

            ## Lessons

            ### 1

            ```
            A = B
            ```
        """)
        with pytest.raises(CurriculumParseError, match="Missing required section.*Objective"):
            CurriculumDocument.from_string(text)

    def test_rejects_missing_approach(self) -> None:
        text = textwrap.dedent("""\
            ## Objective

            Teach something.

            ## Lessons

            ### 1

            ```
            A = B
            ```
        """)
        with pytest.raises(CurriculumParseError, match="Missing required section.*Approach"):
            CurriculumDocument.from_string(text)

    def test_rejects_missing_lessons_section(self) -> None:
        text = textwrap.dedent("""\
            ## Objective

            Teach something.

            ## Approach

            Step by step.
        """)
        with pytest.raises(CurriculumParseError, match="Missing required section.*Lessons"):
            CurriculumDocument.from_string(text)


# ── CRS-6: No lessons ────────────────────────────────────────────────


class TestNoLessons:
    """CRS-6: Parsing rejects a document with no lessons."""

    def test_rejects_no_lessons(self) -> None:
        text = textwrap.dedent("""\
            ## Objective

            Teach something.

            ## Approach

            Step by step.

            ## Lessons
        """)
        with pytest.raises(CurriculumParseError, match="At least one lesson"):
            CurriculumDocument.from_string(text)


# ── CRS-7: Duplicate labels ──────────────────────────────────────────


class TestDuplicateLabels:
    """CRS-7: Parsing rejects duplicate lesson labels."""

    def test_rejects_duplicate_labels(self) -> None:
        text = textwrap.dedent("""\
            ## Objective

            Teach something.

            ## Approach

            Step by step.

            ## Lessons

            ### 1

            ```
            A = B
            ```

            ### 1

            ```
            C = D
            ```
        """)
        with pytest.raises(CurriculumParseError, match="Duplicate lesson label.*1"):
            CurriculumDocument.from_string(text)


# ── CRS-8: Invalid label format ──────────────────────────────────────


class TestInvalidLabel:
    """CRS-8: Parsing rejects invalid label formats."""

    def test_rejects_invalid_label_format(self) -> None:
        text = textwrap.dedent("""\
            ## Objective

            Teach something.

            ## Approach

            Step by step.

            ## Lessons

            ### abc

            ```
            A = B
            ```
        """)
        with pytest.raises(CurriculumParseError, match="Invalid lesson label.*abc"):
            CurriculumDocument.from_string(text)

    def test_rejects_multi_char_sublabel(self) -> None:
        text = textwrap.dedent("""\
            ## Objective

            Teach something.

            ## Approach

            Step by step.

            ## Lessons

            ### 1ab

            ```
            A = B
            ```
        """)
        with pytest.raises(CurriculumParseError, match="Invalid lesson label"):
            CurriculumDocument.from_string(text)


# ── CRS-9: objective property ────────────────────────────────────────


class TestObjectiveProperty:
    """CRS-9: document.objective returns the Objective section content."""

    def test_objective_property(self) -> None:
        doc = CurriculumDocument.from_string(VALID_CURRICULUM)
        assert "Teach Kalvin" in doc.objective
        assert "SVO" in doc.objective


# ── CRS-10: approach property ─────────────────────────────────────────


class TestApproachProperty:
    """CRS-10: document.approach returns the Approach section content."""

    def test_approach_property(self) -> None:
        doc = CurriculumDocument.from_string(VALID_CURRICULUM)
        assert "Introduce components" in doc.approach


# ── CRS-11: lessons ordered ──────────────────────────────────────────


class TestLessonsOrdered:
    """CRS-11: document.lessons returns ordered Lesson objects."""

    def test_lessons_ordered(self) -> None:
        doc = CurriculumDocument.from_string(VALID_CURRICULUM)
        labels = [lesson.label for lesson in doc.lessons]
        assert labels == ["1", "2", "2a", "3"]


# ── CRS-12: lesson label ─────────────────────────────────────────────


class TestLessonLabel:
    """CRS-12: Lesson.label returns the stable heading label."""

    def test_lesson_label(self) -> None:
        doc = CurriculumDocument.from_string(VALID_CURRICULUM)
        assert doc.lessons[0].label == "1"
        assert doc.lessons[2].label == "2a"


# ── CRS-13: lesson prose ─────────────────────────────────────────────


class TestLessonProse:
    """CRS-13: Lesson.prose returns non-code-block text."""

    def test_lesson_prose(self) -> None:
        doc = CurriculumDocument.from_string(VALID_CURRICULUM)
        assert "introduces the subject" in doc.lessons[0].prose
        assert "```" not in doc.lessons[0].prose


# ── CRS-14: lesson kscript ──────────────────────────────────────────


class TestLessonKscript:
    """CRS-14: Lesson.kscript returns contents of fenced code blocks."""

    def test_lesson_kscript(self) -> None:
        doc = CurriculumDocument.from_string(VALID_CURRICULUM)
        assert doc.lessons[0].kscript == ["M = S"]

    def test_lesson_kscript_multiple_blocks(self) -> None:
        text = textwrap.dedent("""\
            ## Objective

            Teach something.

            ## Approach

            Step by step.

            ## Lessons

            ### 1

            First block.

            ```
            A = B
            ```

            Second block.

            ```
            C = D
            ```
        """)
        doc = CurriculumDocument.from_string(text)
        assert doc.lessons[0].kscript == ["A = B", "C = D"]

    def test_lesson_no_code_blocks(self) -> None:
        text = textwrap.dedent("""\
            ## Objective

            Teach something.

            ## Approach

            Step by step.

            ## Lessons

            ### 1

            Just prose, no code.
        """)
        doc = CurriculumDocument.from_string(text)
        assert doc.lessons[0].kscript == []
        assert "Just prose" in doc.lessons[0].prose


# ── CRS-15: find_lesson ─────────────────────────────────────────────


class TestFindLesson:
    """CRS-15: find_lesson returns matching Lesson or None."""

    def test_find_lesson(self) -> None:
        doc = CurriculumDocument.from_string(VALID_CURRICULUM)
        lesson = doc.find_lesson("2a")
        assert lesson is not None
        assert lesson.label == "2a"
        assert "refinement" in lesson.prose

    def test_find_lesson_not_found(self) -> None:
        doc = CurriculumDocument.from_string(VALID_CURRICULUM)
        assert doc.find_lesson("99") is None


# ── CRS-16: all_labels ──────────────────────────────────────────────


class TestAllLabels:
    """CRS-16: all_labels returns labels in document order."""

    def test_all_labels(self) -> None:
        doc = CurriculumDocument.from_string(VALID_CURRICULUM)
        assert doc.all_labels() == ["1", "2", "2a", "3"]


# ── CRS-17..CRS-23: Amendment ───────────────────────────────────────


class TestAmend:
    """CRS-17..CRS-23: Amendment operations."""

    def _write_and_load(self, tmp_path: Path) -> CurriculumDocument:
        """Write VALID_TWO_LESSONS to a temp file and load it."""
        path = tmp_path / "curriculum.md"
        path.write_text(VALID_TWO_LESSONS)
        return CurriculumDocument.from_file(path)

    def test_amend_insert(self, tmp_path: Path) -> None:
        """CRS-17: Insert a lesson after a specified label."""
        doc = self._write_and_load(tmp_path)
        new_lesson = Lesson(label="1a", prose="Bridge lesson.", kscript=["X = Y"])
        doc.amend("insert", after_label="1", lesson=new_lesson)
        assert doc.all_labels() == ["1", "1a", "2"]
        # Verify write-back
        doc2 = CurriculumDocument.from_file(doc.source_path)
        assert doc2.all_labels() == ["1", "1a", "2"]

    def test_amend_append(self, tmp_path: Path) -> None:
        """CRS-18: Append a lesson at the end."""
        doc = self._write_and_load(tmp_path)
        new_lesson = Lesson(label="3", prose="Third lesson.", kscript=["E = F"])
        doc.amend("append", lesson=new_lesson)
        assert doc.all_labels() == ["1", "2", "3"]

    def test_amend_modify(self, tmp_path: Path) -> None:
        """CRS-19: Replace a lesson at a given label."""
        doc = self._write_and_load(tmp_path)
        modified = Lesson(label="1", prose="Updated lesson.", kscript=["Z = W"])
        doc.amend("modify", label="1", lesson=modified)
        assert doc.lessons[0].prose == "Updated lesson."
        assert doc.lessons[0].kscript == ["Z = W"]
        # Label 2 unchanged
        assert doc.lessons[1].label == "2"

    def test_amend_raises_for_missing_target(self, tmp_path: Path) -> None:
        """CRS-20: Amendment raises ValueError for nonexistent target label."""
        doc = self._write_and_load(tmp_path)
        new_lesson = Lesson(label="4", prose="New.", kscript=[])
        with pytest.raises(ValueError, match="not found"):
            doc.amend("insert", after_label="99", lesson=new_lesson)

    def test_amend_raises_for_duplicate_label(self, tmp_path: Path) -> None:
        """CRS-21: Amendment raises ValueError for duplicate label."""
        doc = self._write_and_load(tmp_path)
        dup_lesson = Lesson(label="2", prose="Duplicate.", kscript=[])
        with pytest.raises(ValueError, match="Duplicate label"):
            doc.amend("append", lesson=dup_lesson)

    def test_amend_raises_when_no_source_path(self) -> None:
        """CRS-22: Amendment raises ValueError when source_path is None."""
        doc = CurriculumDocument.from_string(VALID_TWO_LESSONS)
        new_lesson = Lesson(label="3", prose="Third.", kscript=[])
        with pytest.raises(ValueError, match="no source_path"):
            doc.amend("append", lesson=new_lesson)

    def test_amend_writes_to_file(self, tmp_path: Path) -> None:
        """CRS-23: Amendment writes updated markdown to the source file."""
        doc = self._write_and_load(tmp_path)
        new_lesson = Lesson(label="3", prose="Appended.", kscript=["X > Y"])
        doc.amend("append", lesson=new_lesson)
        # Re-read from file
        assert doc.source_path is not None
        content = doc.source_path.read_text()
        assert "### 3" in content
        assert "Appended." in content
        assert "X > Y" in content

    def test_amend_invalid_action(self, tmp_path: Path) -> None:
        """Amendment raises ValueError for unknown action."""
        doc = self._write_and_load(tmp_path)
        with pytest.raises(ValueError, match="Unknown amend action"):
            doc.amend("delete", label="1")

    def test_amend_modify_with_new_label(self, tmp_path: Path) -> None:
        """Modify can change the label if it doesn't collide."""
        doc = self._write_and_load(tmp_path)
        modified = Lesson(label="1b", prose="Renamed.", kscript=[])
        doc.amend("modify", label="1", lesson=modified)
        assert doc.all_labels() == ["1b", "2"]

    def test_amend_modify_new_label_collides(self, tmp_path: Path) -> None:
        """Modify with a new label that collides raises ValueError."""
        doc = self._write_and_load(tmp_path)
        modified = Lesson(label="2", prose="Collision.", kscript=[])
        with pytest.raises(ValueError, match="Duplicate label"):
            doc.amend("modify", label="1", lesson=modified)


# ── Serialization round-trip ─────────────────────────────────────────


class TestRoundTrip:
    """Verify serialization preserves document structure."""

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "roundtrip.md"
        path.write_text(VALID_CURRICULUM)
        doc1 = CurriculumDocument.from_file(path)

        # Amend to trigger write-back
        new_lesson = Lesson(label="4", prose="New lesson.", kscript=["Z = W"])
        doc1.amend("append", lesson=new_lesson)

        doc2 = CurriculumDocument.from_file(path)
        assert doc2.objective == doc1.objective
        assert doc2.approach == doc1.approach
        assert len(doc2.lessons) == 5
        assert doc2.lessons[4].label == "4"
        assert doc2.lessons[4].kscript == ["Z = W"]
