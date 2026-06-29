"""Tests for CurriculumGenerator — CRS-32 through CRS-37."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from training.harness.llm import LLMResponse
from training.trainer.curriculum_document import CurriculumDocument
from training.trainer.curriculum_generator import (
    CurriculumGenerationError,
    CurriculumGenerator,
    _slug_from_goal,
)

# ── Fixtures ──────────────────────────────────────────────────────────

VALID_CURRICULUM_MARKDOWN = textwrap.dedent("""\
    ## Objective

    Teach basic KScript patterns.

    ## Approach

    Introduce one concept at a time.

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

INVALID_MARKDOWN = textwrap.dedent("""\
    ## Objective

    Missing approach and lessons.
""")


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self._last_messages: list[dict] | None = None

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_messages(self) -> list[dict] | None:
        return self._last_messages

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self._call_count += 1
        self._last_messages = messages
        if self._call_count <= len(self._responses):
            return self._responses[self._call_count - 1]
        return LLMResponse(content=None, tool_calls=None, finish_reason="stop")


# ── CRS-32: Generator makes one LLM call ─────────────────────────────


class TestGenerateLLMCall:
    """CRS-32: Generator makes one LLM call with curriculum format system prompt."""

    def test_generate_makes_llm_call(self, tmp_path: Path) -> None:
        client = MockLLMClient(
            [
                LLMResponse(
                    content=VALID_CURRICULUM_MARKDOWN, tool_calls=None, finish_reason="stop"
                ),
            ]
        )
        gen = CurriculumGenerator(client, tmp_path / "curricula")
        gen.generate("basic patterns")
        assert client.call_count == 1
        # System prompt mentions curriculum format
        messages = client.last_messages
        assert messages is not None
        assert messages[0]["role"] == "system"
        assert "## Objective" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "basic patterns" in messages[1]["content"]


# ── CRS-33: Generator parses LLM response ─────────────────────────────


class TestGenerateParseResponse:
    """CRS-33: Generator parses LLM response via from_string and validates."""

    def test_generate_parses_response(self, tmp_path: Path) -> None:
        client = MockLLMClient(
            [
                LLMResponse(
                    content=VALID_CURRICULUM_MARKDOWN, tool_calls=None, finish_reason="stop"
                ),
            ]
        )
        gen = CurriculumGenerator(client, tmp_path / "curricula")
        result = gen.generate("basic patterns")
        assert result.exists()
        # Verify the file is parseable
        doc = CurriculumDocument.from_file(result)
        assert "Teach basic KScript" in doc.objective
        assert len(doc.lessons) == 2


# ── CRS-34: Generator retries on parse failure ────────────────────────


class TestGenerateRetry:
    """CRS-34: Generator retries once on parse failure with error feedback."""

    def test_generate_retries_on_parse_failure(self, tmp_path: Path) -> None:
        client = MockLLMClient(
            [
                # First response: invalid
                LLMResponse(content=INVALID_MARKDOWN, tool_calls=None, finish_reason="stop"),
                # Second response: valid
                LLMResponse(
                    content=VALID_CURRICULUM_MARKDOWN, tool_calls=None, finish_reason="stop"
                ),
            ]
        )
        gen = CurriculumGenerator(client, tmp_path / "curricula")
        result = gen.generate("basic patterns")
        assert client.call_count == 2
        assert result.exists()
        # Verify the retry prompt includes the error
        messages = client.last_messages
        assert messages is not None
        # The third message should mention the error
        error_msgs = [m for m in messages if "invalid" in m.get("content", "").lower()]
        assert len(error_msgs) > 0


# ── CRS-35: Generator raises on second failure ────────────────────────


class TestGenerateSecondFailure:
    """CRS-35: Generator raises CurriculumGenerationError on second failure."""

    def test_generate_raises_on_second_failure(self, tmp_path: Path) -> None:
        client = MockLLMClient(
            [
                # First: invalid
                LLMResponse(content=INVALID_MARKDOWN, tool_calls=None, finish_reason="stop"),
                # Second: also invalid
                LLMResponse(content=INVALID_MARKDOWN, tool_calls=None, finish_reason="stop"),
            ]
        )
        gen = CurriculumGenerator(client, tmp_path / "curricula")
        with pytest.raises(CurriculumGenerationError, match="failed after retry"):
            gen.generate("basic patterns")
        assert client.call_count == 2


# ── CRS-36: Generator writes to file ──────────────────────────────────


class TestGenerateWriteFile:
    """CRS-36: Generator writes validated markdown to curricula/<slug>.md."""

    def test_generate_writes_to_file(self, tmp_path: Path) -> None:
        curricula_dir = tmp_path / "curricula"
        client = MockLLMClient(
            [
                LLMResponse(
                    content=VALID_CURRICULUM_MARKDOWN, tool_calls=None, finish_reason="stop"
                ),
            ]
        )
        gen = CurriculumGenerator(client, curricula_dir)
        result = gen.generate("basic patterns")
        assert result.parent == curricula_dir
        assert result.suffix == ".md"
        assert curricula_dir.exists()
        # Content is parseable
        doc = CurriculumDocument.from_file(result)
        assert doc.lessons[0].kscript == ["A = B"]


# ── CRS-37: Slug from goal ───────────────────────────────────────────


class TestSlugFromGoal:
    """CRS-37: Slug derived from goal (lowercase, hyphens, non-alphanumeric stripped)."""

    def test_generate_slug_from_goal(self, tmp_path: Path) -> None:
        client = MockLLMClient(
            [
                LLMResponse(
                    content=VALID_CURRICULUM_MARKDOWN, tool_calls=None, finish_reason="stop"
                ),
            ]
        )
        gen = CurriculumGenerator(client, tmp_path / "curricula")
        result = gen.generate("Mary Had a Little Lamb!")
        assert result.name == "mary-had-a-little-lamb.md"

    def test_slug_simple(self) -> None:
        assert _slug_from_goal("Basic patterns") == "basic-patterns"

    def test_slug_special_chars(self) -> None:
        assert _slug_from_goal("Hello, World! #1") == "hello-world-1"

    def test_slug_truncation(self) -> None:
        long_goal = "a" * 100
        assert len(_slug_from_goal(long_goal)) == 60

    def test_slug_leading_trailing_stripped(self) -> None:
        assert _slug_from_goal("---hello---") == "hello"
