"""Tests for reactive scaffolding submission.

Spec criteria live in the cogitator spec §Reactive Scaffolding Submission
(test matrix AGT-49 through AGT-57). Validates the fixes that ensure
LLM-generated scaffolding is compiled, sanitised, and submitted to Kalvin
instead of discarded.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from training.harness.bus import MessageBus
from training.harness.constants import TRAINEE_ROLE
from training.harness.message import Message
from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from tests.conftest import requires_nlp_data
from training.trainer.cogitation import (
    _SYSTEM_PROMPT,
    Cogitator,
    LLMResponse,
    _strip_hash_comments,
)
from training.trainer.curriculum import CurriculumState
from training.trainer.reactor import Reactor

# ── AGT-49: System prompt contains no hex literal syntax ────────────


class TestSystemPrompt:
    """Validate the corrected system prompt."""

    def test_system_prompt_no_hex(self):
        """AGT-49: Prompt must not instruct LLM to use hex literals."""
        # The prompt should not tell the LLM to USE hex — it may
        # mention hex in a "never use" warning, which is correct.
        lines = _SYSTEM_PROMPT.split("\n")
        # No line should say identifiers ARE hex or use 0x prefix
        for line in lines:
            lower = line.lower()
            if "0x" in line:
                # Only allowed in a "never use" or "do not" context
                assert "never" in lower or "not" in lower or "do not" in lower, (
                    f"Line mentions 0x without a prohibition: {line}"
                )
        # The old claim 'Signatures and nodes are hexadecimal integers' must be gone
        assert "Signatures and nodes are hexadecimal" not in _SYSTEM_PROMPT

    def test_system_prompt_no_invalid_operators(self):
        """AGT-50: Prompt must not list ~>, <-, -> as valid operators."""
        # The prompt should warn AGAINST these, not endorse them
        # Check the syntax overview section doesn't list them as valid
        lines = _SYSTEM_PROMPT.split("\n")
        in_overview = False
        for line in lines:
            if "syntax overview" in line.lower():
                in_overview = True
                continue
            if in_overview and line.startswith("- "):
                assert "~>" not in line, f"Syntax overview lists ~> as valid: {line}"
                assert "<-" not in line, f"Syntax overview lists <- as valid: {line}"
                assert "->" not in line, f"Syntax overview lists -> as valid: {line}"
            elif in_overview and not line.startswith("-") and not line.startswith(" "):
                in_overview = False
        # The explicit prohibition must be present
        assert "Do not use" in _SYSTEM_PROMPT

    def test_system_prompt_has_valid_operators(self):
        """Prompt must list the actual valid operators."""
        assert "==" in _SYSTEM_PROMPT  # countersign
        assert "=>" in _SYSTEM_PROMPT  # canonize


# ── AGT-51, AGT-52: Hash comment stripping ─────────────────────────


class TestStripHashComments:
    """Validate _strip_hash_comments utility."""

    def test_strips_hash_comments(self):
        """AGT-51: Lines starting with # are removed."""
        source = "# This is a comment\nM > H\n# Another comment\nH => M"
        result = _strip_hash_comments(source)
        assert result == "M > H\nH => M"

    def test_all_comments_returns_empty(self):
        """AGT-52: All-comment input returns empty string."""
        source = "# comment 1\n# comment 2\n# comment 3"
        result = _strip_hash_comments(source)
        assert result == ""

    def test_preserves_kscript(self):
        """AGT-51: Valid KScript lines are preserved unchanged."""
        source = "M > H\nMH => H A\nH == M"
        result = _strip_hash_comments(source)
        assert result == source

    def test_removes_blank_lines(self):
        """Blank lines are also removed."""
        source = "M > H\n\n\nH => M"
        result = _strip_hash_comments(source)
        assert result == "M > H\nH => M"

    def test_hash_midline_preserved(self):
        """Lines with # not at start are preserved."""
        source = "M (comment with # inside) > H"
        result = _strip_hash_comments(source)
        assert "M" in result


# ── AGT-51, AGT-53: Cogitator sanitisation ──────────────────────────


class TestCogitatorSanitisation:
    """Validate that Cogitator strips # comments before compilation."""

    @requires_nlp_data
    def test_cogitator_strips_and_logs(self, caplog):
        """AGT-51: Cogitator logs when # comments are stripped."""
        client = MagicMock()
        client.complete.return_value = LLMResponse(
            content=None,
            tool_calls=[
                {
                    "function": {
                        "name": "submit_scaffolding",
                        "arguments": '{"kscript_source": "# comment\\nM > H", '
                        '"confidence": 0.8, '
                        '"reasoning": "test"}',
                    },
                }
            ],
            finish_reason="tool_calls",
        )

        cogitator = Cogitator(client=client)
        from training.trainer.cogitation import CogitationRequest, MisfitInfo

        event = RationaliseEvent(
            kind="frame",
            query=KLine(signature=0x1, nodes=[]),
            proposal=KLine(signature=0x2, nodes=[]),
            significance=100,
        )
        misfit = MisfitInfo(
            underfit=True,
            overfit=False,
            underfit_gap=0x1,
            overfit_mask=0,
            expectation_summary="M",
            proposal_summary="H",
        )
        request = CogitationRequest(
            events=[event],
            misfits=[misfit],
            curriculum_context="",
            conversation_history=[],
            round_number=1,
            max_rounds=3,
        )

        with caplog.at_level(logging.INFO, logger="training.trainer.cogitation"):
            result = cogitator.cogitate(request)

        assert result.scaffolding == "M > H"
        assert any("stripped # comments" in r.message for r in caplog.records)

    def test_cogitator_all_comments_returns_none(self):
        """AGT-53: Cogitator returns None when scaffolding is all comments."""
        client = MagicMock()
        client.complete.return_value = LLMResponse(
            content=None,
            tool_calls=[
                {
                    "function": {
                        "name": "submit_scaffolding",
                        "arguments": '{"kscript_source": "# only comments\\n# no kscript", '
                        '"confidence": 0.5, '
                        '"reasoning": "test"}',
                    },
                }
            ],
            finish_reason="tool_calls",
        )

        cogitator = Cogitator(client=client)
        from training.trainer.cogitation import CogitationRequest, MisfitInfo

        event = RationaliseEvent(
            kind="frame",
            query=KLine(signature=0x1, nodes=[]),
            proposal=KLine(signature=0x2, nodes=[]),
            significance=100,
        )
        misfit = MisfitInfo(
            underfit=True,
            overfit=False,
            underfit_gap=0x1,
            overfit_mask=0,
            expectation_summary="M",
            proposal_summary="H",
        )
        request = CogitationRequest(
            events=[event],
            misfits=[misfit],
            curriculum_context="",
            conversation_history=[],
            round_number=1,
            max_rounds=3,
        )

        result = cogitator.cogitate(request)
        assert result.scaffolding is None


# ── AGT-57: Reactor log line ────────────────────────────────────────


class _BusCapture:
    """Capture bus messages for test assertions."""

    def __init__(self, bus: MessageBus) -> None:
        self.messages: list[Message] = []
        self._bus = bus
        self._original = bus.send

    def install(self) -> None:
        cap = self

        def _send(msg: Message) -> None:
            cap.messages.append(msg)

        self._bus.send = _send  # type: ignore[assignment]

    def find_all(self, role: str, action: str) -> list[Message]:
        return [m for m in self.messages if m.role == role and m.action == action]


class TestReactorSubmittedLog:
    """AGT-57: Reactor logs 'submitted reactive scaffolding'."""

    def test_reactor_submitted_log_line(self, caplog):
        """Reactor logs confirmation after sending reactive scaffolding."""
        bus = MessageBus()
        cap = _BusCapture(bus)
        cap.install()

        state = MagicMock(spec=CurriculumState)
        state.log_event = MagicMock()
        state.curriculum = MagicMock()
        state.curriculum.position = 0

        def cogitate_fn(event):
            return ("M > H", 0.8)

        reactor = Reactor(
            bus,
            state,
            role="trainer",
            max_reactive_rounds=5,
            cogitate_fn=cogitate_fn,
        )

        event = RationaliseEvent(
            kind="frame",
            query=KLine(signature=0x1, nodes=[]),
            proposal=KLine(signature=0x2, nodes=[]),
            significance=100,
        )

        with caplog.at_level(logging.INFO, logger="training.trainer.reactor"):
            reactor.process_s2_s3(event)

        # Check log line
        assert any("submitted reactive scaffolding" in r.message for r in caplog.records)

        # Check bus message was sent to trainee with submit action
        submits = cap.find_all(TRAINEE_ROLE, "submit")
        assert len(submits) == 1
        assert submits[0].message == "M > H"
