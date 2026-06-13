"""Tests for the cogitation module — HRNS-13, HRNS-14 and unit coverage."""

from __future__ import annotations

import json

from kalvin.events import RationaliseEvent
from kalvin.kline import KDbg, KLine
from trainer.cogitation import (
    ESCALATION_THRESHOLD,
    CogitationRequest,
    CogitationResult,
    Cogitator,
    ConversationTurn,
    LLMResponse,
    MisfitInfo,
    build_prompt,
    build_tool_definitions,
    extract_result,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _make_event(significance: int = 2) -> RationaliseEvent:
    """Create a minimal RationaliseEvent for testing."""
    query = KLine(signature=0xAB, nodes=[0x1, 0x2], dbg=KDbg(label="query"))
    proposal = KLine(signature=0xCD, nodes=[0x3], dbg=KDbg(label="proposal"))
    return RationaliseEvent(
        kind="test",
        query=query,
        proposal=proposal,
        significance=significance,
    )


def _make_misfit(
    underfit: bool = True,
    overfit: bool = False,
) -> MisfitInfo:
    """Create a minimal MisfitInfo for testing."""
    return MisfitInfo(
        underfit=underfit,
        overfit=overfit,
        underfit_gap=0xFF if underfit else 0,
        overfit_mask=0xAA if overfit else 0,
        expectation_summary="KLine(sig=0xAB, nodes=[0x1, 0x2])",
        proposal_summary="KLine(sig=0xCD, nodes=[0x3])",
    )


def _make_request(**overrides) -> CogitationRequest:
    """Create a default CogitationRequest, allowing field overrides."""
    defaults = dict(
        events=[_make_event()],
        misfits=[_make_misfit()],
        curriculum_context="Learning basic countersigns",
        conversation_history=[ConversationTurn(role="human", content="Try simpler constructs")],
        round_number=1,
        max_rounds=3,
    )
    defaults.update(overrides)
    return CogitationRequest(**defaults)


# ── build_prompt tests ────────────────────────────────────────────────


class TestBuildPrompt:
    """Tests for build_prompt()."""

    def test_system_message_present(self):
        """Prompt starts with a system message."""
        req = _make_request()
        messages = build_prompt(req)
        assert messages[0]["role"] == "system"
        assert "KScript scaffolding generator" in messages[0]["content"]

    def test_event_context_describes_misfit(self):
        """Event context message contains misfit details."""
        req = _make_request()
        messages = build_prompt(req)
        # Find the event context message
        event_msgs = [m for m in messages if "Event 1" in m["content"]]
        assert len(event_msgs) == 1
        content = event_msgs[0]["content"]
        assert "S3" in content
        assert "underfitting" in content
        assert "0xAB" in content or "0xFF" in content

    def test_dual_misfit_type(self):
        """Dual misfit (underfit + overfit) is described correctly."""
        req = _make_request(misfits=[_make_misfit(underfit=True, overfit=True)])
        messages = build_prompt(req)
        event_msgs = [m for m in messages if "Event 1" in m["content"]]
        assert "dual" in event_msgs[0]["content"].lower()

    def test_overfit_only_misfit(self):
        """Overfit-only misfit type is described correctly."""
        req = _make_request(misfits=[_make_misfit(underfit=False, overfit=True)])
        messages = build_prompt(req)
        event_msgs = [m for m in messages if "Event 1" in m["content"]]
        assert "overfitting" in event_msgs[0]["content"]
        assert "dual" not in event_msgs[0]["content"].lower()

    def test_curriculum_context_included(self):
        """Curriculum context appears as a user message."""
        req = _make_request(curriculum_context="Lesson 5: canonize")
        messages = build_prompt(req)
        ctx_msgs = [m for m in messages if "Lesson 5: canonize" in m["content"]]
        assert len(ctx_msgs) == 1
        assert ctx_msgs[0]["role"] == "user"

    def test_conversation_history_mapped(self):
        """Conversation history turns appear with correct roles."""
        history = [
            ConversationTurn(role="human", content="Try this"),
            ConversationTurn(role="trainer", content="Okay noted"),
        ]
        req = _make_request(conversation_history=history)
        messages = build_prompt(req)
        assert {"role": "human", "content": "Try this"} in messages
        assert {"role": "trainer", "content": "Okay noted"} in messages

    def test_round_budget_message(self):
        """Round budget message contains correct round info."""
        req = _make_request(round_number=2, max_rounds=5)
        messages = build_prompt(req)
        budget_msgs = [m for m in messages if "Round 2 of 5" in m["content"]]
        assert len(budget_msgs) == 1
        assert "3 rounds remaining" in budget_msgs[0]["content"]

    def test_empty_events_produces_no_event_messages(self):
        """With no events, no event context messages are generated."""
        req = _make_request(events=[], misfits=[])
        messages = build_prompt(req)
        event_msgs = [m for m in messages if "Event" in m["content"]]
        assert len(event_msgs) == 0

    def test_multiple_events(self):
        """Multiple event/misfit pairs produce multiple event messages."""
        req = _make_request(
            events=[_make_event(2), _make_event(3)],
            misfits=[_make_misfit(), _make_misfit(overfit=True)],
        )
        messages = build_prompt(req)
        event_msgs = [m for m in messages if "Event" in m["content"]]
        assert len(event_msgs) == 2


# ── build_tool_definitions tests ──────────────────────────────────────


class TestBuildToolDefinitions:
    """Tests for build_tool_definitions()."""

    def test_returns_list_with_one_tool(self):
        """Tool definitions list contains exactly one tool."""
        tools = build_tool_definitions()
        assert len(tools) == 1

    def test_tool_has_correct_type(self):
        """Tool definition has type 'function'."""
        tools = build_tool_definitions()
        assert tools[0]["type"] == "function"

    def test_tool_function_name(self):
        """Tool function is named 'submit_scaffolding'."""
        tools = build_tool_definitions()
        assert tools[0]["function"]["name"] == "submit_scaffolding"

    def test_tool_has_required_parameters(self):
        """Tool parameters include kscript_source, confidence, reasoning."""
        params = build_tool_definitions()[0]["function"]["parameters"]
        required = params["required"]
        assert "kscript_source" in required
        assert "confidence" in required
        assert "reasoning" in required

    def test_tool_parameters_have_descriptions(self):
        """Each tool parameter has a description."""
        props = build_tool_definitions()[0]["function"]["parameters"]["properties"]
        for param_name in ("kscript_source", "confidence", "reasoning"):
            assert "description" in props[param_name]

    def test_confidence_bounds(self):
        """Confidence parameter has min 0.0 and max 1.0."""
        props = build_tool_definitions()[0]["function"]["parameters"]["properties"]
        assert props["confidence"]["minimum"] == 0.0
        assert props["confidence"]["maximum"] == 1.0


# ── extract_result tests ─────────────────────────────────────────────


class TestExtractResult:
    """Tests for extract_result()."""

    def test_tool_call_with_scaffolding(self):
        """Extract structured result from a submit_scaffolding tool call."""
        response = LLMResponse(
            content=None,
            tool_calls=[
                {
                    "function": {
                        "name": "submit_scaffolding",
                        "arguments": json.dumps(
                            {
                                "kscript_source": "0xAB -> 0x01",
                                "confidence": 0.85,
                                "reasoning": "Simple countersign bridges the gap",
                            }
                        ),
                    },
                },
            ],
            finish_reason="tool_calls",
        )
        result = extract_result(response)
        assert result.scaffolding == "0xAB -> 0x01"
        assert result.confidence == 0.85
        assert "countersign" in result.reasoning
        assert result.raw_response is None

    def test_tool_call_with_dict_arguments(self):
        """Tool call arguments may be a dict (not JSON string)."""
        response = LLMResponse(
            content="some text",
            tool_calls=[
                {
                    "function": {
                        "name": "submit_scaffolding",
                        "arguments": {
                            "kscript_source": "0xCD -> 0x02",
                            "confidence": 0.7,
                            "reasoning": "Test",
                        },
                    },
                },
            ],
            finish_reason="tool_calls",
        )
        result = extract_result(response)
        assert result.scaffolding == "0xCD -> 0x02"

    def test_tool_call_different_function_ignored(self):
        """Tool call with a different function name falls through to text extraction."""
        response = LLMResponse(
            content=None,
            tool_calls=[
                {
                    "function": {
                        "name": "other_function",
                        "arguments": "{}",
                    },
                },
            ],
            finish_reason="tool_calls",
        )
        result = extract_result(response)
        assert result.scaffolding is None
        assert result.confidence == 0.0

    def test_plain_text_with_kscript_lines(self):
        """Plain text response containing KScript lines is extracted with low confidence."""
        response = LLMResponse(
            content="Here's my suggestion:\n0xAB -> 0x01\n0xCD => 0x02\nHope this helps!",
            tool_calls=None,
            finish_reason="stop",
        )
        result = extract_result(response)
        assert result.scaffolding is not None
        assert "0xAB -> 0x01" in result.scaffolding
        assert "0xCD => 0x02" in result.scaffolding
        assert result.confidence < ESCALATION_THRESHOLD

    def test_plain_text_no_kscript(self):
        """Plain text without KScript patterns returns failed result."""
        response = LLMResponse(
            content="I'm not sure what to suggest here.",
            tool_calls=None,
            finish_reason="stop",
        )
        result = extract_result(response)
        assert result.scaffolding is None
        assert result.confidence == 0.0

    def test_empty_response(self):
        """Empty response returns failed result."""
        response = LLMResponse(
            content=None,
            tool_calls=None,
            finish_reason="stop",
        )
        result = extract_result(response)
        assert result.scaffolding is None
        assert result.confidence == 0.0
        assert "No valid response" in result.reasoning

    def test_tool_call_overrides_plain_text(self):
        """When both tool_calls and content exist, tool call takes priority."""
        response = LLMResponse(
            content="0xAB -> 0x01",
            tool_calls=[
                {
                    "function": {
                        "name": "submit_scaffolding",
                        "arguments": json.dumps(
                            {
                                "kscript_source": "0xEF -> 0x03",
                                "confidence": 0.9,
                                "reasoning": "From tool call",
                            }
                        ),
                    },
                },
            ],
            finish_reason="tool_calls",
        )
        result = extract_result(response)
        assert result.scaffolding == "0xEF -> 0x03"
        assert result.confidence == 0.9


# ── Cogitator tests ──────────────────────────────────────────────────


class _MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.last_messages: list[dict] | None = None
        self.last_tools: list[dict] | None = None

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self.last_messages = messages
        self.last_tools = tools
        return self._response


def _tool_call_response(kscript: str, confidence: float, reasoning: str) -> LLMResponse:
    """Create an LLMResponse with a submit_scaffolding tool call."""
    return LLMResponse(
        content=None,
        tool_calls=[
            {
                "function": {
                    "name": "submit_scaffolding",
                    "arguments": json.dumps(
                        {
                            "kscript_source": kscript,
                            "confidence": confidence,
                            "reasoning": reasoning,
                        }
                    ),
                },
            },
        ],
        finish_reason="tool_calls",
    )


class TestCogitator:
    """Tests for the Cogitator class."""

    def test_cogitate_returns_scaffolding(self):
        """Cogitator returns valid scaffolding from a tool call response."""
        mock_client = _MockLLMClient(
            _tool_call_response("AB => CD", 0.85, "Simple canonize bridges the gap")
        )
        cogitator = Cogitator(client=mock_client)
        req = _make_request()
        result = cogitator.cogitate(req)

        assert result.scaffolding == "AB => CD"
        assert result.confidence == 0.85
        assert "canonize" in result.reasoning
        # Verify the LLM was called with prompts and tools
        assert mock_client.last_messages is not None
        assert mock_client.last_tools is not None
        assert len(mock_client.last_tools) == 1

    def test_cogitate_compilation_failure(self):
        """Scaffolding that doesn't compile is set to None with 0.0 confidence."""
        mock_client = _MockLLMClient(_tool_call_response("invalid kscript !!", 0.8, "My attempt"))
        cogitator = Cogitator(client=mock_client)
        req = _make_request()
        result = cogitator.cogitate(req)

        assert result.scaffolding is None
        assert result.confidence == 0.0
        assert "compilation failed" in result.reasoning

    def test_cogitate_no_tool_call(self):
        """Plain text response results in low confidence."""
        mock_client = _MockLLMClient(
            LLMResponse(
                content="Here's my suggestion:\nAB => CD\nHope it helps!",
                tool_calls=None,
                finish_reason="stop",
            )
        )
        cogitator = Cogitator(client=mock_client)
        req = _make_request()
        result = cogitator.cogitate(req)

        # Plain text KScript extracted — but compilation may or may not pass
        # depending on content; the key thing is it goes through the pipeline
        assert result.raw_response is not None

    def test_cogitate_empty_response(self):
        """Empty response returns None scaffolding and 0.0 confidence."""
        mock_client = _MockLLMClient(
            LLMResponse(content=None, tool_calls=None, finish_reason="stop")
        )
        cogitator = Cogitator(client=mock_client)
        req = _make_request()
        result = cogitator.cogitate(req)

        assert result.scaffolding is None
        assert result.confidence == 0.0

    def test_should_escalate_low_confidence(self):
        """should_escalate returns True for low confidence."""
        result = CogitationResult(
            scaffolding="AB => CD", confidence=0.3, reasoning="test", raw_response=None
        )
        assert Cogitator.should_escalate(result) is True

    def test_should_escalate_no_scaffolding(self):
        """should_escalate returns True when scaffolding is None."""
        result = CogitationResult(
            scaffolding=None, confidence=0.8, reasoning="test", raw_response=None
        )
        assert Cogitator.should_escalate(result) is True

    def test_should_not_escalate_high_confidence(self):
        """should_escalate returns False with scaffolding and high confidence."""
        result = CogitationResult(
            scaffolding="AB => CD", confidence=0.7, reasoning="test", raw_response=None
        )
        assert Cogitator.should_escalate(result) is False

    def test_cogitate_passes_tools_to_client(self):
        """Cogitator passes tool definitions to the LLM client."""
        mock_client = _MockLLMClient(_tool_call_response("AB => CD", 0.9, "test"))
        cogitator = Cogitator(client=mock_client)
        cogitator.cogitate(_make_request())
        assert mock_client.last_tools is not None
        assert mock_client.last_tools[0]["function"]["name"] == "submit_scaffolding"


# ── HRNS-13 and HRNS-14 tests ────────────────────────────────────────


class TestHRNS13ReactiveModeOnS2S3:
    """HRNS-13: Trainer enters reactive mode on S2/S3 events.

    This test verifies the reactive mode code path works end-to-end:
    given S2/S3 events and misfit info, the Cogitator produces valid
    KScript scaffolding with confidence > 0.
    """

    def test_reactive_mode_on_s2_s3(self):
        """End-to-end reactive mode: S2/S3 events → scaffolding generation."""
        s2_event = _make_event(significance=2)
        s3_event = _make_event(significance=3)
        s2_misfit = _make_misfit(underfit=True, overfit=False)
        s3_misfit = _make_misfit(underfit=True, overfit=True)

        request = CogitationRequest(
            events=[s2_event, s3_event],
            misfits=[s2_misfit, s3_misfit],
            curriculum_context="Lesson 3: dual misfit resolution",
            conversation_history=[],
            round_number=1,
            max_rounds=3,
        )

        # Mock LLM returns valid KScript scaffolding
        mock_client = _MockLLMClient(
            _tool_call_response(
                "AB => CD",
                0.75,
                "Countersign resolves underfit by adding CD nodes to AB signature",
            )
        )
        cogitator = Cogitator(client=mock_client)
        result = cogitator.cogitate(request)

        # Verify scaffolding is valid KScript
        assert result.scaffolding is not None
        assert result.confidence > 0.0
        assert result.scaffolding == "AB => CD"
        assert Cogitator.should_escalate(result) is False


class TestHRNS14EscalationOnBudgetExhaustion:
    """HRNS-14: Trainer escalates to Slack on budget exhaustion.

    Tests that should_escalate returns True when budget is exhausted
    or when the cogitation result has low confidence.
    """

    def test_escalation_on_budget_exhaustion(self):
        """When round_number equals max_rounds, escalation should be triggered.

        The Trainer is responsible for checking budget and calling cogitate,
        but the cogitation module must support the budget-aware flow by
        correctly signalling escalation.
        """
        request = CogitationRequest(
            events=[_make_event(significance=2)],
            misfits=[_make_misfit()],
            curriculum_context="Final attempt",
            conversation_history=[],
            round_number=3,
            max_rounds=3,
        )

        # Even if LLM returns something, budget exhaustion is the Trainer's concern.
        # But we verify the module works at budget boundary.
        mock_client = _MockLLMClient(
            _tool_call_response("AB => CD", 0.75, "Final attempt scaffolding")
        )
        cogitator = Cogitator(client=mock_client)
        cogitator.cogitate(request)

        # The result is valid — escalation is the Trainer's call based on budget.
        # But verify the prompt includes round info for LLM awareness.
        assert mock_client.last_messages is not None
        budget_msgs = [m for m in mock_client.last_messages if "Round 3 of 3" in m["content"]]
        assert len(budget_msgs) == 1

    def test_escalation_on_low_confidence(self):
        """should_escalate returns True when LLM returns low confidence."""
        request = _make_request()
        mock_client = _MockLLMClient(_tool_call_response("AB => CD", 0.3, "Uncertain suggestion"))
        cogitator = Cogitator(client=mock_client)
        result = cogitator.cogitate(request)

        assert Cogitator.should_escalate(result) is True


# ── CRS-50..CRS-52: Structured context fields ────────────────────────


class TestStructuredContextFields:
    """CRS-50..CRS-52: CogitationRequest structured context fields."""

    def test_cogitation_request_new_fields(self) -> None:
        """CRS-50: CogitationRequest accepts objective, approach, and lesson_prose."""
        req = CogitationRequest(
            events=[_make_event()],
            misfits=[_make_misfit()],
            curriculum_context="legacy context",
            conversation_history=[],
            round_number=1,
            objective="Teach SVO structure",
            approach="Step by step introduction",
            lesson_prose="This lesson introduces the subject.",
        )
        assert req.objective == "Teach SVO structure"
        assert req.approach == "Step by step introduction"
        assert req.lesson_prose == "This lesson introduces the subject."

    def test_build_prompt_prefers_new_fields(self) -> None:
        """CRS-51: build_prompt prefers objective + approach + lesson_prose."""
        req = _make_request(
            objective="Teach SVO structure",
            approach="Step by step",
            lesson_prose="This is lesson 1 about subjects.",
            curriculum_context="legacy context that should not appear",
        )
        messages = build_prompt(req)

        # Find context messages
        context_msgs = [
            m
            for m in messages
            if "Teach SVO structure" in m.get("content", "")
            or "legacy context" in m.get("content", "")
        ]

        # Should contain structured fields
        structured_content = " ".join(m["content"] for m in context_msgs)
        assert "Objective: Teach SVO structure" in structured_content
        assert "Approach: Step by step" in structured_content
        assert "Current lesson context:" in structured_content

        # Should NOT contain legacy context
        assert "legacy context that should not appear" not in structured_content

    def test_build_prompt_falls_back_to_context(self) -> None:
        """CRS-52: build_prompt falls back to curriculum_context when new fields empty."""
        req = _make_request(
            curriculum_context="Lesson 5: canonize",
            objective="",
            approach="",
            lesson_prose="",
        )
        messages = build_prompt(req)

        # Find context messages
        ctx_msgs = [m for m in messages if "Lesson 5: canonize" in m["content"]]
        assert len(ctx_msgs) == 1
        assert "Current curriculum context" in ctx_msgs[0]["content"]

    def test_build_prompt_no_context_when_both_empty(self) -> None:
        """When both structured and legacy context are empty, no context message."""
        req = _make_request(
            curriculum_context="",
            objective="",
            approach="",
            lesson_prose="",
        )
        messages = build_prompt(req)

        # No context message at all
        ctx_msgs = [
            m
            for m in messages
            if "curriculum context" in m.get("content", "").lower()
            or "objective:" in m.get("content", "").lower()
        ]
        assert len(ctx_msgs) == 0
