"""Tests for the LLMSupervisor — the LLM decider supervisor participant.

Validates the decision core (SD-2: the LLMSupervisor resolves ratify_request
via a Cogitator and answers via supervisor_decision) and the
LLMSupervisor-pipeline wiring (SD-16…21: misfit summaries decompiled,
curriculum context lifted). The Cogitator and display helpers are injected
fakes so these tests run without tokenizer data or a live LLM.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.supervisors.llm_supervisor import LLMSupervisor, _to_kvalue

# ── Helpers ───────────────────────────────────────────────────────────


def _frame(
    *,
    proposal_sig: int = 0x0F,
    query_sig: int = 0xFF,
    misfit: dict | None = None,
    curriculum_context: object = None,
) -> dict:
    """Build an enriched ratify_request frame message (wire-dict shape)."""
    return {
        "proposal": {"signature": proposal_sig, "nodes": [0x01, 0x02], "significance": 100},
        "query": {"signature": query_sig, "nodes": [0x10]},
        "significance": 100,
        "misfit": misfit
        if misfit is not None
        else {"underfit": True, "overfit": False, "underfit_gap": 0x1, "overfit_mask": 0},
        "curriculum_context": curriculum_context,
    }


def _make_supervisor(cogitate_result) -> LLMSupervisor:
    """Build an LLMSupervisor with a fake Cogitator returning cogitate_result."""
    cogitator = MagicMock()
    cogitator.cogitate.return_value = cogitate_result
    # Fake display helpers — return a fixed decompiled string, never raise.
    fake_tok = MagicMock()
    fake_sig = MagicMock()
    return LLMSupervisor(
        "ws://localhost:8765",
        cogitator,
        tokenizer=fake_tok,
        signifier=fake_sig,
    )


# ── decide(): decision mapping (SD-2) ─────────────────────────────────


class TestDecideMapping:
    """The CogitationResult → supervisor_decision mapping."""

    def test_scaffold_when_adequate_confidence(self) -> None:
        """Adequate scaffolding (confidence >= threshold) → scaffold answer."""
        from training.supervisors.llm_supervisor import CogitationResult

        result = CogitationResult(
            scaffolding="M > H",
            confidence=0.85,
            reasoning="bridge",
            raw_response=None,
        )
        sup = _make_supervisor(result)
        frame = _frame()

        decision = sup.decide(frame)

        assert decision["decision"] == "scaffold"
        assert decision["text"] == "M > H"
        # Proposal passed through verbatim (the exact kline Kalvin proposed).
        assert decision["proposal"] == frame["proposal"]

    def test_continue_when_low_confidence(self) -> None:
        """Low confidence (< threshold) → continue (decider cannot help)."""
        from training.supervisors.llm_supervisor import CogitationResult

        result = CogitationResult(
            scaffolding="M > H",
            confidence=0.2,  # below ESCALATION_THRESHOLD (0.5)
            reasoning="uncertain",
            raw_response=None,
        )
        sup = _make_supervisor(result)

        decision = sup.decide(_frame())

        assert decision["decision"] == "continue"
        assert "text" not in decision  # no scaffolding carried

    def test_continue_when_no_scaffolding(self) -> None:
        """No scaffolding produced → continue."""
        from training.supervisors.llm_supervisor import CogitationResult

        result = CogitationResult(
            scaffolding=None,
            confidence=0.0,
            reasoning="failed",
            raw_response=None,
        )
        sup = _make_supervisor(result)

        decision = sup.decide(_frame())

        assert decision["decision"] == "continue"
        assert "text" not in decision

    def test_decide_passes_proposal_through_verbatim(self) -> None:
        """The proposal is carried unchanged so the Trainer countersigns the
        exact kline Kalvin proposed (no reconstruction drift)."""
        from training.supervisors.llm_supervisor import CogitationResult

        result = CogitationResult("M > H", 0.9, "ok", None)
        sup = _make_supervisor(result)
        frame = _frame(proposal_sig=0xAB)

        decision = sup.decide(frame)

        assert decision["proposal"] is frame["proposal"]
        assert decision["proposal"]["signature"] == 0xAB

    def test_decide_calls_cogitator_once(self) -> None:
        """One ratify_request → one cogitate call (single-shot; the Trainer
        gates between requests, so there is no multi-round within a request)."""
        from training.supervisors.llm_supervisor import CogitationResult

        cogitator = MagicMock()
        cogitator.cogitate.return_value = CogitationResult("M > H", 0.9, "ok", None)
        sup = LLMSupervisor(
            "ws://localhost:8765",
            cogitator,
            tokenizer=MagicMock(),
            signifier=MagicMock(),
        )

        sup.decide(_frame())

        assert cogitator.cogitate.call_count == 1


# ── _build_request(): pipeline wiring (SD-16…21) ─────────────────────


class TestBuildRequest:
    """The frame → CogitationRequest reconstruction."""

    def test_lifts_structured_curriculum_context(self) -> None:
        """A structured curriculum_context dict populates objective/approach/lesson_prose."""
        from training.supervisors.llm_supervisor import CogitationRequest

        sup = _make_supervisor(None)
        frame = _frame(
            curriculum_context={
                "objective": "Teach X",
                "approach": "stepwise",
                "lesson_prose": "lesson 1 prose",
            }
        )

        request = sup._build_request(frame)

        assert isinstance(request, CogitationRequest)
        assert request.objective == "Teach X"
        assert request.approach == "stepwise"
        assert request.lesson_prose == "lesson 1 prose"

    def test_legacy_string_curriculum_context(self) -> None:
        """A legacy string curriculum_context goes into curriculum_context."""
        sup = _make_supervisor(None)
        frame = _frame(curriculum_context="some flat context string")

        request = sup._build_request(frame)

        assert request.curriculum_context == "some flat context string"
        assert request.objective == ""

    def test_misfit_lifted_into_misfitinfo(self) -> None:
        """The frame's misfit dict becomes a MisfitInfo on the request."""
        sup = _make_supervisor(None)
        frame = _frame(
            misfit={"underfit": True, "overfit": True, "underfit_gap": 0x10, "overfit_mask": 0x20}
        )

        request = sup._build_request(frame)

        assert len(request.misfits) == 1
        info = request.misfits[0]
        assert info.underfit is True
        assert info.overfit is True
        assert info.underfit_gap == 0x10
        assert info.overfit_mask == 0x20

    def test_proposal_and_query_reconstructed_as_kvalues(self) -> None:
        """The wire-dict proposal/query are reconstructed into KValues on the event."""
        sup = _make_supervisor(None)
        frame = _frame(proposal_sig=0x0F, query_sig=0xFF)

        request = sup._build_request(frame)

        event = request.events[0]
        assert event.proposal.kline.signature == 0x0F
        assert event.proposal.kline.nodes == [0x01, 0x02]
        assert event.proposal.significance == 100
        assert event.query.kline.signature == 0xFF

    def test_misfit_summaries_are_decompiled(self) -> None:
        """RS-2 / SD-18: misfit summaries use the decompiled KScript (here the
        injected fake kline_display output), not hex repr."""
        from unittest.mock import patch

        sup = _make_supervisor(None)

        def fake_display(kline, tok, sig):
            return f"KSCRIPT<{kline.signature:#x}>"

        with patch("training.supervisors.llm_supervisor.kline_display", side_effect=fake_display):
            request = sup._build_request(_frame(proposal_sig=0x0F, query_sig=0xFF))

        info = request.misfits[0]
        assert info.expectation_summary == "KSCRIPT<0xff>"
        assert info.proposal_summary == "KSCRIPT<0xf>"

    def test_misfit_summary_falls_back_on_decompile_failure(self) -> None:
        """RS-2 / SD-19: when decompilation fails, summaries fall back to repr."""
        from unittest.mock import patch

        sup = _make_supervisor(None)

        def failing_display(kline, tok, sig):
            raise RuntimeError("no tokenizer data")

        with patch(
            "training.supervisors.llm_supervisor.kline_display",
            side_effect=failing_display,
        ):
            request = sup._build_request(_frame())

        info = request.misfits[0]
        # repr-based fallback — contains the signature hex.
        assert (
            "0x" in info.expectation_summary
            or "signature" in info.expectation_summary.lower()
        )


# ── _to_kvalue wire-shape acceptance ─────────────────────────────────


class TestToKvalue:
    """_to_kvalue accepts the KValue and KLine wire shapes."""

    def test_kvalue_wire_shape(self) -> None:
        kv = _to_kvalue({"signature": 5, "nodes": [1, 2], "significance": 99})
        assert kv.kline.signature == 5
        assert kv.kline.nodes == [1, 2]
        assert kv.significance == 99

    def test_kline_wire_shape_defaults_significance(self) -> None:
        kv = _to_kvalue({"signature": 5, "nodes": [1, 2]})
        assert kv.significance == 0

    def test_passes_through_kvalue(self) -> None:
        original = KValue(KLine(7, [3]), 42)
        assert _to_kvalue(original) is original
