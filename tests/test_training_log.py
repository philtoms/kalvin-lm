"""Tests for training log output.

Verifies that the Trainer, Reactor, and Adapter emit structured log
messages at the correct levels during training operations.

Spec ref: specs/training-log.md TL-1 through TL-20
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from kalvin.events import RationaliseEvent
from kalvin.expand import D_MAX, SIG_S1
from kalvin.kline import KDbg, KLine
from kalvin.kvalue import KValue
from tests.conftest import requires_tokenizer_data
from training.harness.adapter import KAgentAdapter
from training.harness.bus import MessageBus
from training.harness.constants import TRAINEE_ROLE, TRAINER_ROLE
from training.harness.message import Message
from training.trainer.curriculum import Curriculum
from training.trainer.curriculum_document import CurriculumDocument, Lesson
from training.trainer.reactor import Reactor
from training.trainer.trainer import Trainer

# ── Significance constants ────────────────────────────────────────────

_S1_SIGNIFICANCE = D_MAX  # S1 threshold (distance 0)
_S2_SIGNIFICANCE = (~100) & 0xFFFF_FFFF_FFFF_FFFF  # S2 at distance 100
_S2_DISTANCE = 100  # raw distance for assertions


# ── Helpers ──────────────────────────────────────────────────────────


def _make_entry(sig: int, nodes: list[int]) -> KValue:
    return KValue(
        KLine(signature=sig, nodes=nodes, dbg=KDbg(label=f"test-{sig:#x}")),
        SIG_S1,
    )


def _make_event(
    kind: str,
    query: KLine,
    proposal: KLine | None = None,
    significance: int = 0,
) -> RationaliseEvent:
    proposal_kline = proposal if proposal is not None else query
    return RationaliseEvent(
        kind=kind,
        query=KValue(query, significance),
        proposal=KValue(proposal_kline, significance),
    )


def _entry_key(value: KValue) -> tuple[int, tuple[int, ...]]:
    return (value.kline.signature, tuple(value.kline.nodes))


def _drain(trainer: Trainer) -> None:
    """Simulate the cogitator drain completing.

    The Trainer sends a ``drain`` message before each lesson and only
    compiles/submits the lesson once a ``drained`` reply arrives (see
    ``_submit_next_lesson`` / ``_handle_drained``).  In the unit tests
    there is no real cogitator, so we inject the reply directly.
    """
    trainer.on_message(Message(role="adapter", action="drained", message=None))


class BusCapture:
    """Captures messages sent via bus.send()."""

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self.messages: list[Message] = []
        self._original_send = bus.send

        def capturing_send(msg: Message) -> None:
            self.messages.append(msg)
            self._original_send(msg)

        bus.send = capturing_send  # type: ignore[assignment]


def _make_trainer(
    bus: MessageBus,
    curriculum: Curriculum,
    *,
    cogitate_fn=None,
    curriculum_file: str | None = None,
) -> Trainer:
    return Trainer(
        bus,
        curriculum,
        role=TRAINER_ROLE,
        cogitate_fn=cogitate_fn,
        curriculum_file=curriculum_file,
    )


def _simple_curriculum() -> Curriculum:
    doc = CurriculumDocument(
        objective="test",
        approach="test",
        lessons=[
            Lesson(label="1", prose="test", kscript=["M"]),
            Lesson(label="2", prose="test", kscript=["H"]),
            Lesson(label="3", prose="test", kscript=["M == H"]),
        ],
    )
    return Curriculum(doc)


# ═══════════════════════════════════════════════════════════════════════
# Trainer logging tests
# ═══════════════════════════════════════════════════════════════════════


class TestTrainerLogging:
    """TL-1 through TL-9: Trainer log output."""

    def test_session_start_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-1: Session start logs lesson count and curriculum path."""
        bus = MessageBus()
        curriculum = _simple_curriculum()
        trainer = _make_trainer(bus, curriculum, curriculum_file="curricula/test.md")
        caplog.set_level(logging.INFO, logger="training.trainer.trainer")

        trainer.start_session()

        assert any(
            "Session started" in r.message
            and "3 lessons" in r.message
            and "curricula/test.md" in r.message
            for r in caplog.records
        )

    @requires_tokenizer_data
    def test_lesson_submit_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-2: Lesson submit logs label and progress."""
        bus = MessageBus()
        curriculum = _simple_curriculum()
        trainer = _make_trainer(bus, curriculum, curriculum_file="test.md")
        caplog.set_level(logging.INFO, logger="training.trainer.trainer")

        # Start session triggers lesson submission
        trainer.start_session()
        _drain(trainer)

        assert any("Submitting lesson 1 (1/3)" in r.message for r in caplog.records)

    @requires_tokenizer_data
    def test_lesson_submit_debug_kscript(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-3: Lesson submit logs KScript source at DEBUG level."""
        bus = MessageBus()
        curriculum = _simple_curriculum()
        trainer = _make_trainer(bus, curriculum, curriculum_file="test.md")
        caplog.set_level(logging.DEBUG, logger="training.trainer.trainer")

        trainer.start_session()
        _drain(trainer)

        assert any(r.levelno == logging.DEBUG and "kscript:" in r.message for r in caplog.records)

    @requires_tokenizer_data
    def test_compiled_entry_count_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-4: Compiled entry count logged after compilation."""
        bus = MessageBus()
        curriculum = _simple_curriculum()
        trainer = _make_trainer(bus, curriculum, curriculum_file="test.md")
        caplog.set_level(logging.INFO, logger="training.trainer.trainer")

        trainer.start_session()
        _drain(trainer)

        assert any(
            "Compiled" in r.message and "entries for lesson" in r.message for r in caplog.records
        )

    @requires_tokenizer_data
    def test_s1_fast_path_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-5: S1 events log with decompiled query and 'fast path'."""
        bus = MessageBus()
        _cap = BusCapture(bus)
        curriculum = _simple_curriculum()
        trainer = _make_trainer(bus, curriculum, curriculum_file="test.md")
        caplog.set_level(logging.INFO, logger="training.trainer.trainer")

        # Activate session so events are processed
        trainer._session_active = True

        # Simulate S1 ground event
        query = KLine(signature=256, nodes=[])  # identity M
        event = _make_event("ground", query, significance=_S1_SIGNIFICANCE)
        trainer.on_message(Message(role=TRAINEE_ROLE, action="ground", message=event))

        assert any("S1 (fast path)" in r.message for r in caplog.records)

    @requires_tokenizer_data
    def test_s2_significance_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-6: S2/S3 events log with normalised significance and proposal."""
        bus = MessageBus()
        _cap = BusCapture(bus)
        curriculum = _simple_curriculum()
        trainer = _make_trainer(bus, curriculum, curriculum_file="test.md")
        trainer._session_active = True
        caplog.set_level(logging.INFO, logger="training.trainer.trainer")

        query = KLine(signature=256, nodes=[8192])
        proposal = KLine(signature=256, nodes=[4096])
        event = _make_event("frame", query, proposal=proposal, significance=_S2_SIGNIFICANCE)
        trainer.on_message(Message(role=TRAINEE_ROLE, action="frame", message=event))

        assert any(
            "→ 0.50" in r.message and f"(d={_S2_DISTANCE})" in r.message for r in caplog.records
        )

    @requires_tokenizer_data
    def test_decompile_fallback_repr(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-7: Decompilation failure falls back to repr()."""
        bus = MessageBus()
        _cap = BusCapture(bus)
        curriculum = _simple_curriculum()
        trainer = _make_trainer(bus, curriculum, curriculum_file="test.md")
        trainer._session_active = True
        caplog.set_level(logging.INFO, logger="training.trainer.trainer")

        # Use a KLine that may not decompile cleanly but will still log
        query = KLine(signature=999999, nodes=[888888])
        event = _make_event("ground", query, significance=_S1_SIGNIFICANCE)
        trainer.on_message(Message(role=TRAINEE_ROLE, action="ground", message=event))

        # Should log without raising — the message just contains repr output
        assert any("GROUND" in r.message or "FRAME" in r.message for r in caplog.records)

    def test_lesson_complete_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-8: Lesson complete logs satisfaction counts."""
        bus = MessageBus()
        _cap = BusCapture(bus)
        curriculum = _simple_curriculum()
        trainer = _make_trainer(bus, curriculum, curriculum_file="test.md")
        caplog.set_level(logging.INFO, logger="training.trainer.trainer")

        # Directly invoke _check_lesson_complete with a satisfied lesson.
        # The completion check compares the satisfied vs submitted sets,
        # so seed both for the lesson entry before invoking it.
        entry = _make_entry(256, [])
        trainer._session_active = True
        trainer._reactor.load_lesson([entry])
        key = _entry_key(entry)
        trainer._state.mark_submitted(key)
        trainer._state.mark_satisfied(key)
        trainer._check_lesson_complete()

        assert any("complete" in r.message and "satisfied" in r.message for r in caplog.records)

    def test_curriculum_complete_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-9: Curriculum complete logged at INFO level."""
        bus = MessageBus()
        _cap = BusCapture(bus)
        curriculum = _simple_curriculum()
        trainer = _make_trainer(bus, curriculum, curriculum_file="test.md")
        caplog.set_level(logging.INFO, logger="training.trainer.trainer")

        # Move curriculum position past all lessons
        curriculum.position = curriculum.total()
        trainer._session_active = True
        trainer._submit_next_lesson()
        # _submit_next_lesson only sends the drain; the "Curriculum
        # complete" log fires inside _do_submit_lesson once drained.
        _drain(trainer)

        assert any("Curriculum complete" in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════
# Reactor logging tests
# ═══════════════════════════════════════════════════════════════════════


class TestReactorLogging:
    """TL-10 through TL-15: Reactor log output."""

    def _make_reactor(self, bus: MessageBus, *, max_rounds: int = 5, cogitate_fn=None) -> Reactor:
        doc = CurriculumDocument(
            objective="test",
            approach="test",
            lessons=[Lesson(label="1", prose="test", kscript=["M"])],
        )
        curriculum = Curriculum(doc)
        from training.trainer.curriculum import CurriculumState

        cs = CurriculumState(curriculum)
        return Reactor(
            bus, cs, role=TRAINER_ROLE, max_reactive_rounds=max_rounds, cogitate_fn=cogitate_fn
        )

    def test_auto_countersign_match_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-10: Auto-countersign match logged at INFO."""
        bus = MessageBus()
        _cap = BusCapture(bus)
        reactor = self._make_reactor(bus, cogitate_fn=None)
        caplog.set_level(logging.INFO, logger="training.trainer.reactor")

        entry = _make_entry(256, [])
        reactor.load_lesson([entry])

        proposal = KLine(signature=256, nodes=[])
        event = _make_event("frame", proposal, proposal=proposal, significance=_S2_SIGNIFICANCE)
        reactor.process_s2_s3(event)

        assert any(
            r.levelno == logging.INFO and "Auto-countersign" in r.message and "matched" in r.message
            for r in caplog.records
        )

    def test_auto_countersign_miss_debug_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-11: Auto-countersign miss logged at DEBUG."""
        bus = MessageBus()
        _cap = BusCapture(bus)
        reactor = self._make_reactor(bus)
        caplog.set_level(logging.DEBUG, logger="training.trainer.reactor")

        entry = _make_entry(256, [])
        reactor.load_lesson([entry])

        # Proposal that doesn't match any entry
        proposal = KLine(signature=999, nodes=[])
        event = _make_event("frame", proposal, proposal=proposal, significance=_S2_SIGNIFICANCE)

        # No cogitate_fn → will escalate. We just check the debug log before escalation.
        reactor.process_s2_s3(event)

        # Just check the log appeared — the miss is logged before escalation

        assert any(r.levelno == logging.DEBUG and "no match" in r.message for r in caplog.records)

    def test_reactive_scaffolding_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-12: Reactive scaffolding logged with round, confidence, source."""
        bus = MessageBus()
        _cap = BusCapture(bus)

        def mock_cogitate(event):
            return ("M = H", 0.85)

        reactor = self._make_reactor(bus, cogitate_fn=mock_cogitate)
        caplog.set_level(logging.INFO, logger="training.trainer.reactor")

        entry = _make_entry(256, [])
        reactor.load_lesson([entry])

        proposal = KLine(signature=999, nodes=[])
        event = _make_event("frame", proposal, proposal=proposal, significance=_S2_SIGNIFICANCE)
        reactor.process_s2_s3(event)

        assert any(
            "Reactive scaffolding" in r.message and "0.85" in r.message for r in caplog.records
        )

    def test_cogitation_failure_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-13: Cogitation failure logged at WARNING."""
        bus = MessageBus()
        _cap = BusCapture(bus)

        def mock_cogitate(event):
            return None

        reactor = self._make_reactor(bus, cogitate_fn=mock_cogitate)
        caplog.set_level(logging.WARNING, logger="training.trainer.reactor")

        entry = _make_entry(256, [])
        reactor.load_lesson([entry])

        proposal = KLine(signature=999, nodes=[])
        event = _make_event("frame", proposal, proposal=proposal, significance=_S2_SIGNIFICANCE)
        reactor.process_s2_s3(event)

        assert any(
            r.levelno == logging.WARNING and "no scaffolding" in r.message for r in caplog.records
        )

    def test_budget_exhaustion_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-14: Budget exhaustion logged at WARNING with round count."""
        bus = MessageBus()
        _cap = BusCapture(bus)

        reactor = self._make_reactor(bus, max_rounds=1)
        caplog.set_level(logging.WARNING, logger="training.trainer.reactor")

        entry = _make_entry(256, [])
        reactor.load_lesson([entry])

        proposal = KLine(signature=999, nodes=[])
        event = _make_event("frame", proposal, proposal=proposal, significance=_S2_SIGNIFICANCE)
        reactor.process_s2_s3(event)

        assert any(
            r.levelno == logging.WARNING and "budget exhausted" in r.message for r in caplog.records
        )

    def test_escalation_error_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-15: Escalation logged at ERROR with reason."""
        bus = MessageBus()
        _cap = BusCapture(bus)

        reactor = self._make_reactor(bus, max_rounds=1)
        caplog.set_level(logging.ERROR, logger="training.trainer.reactor")

        entry = _make_entry(256, [])
        reactor.load_lesson([entry])

        proposal = KLine(signature=999, nodes=[])
        event = _make_event("frame", proposal, proposal=proposal, significance=_S2_SIGNIFICANCE)
        reactor.process_s2_s3(event)

        assert any(r.levelno == logging.ERROR and "Escalation" in r.message for r in caplog.records)


# ═══════════════════════════════════════════════════════════════════════
# Adapter logging tests
# ═══════════════════════════════════════════════════════════════════════


class TestAdapterLogging:
    """TL-16 through TL-18: Adapter log output."""

    def _make_adapter(self, bus: MessageBus) -> KAgentAdapter:
        adapter = KAgentAdapter(bus, role=TRAINEE_ROLE)
        # Bind a fake KAgent
        fake = MagicMock()
        fake.rationalise = MagicMock(return_value=True)
        fake.countersign = MagicMock(return_value=True)
        adapter.bind(fake)
        return adapter

    @requires_tokenizer_data
    def test_entry_submit_count_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-16: Entry submission logged with count at INFO."""
        bus = MessageBus()
        _cap = BusCapture(bus)
        adapter = self._make_adapter(bus)
        caplog.set_level(logging.INFO, logger="training.harness.adapter")

        adapter.on_message(
            Message(
                role=TRAINER_ROLE,
                action="submit",
                message="M",
                sender=TRAINER_ROLE,
            )
        )

        assert any(
            "Submitting" in r.message and "compiled entries" in r.message for r in caplog.records
        )

    def test_compilation_error_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-17: Compilation error logged at ERROR."""
        bus = MessageBus()
        _cap = BusCapture(bus)
        adapter = self._make_adapter(bus)
        caplog.set_level(logging.ERROR, logger="training.harness.adapter")

        # Send invalid KScript
        adapter.on_message(
            Message(
                role=TRAINER_ROLE,
                action="submit",
                message="!!!invalid!!!",
                sender=TRAINER_ROLE,
            )
        )

        assert any(
            r.levelno == logging.ERROR and "Compilation error" in r.message for r in caplog.records
        )

    def test_countersign_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """TL-18: Countersign logged with KLine at INFO."""
        bus = MessageBus()
        _cap = BusCapture(bus)
        adapter = self._make_adapter(bus)
        caplog.set_level(logging.INFO, logger="training.harness.adapter")

        kline = KLine(signature=256, nodes=[8192])
        adapter.on_message(
            Message(
                role=TRAINER_ROLE,
                action="countersign",
                message=kline,
                sender=TRAINER_ROLE,
            )
        )

        assert any("Countersign" in r.message for r in caplog.records)
