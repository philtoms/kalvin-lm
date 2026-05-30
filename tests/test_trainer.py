"""Tests for the Trainer participant.

Covers: HRNS-12, HRNS-13, HRNS-14, HRNS-16, HRNS-19, HRNS-20, HRNS-24.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from harness.bus import MessageBus
from harness.message import Message
from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kscript.token_encoder import CompiledEntry
from trainer.curriculum import Curriculum, CurriculumState, EntryKey
from trainer.trainer import Trainer

# ── Significance constants for test events ────────────────────────────

# S1 threshold: D_MAX - 1 = 0xFFFF_FFFF_FFFF_FFFE
_S1_SIGNIFICANCE = 0xFFFF_FFFF_FFFF_FFFE
# S2/S3: any low significance value
_S2_SIGNIFICANCE = 100


# ── Test helpers ──────────────────────────────────────────────────────


def _make_entry(sig: int, nodes: list[int]) -> CompiledEntry:
    """Create a CompiledEntry with the given signature and nodes."""
    return CompiledEntry(signature=sig, nodes=nodes, dbg_text=f"test-{sig:#x}")


def _make_event(
    kind: str,
    query: KLine,
    proposal: KLine,
    significance: int,
) -> RationaliseEvent:
    """Create a RationaliseEvent."""
    return RationaliseEvent(
        kind=kind,
        query=query,
        proposal=proposal,
        significance=significance,
    )


def _entry_key(kline: KLine) -> EntryKey:
    """Create an EntryKey from a KLine."""
    return (kline.signature, tuple(kline.nodes))


class BusCapture:
    """Captures messages sent via bus.send() for test assertions."""

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self.messages: list[Message] = []
        self._original_send = bus.send

    def install(self) -> None:
        """Replace bus.send with our capturing wrapper."""
        capture = self

        def capturing_send(msg: Message) -> None:
            capture.messages.append(msg)
            # Don't forward to real bus (avoids threading issues)

        self._bus.send = capturing_send  # type: ignore[assignment]

    def find_all(self, address: str, action: str) -> list[Message]:
        """Return all captured messages matching address and action."""
        return [
            m
            for m in self.messages
            if m.address == address and m.action == action
        ]

    def find_one(self, address: str, action: str) -> Message | None:
        """Return the first captured message matching address and action, or None."""
        matches = self.find_all(address, action)
        return matches[0] if matches else None

    def reset(self) -> None:
        """Clear captured messages."""
        self.messages.clear()


def _make_trainer(
    bus: MessageBus,
    curriculum: Curriculum,
    *,
    save_path: str | Path | None = None,
    max_reactive_rounds: int = 5,
    cogitate_fn=None,
) -> tuple[Trainer, BusCapture]:
    """Create a Trainer with BusCapture installed."""
    trainer = Trainer(
        bus,
        curriculum,
        save_path=save_path,
        max_reactive_rounds=max_reactive_rounds,
        cogitate_fn=cogitate_fn,
    )
    capture = BusCapture(bus)
    capture.install()
    return trainer, capture


# ── HRNS-12: Auto-countersign on structural match ────────────────────


class TestAutoCountersignStructuralMatch:
    """HRNS-12: Trainer auto-countersigns structurally matching proposals."""

    @patch("trainer.trainer.compile_source")
    def test_auto_countersign_structural_match(self, mock_compile: MagicMock) -> None:
        bus = MessageBus()
        entry = _make_entry(100, [10, 20])
        mock_compile.return_value = [entry]

        curriculum = Curriculum(["MHALL = SVO"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        # Clear startup messages
        capture.reset()

        # Simulate KAgent frame event with matching proposal (non-S1)
        proposal = KLine(signature=100, nodes=[10, 20])
        query = KLine(signature=999, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(
            Message(address="trainer", action="frame", message=event)
        )

        # Verify countersign was sent to kalvin
        cs_msgs = capture.find_all("kalvin", "countersign")
        assert len(cs_msgs) == 1
        assert cs_msgs[0].sender == "trainer"
        # The countersign message contains the proposal KLine
        assert cs_msgs[0].message == proposal

        # Verify entry is marked satisfied
        key = _entry_key(entry)
        assert trainer.state.is_satisfied(key)


# ── HRNS-13: Reactive mode on S2/S3 ──────────────────────────────────


class TestReactiveModeOnS2S3:
    """HRNS-13: Trainer enters reactive mode on S2/S3 events."""

    @patch("trainer.trainer.compile_source")
    def test_reactive_mode_on_s2_s3(self, mock_compile: MagicMock) -> None:
        bus = MessageBus()
        entry = _make_entry(100, [10, 20])
        mock_compile.return_value = [entry]

        curriculum = Curriculum(["MHALL = SVO", "S = M / V = H"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        # Clear startup messages
        capture.reset()

        # Simulate KAgent frame event with non-matching proposal
        proposal = KLine(signature=999, nodes=[88])  # doesn't match entry
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(
            Message(address="trainer", action="frame", message=event)
        )

        # Verify NO countersign was sent
        cs_msgs = capture.find_all("kalvin", "countersign")
        assert len(cs_msgs) == 0

        # Verify escalation to slack (low_confidence since no cogitate_fn)
        notify_msgs = capture.find_all("slack", "notify")
        assert len(notify_msgs) >= 1
        escalation = notify_msgs[0].message
        assert escalation["reason"] == "low_confidence"


# ── HRNS-14: Escalation on budget exhaustion ─────────────────────────


class TestEscalationOnBudgetExhaustion:
    """HRNS-14: Trainer escalates to Slack on budget exhaustion."""

    @patch("trainer.trainer.compile_source")
    def test_escalation_on_budget_exhaustion(self, mock_compile: MagicMock) -> None:
        # Use max_reactive_rounds=3 and a lesson with 3 entries
        # so reactive_rounds can reach 3 within a single lesson
        entries = [
            _make_entry(100 + i, [10 + i])
            for i in range(3)
        ]
        mock_compile.return_value = entries

        bus = MessageBus()
        curriculum = Curriculum(["lesson1"])
        trainer, capture = _make_trainer(
            bus, curriculum, max_reactive_rounds=3
        )
        trainer.start_session()
        capture.reset()

        # Send 3 non-matching S2/S3 frame events
        for i in range(3):
            proposal = KLine(signature=900 + i, nodes=[99 + i])
            query = KLine(signature=800 + i, nodes=[i])
            event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)
            trainer.on_message(
                Message(address="trainer", action="frame", message=event)
            )

        # Verify budget_exhaustion escalation
        notify_msgs = capture.find_all("slack", "notify")
        budget_esc = [
            m for m in notify_msgs if m.message["reason"] == "budget_exhaustion"
        ]
        assert len(budget_esc) >= 1


# ── HRNS-16: One session at a time ───────────────────────────────────


class TestOneSessionAtATime:
    """HRNS-16: Trainer accepts one session at a time; queues additional goals."""

    @patch("trainer.trainer.compile_source")
    def test_one_session_at_a_time(self, mock_compile: MagicMock) -> None:
        mock_compile.return_value = [_make_entry(100, [10])]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum)

        # Start first session
        trainer.start_session(goal="first")
        assert trainer._session_active

        # Clear startup messages
        capture.reset()

        # Send input with "goal: second" from slack
        trainer.on_message(
            Message(address="trainer", action="input", message="goal: second", sender="slack")
        )

        # Verify second goal is queued
        assert trainer._pending_goals == ["second"]

        # First session still active
        assert trainer._session_active

        # Verify a log event was recorded for the queued goal
        queued_events = [
            e for e in trainer.state.event_log if e["type"] == "goal_queued"
        ]
        assert len(queued_events) == 1
        assert queued_events[0]["data"]["goal"] == "second"


# ── HRNS-19: Session pause ───────────────────────────────────────────


class TestSessionPause:
    """HRNS-19: Session pause stops submitting but stays active."""

    @patch("trainer.trainer.compile_source")
    def test_session_pause(self, mock_compile: MagicMock) -> None:
        mock_compile.return_value = [_make_entry(100, [10])]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        # Verify first lesson submitted
        submit_msgs = capture.find_all("kalvin", "submit")
        assert len(submit_msgs) == 1

        # Send pause
        trainer.on_message(
            Message(address="trainer", action="input", message="pause", sender="slack")
        )
        assert trainer._session_paused

        capture.reset()

        # Complete the current lesson with S1 event
        query = KLine(signature=100, nodes=[10])
        proposal = KLine(signature=100, nodes=[10])
        event = _make_event("ground", query, proposal, _S1_SIGNIFICANCE)
        trainer.on_message(
            Message(address="trainer", action="ground", message=event)
        )

        # Curriculum position should advance (lesson complete)
        assert trainer.state.curriculum.position == 1

        # But NO auto-submit of next lesson (paused)
        submit_after = capture.find_all("kalvin", "submit")
        assert len(submit_after) == 0

        # Resume
        trainer.on_message(
            Message(address="trainer", action="input", message="resume", sender="slack")
        )
        assert not trainer._session_paused

        # After resume, next lesson should be submitted
        submit_resume = capture.find_all("kalvin", "submit")
        assert len(submit_resume) == 1


# ── HRNS-20: Session stop ────────────────────────────────────────────


class TestSessionStop:
    """HRNS-20: Session stop ends session, persists state, goes dormant."""

    @patch("trainer.trainer.compile_source")
    def test_session_stop(self, mock_compile: MagicMock, tmp_path: Path) -> None:
        mock_compile.return_value = [_make_entry(100, [10])]

        save_file = tmp_path / "state.json"
        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum, save_path=save_file)
        trainer.start_session()

        capture.reset()

        # Send stop
        trainer.on_message(
            Message(address="trainer", action="input", message="stop", sender="slack")
        )

        # Verify session ended
        assert not trainer._session_active

        # Verify state persisted
        assert save_file.exists()

        capture.reset()

        # Subsequent KAgent events should be ignored
        query = KLine(signature=100, nodes=[10])
        proposal = KLine(signature=100, nodes=[10])
        event = _make_event("ground", query, proposal, _S1_SIGNIFICANCE)
        trainer.on_message(
            Message(address="trainer", action="ground", message=event)
        )

        # No messages sent in response
        assert len(capture.messages) == 0


# ── HRNS-24: Entry counting / lesson complete ─────────────────────────


class TestEntryCountingLessonComplete:
    """HRNS-24: Trainer counts submitted entries; knows when lesson is complete."""

    @patch("trainer.trainer.compile_source")
    def test_entry_counting_lesson_complete(self, mock_compile: MagicMock) -> None:
        # Lesson compiles to 3 entries
        entries = [
            _make_entry(100, [10]),
            _make_entry(200, [20]),
            _make_entry(300, [30]),
        ]
        mock_compile.return_value = entries

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        # Clear startup messages
        capture.reset()

        # Simulate 3 KAgent response events (S1 ground events)
        for entry in entries:
            event = _make_event(
                "ground",
                query=KLine(signature=entry.signature, nodes=entry.nodes),
                proposal=KLine(signature=entry.signature, nodes=entry.nodes),
                significance=_S1_SIGNIFICANCE,
            )
            trainer.on_message(
                Message(address="trainer", action="ground", message=event)
            )

        # After 3 events: curriculum position advances
        assert trainer.state.curriculum.position == 1

        # Next lesson submitted (not paused, curriculum has more lessons)
        submit_msgs = capture.find_all("kalvin", "submit")
        # At least the submit for lesson2
        assert any(m.message == "lesson2" for m in submit_msgs)


# ── Additional tests ─────────────────────────────────────────────────


class TestCurriculumCompleteEndsSession:
    """Advance through all lessons — session ends when curriculum is complete."""

    @patch("trainer.trainer.compile_source")
    def test_curriculum_complete_ends_session(self, mock_compile: MagicMock) -> None:
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["only_lesson"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        capture.reset()

        # Complete the only lesson
        event = _make_event(
            "ground",
            query=KLine(signature=100, nodes=[10]),
            proposal=KLine(signature=100, nodes=[10]),
            significance=_S1_SIGNIFICANCE,
        )
        trainer.on_message(
            Message(address="trainer", action="ground", message=event)
        )

        # Curriculum complete → session ends
        assert not trainer._session_active
        assert trainer.state.curriculum.is_complete()


class TestFastPathAutoSatisfy:
    """S1 event auto-satisfies entry without countersign."""

    @patch("trainer.trainer.compile_source")
    def test_fast_path_auto_satisfy(self, mock_compile: MagicMock) -> None:
        entry = _make_entry(42, [1, 2, 3])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        capture.reset()

        # S1 ground event
        event = _make_event(
            "ground",
            query=KLine(signature=42, nodes=[1, 2, 3]),
            proposal=KLine(signature=42, nodes=[1, 2, 3]),
            significance=_S1_SIGNIFICANCE,
        )
        trainer.on_message(
            Message(address="trainer", action="ground", message=event)
        )

        # Entry is satisfied
        key = _entry_key(entry)
        assert trainer.state.is_satisfied(key)

        # No countersign sent (fast path auto-satisfies, no countersign needed)
        cs_msgs = capture.find_all("kalvin", "countersign")
        assert len(cs_msgs) == 0


class TestCompilationErrorFromKalvin:
    """Error event from KAgent is logged and counts toward entry counting."""

    @patch("trainer.trainer.compile_source")
    def test_compilation_error_from_kalvin(self, mock_compile: MagicMock) -> None:
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        capture.reset()

        # Simulate error event from KAgent
        trainer.on_message(
            Message(
                address="trainer",
                action="error",
                message="ParseError at line 1: unexpected token",
            )
        )

        # Error is logged
        error_events = [
            e for e in trainer.state.event_log if e["type"] == "kagent_error"
        ]
        assert len(error_events) == 1
        assert "ParseError" in error_events[0]["data"]["message"]

        # After 1 error for 1-entry lesson, lesson should complete
        assert trainer.state.curriculum.position == 1

        # Next lesson submitted
        submit_msgs = capture.find_all("kalvin", "submit")
        assert any(m.message == "lesson2" for m in submit_msgs)


class TestStopPersistsState:
    """Submit some entries, stop, reload state from disk, verify position and sets match."""

    @patch("trainer.trainer.compile_source")
    def test_stop_persists_state(self, mock_compile: MagicMock, tmp_path: Path) -> None:
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        save_file = tmp_path / "trainer_state.json"
        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2", "lesson3"])
        trainer, _capture = _make_trainer(bus, curriculum, save_path=save_file)
        trainer.start_session()

        # Mark entry as submitted (done via start_session → _submit_next_lesson)
        key = _entry_key(entry)
        assert trainer.state.is_submitted(key)

        # Satisfy the entry and complete lesson 1
        event = _make_event(
            "ground",
            query=KLine(signature=100, nodes=[10]),
            proposal=KLine(signature=100, nodes=[10]),
            significance=_S1_SIGNIFICANCE,
        )
        trainer.on_message(
            Message(address="trainer", action="ground", message=event)
        )

        # Now at lesson 2 (position=1)
        assert trainer.state.curriculum.position == 1

        # Stop session
        trainer.on_message(
            Message(address="trainer", action="input", message="stop", sender="slack")
        )

        # Verify file persisted
        assert save_file.exists()

        # Load state back
        loaded = CurriculumState.load(save_file)
        assert loaded.curriculum.position == 1
        assert loaded.curriculum.lessons == ["lesson1", "lesson2", "lesson3"]
        assert loaded.submitted == {key}
        assert loaded.satisfied == {key}


class TestCogitateFnInjection:
    """Provide a cogitate_fn that returns scaffolding — reactive mode uses it."""

    @patch("trainer.trainer.compile_source")
    def test_cogitate_fn_injection(self, mock_compile: MagicMock) -> None:
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        # Mock cogitate function that returns scaffolding
        mock_cogitate = MagicMock(
            return_value=("S = X / V = Y", 0.85)
        )

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(
            bus, curriculum, cogitate_fn=mock_cogitate
        )
        trainer.start_session()
        capture.reset()

        # Non-matching S2/S3 event
        proposal = KLine(signature=999, nodes=[88])
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(
            Message(address="trainer", action="frame", message=event)
        )

        # Cogitate was called
        mock_cogitate.assert_called_once_with(event)

        # Reactive scaffolding was submitted to kalvin
        submit_msgs = capture.find_all("kalvin", "submit")
        scaffolding_msgs = [m for m in submit_msgs if m.message == "S = X / V = Y"]
        assert len(scaffolding_msgs) == 1
        assert scaffolding_msgs[0].sender == "trainer"

        # No escalation (low_confidence) should have occurred
        notify_msgs = capture.find_all("slack", "notify")
        escalation_msgs = [
            m for m in notify_msgs
            if m.message.get("reason") in ("low_confidence", "budget_exhaustion")
        ]
        assert len(escalation_msgs) == 0


class TestGuidanceTextAppended:
    """Free-text input from Slack is appended to conversation history."""

    @patch("trainer.trainer.compile_source")
    def test_guidance_text(self, mock_compile: MagicMock) -> None:
        mock_compile.return_value = [_make_entry(100, [10])]

        bus = MessageBus()
        curriculum = Curriculum(["lesson"])
        trainer, _capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        # Send free-text guidance
        trainer.on_message(
            Message(
                address="trainer",
                action="input",
                message="try using simpler constructs",
                sender="slack",
            )
        )

        assert "try using simpler constructs" in trainer._conversation_history
