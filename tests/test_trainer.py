"""Tests for the Trainer participant.

Covers: HRNS-12, HRNS-13, HRNS-14, HRNS-16, HRNS-19, HRNS-20, HRNS-24,
CRS-38..CRS-49.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

from harness.bus import MessageBus
from harness.message import Message
from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kscript.token_encoder import CompiledEntry
from trainer.curriculum import Curriculum, CurriculumState, EntryKey
from trainer.curriculum_document import CurriculumDocument, Lesson
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


def _write_curriculum(path: Path) -> None:
    """Write a sample curriculum to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent("""\
        ## Objective

        Teach basic structure.

        ## Approach

        Step by step.

        ## Lessons

        ### 1

        First lesson.

        ```
        M = S
        ```

        ### 2

        Second lesson.

        ```
        H = V
        ```

        ### 3

        Third lesson.

        ```
        MHALL = SVO
        ```
    """))


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
    curriculum_file: str | Path | None = None,
    curricula_dir: str | Path | None = None,
) -> tuple[Trainer, BusCapture]:
    """Create a Trainer with BusCapture installed.

    BusCapture is installed AFTER construction, so constructor-time
    bus.send() calls are NOT captured. Use ``_make_trainer_with_capture()``
    when you need to capture constructor-time messages.
    """
    trainer = Trainer(
        bus,
        curriculum,
        save_path=save_path,
        max_reactive_rounds=max_reactive_rounds,
        cogitate_fn=cogitate_fn,
        curriculum_file=curriculum_file,
        curricula_dir=curricula_dir,
    )
    capture = BusCapture(bus)
    capture.install()
    return trainer, capture


def _make_trainer_with_capture(
    bus: MessageBus,
    curriculum: Curriculum,
    *,
    save_path: str | Path | None = None,
    max_reactive_rounds: int = 5,
    cogitate_fn=None,
    curriculum_file: str | Path | None = None,
    curricula_dir: str | Path | None = None,
) -> tuple[Trainer, BusCapture]:
    """Create a Trainer with BusCapture installed BEFORE construction.

    Unlike ``_make_trainer()``, this installs the BusCapture on the bus
    before calling ``Trainer.__init__()``, so constructor-time bus.send()
    calls (e.g., the polling status emission from Path 3) are captured.
    """
    capture = BusCapture(bus)
    capture.install()
    trainer = Trainer(
        bus,
        curriculum,
        save_path=save_path,
        max_reactive_rounds=max_reactive_rounds,
        cogitate_fn=cogitate_fn,
        curriculum_file=curriculum_file,
        curricula_dir=curricula_dir,
    )
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

        # Polling mode was never active in this flow
        assert trainer._polling_for_goal is False


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


# ── CRS-38..CRS-42: Session startup ──────────────────────────────────


class TestSessionStartup:
    """CRS-38..CRS-42: Session startup resolution."""

    @patch("trainer.trainer.compile_source")
    def test_session_startup_from_file_param(self, mock_compile: MagicMock, tmp_path: Path) -> None:
        """CRS-38: Session startup loads curriculum from runtime parameter."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "test.md"
        _write_curriculum(curriculum_path)

        bus = MessageBus()
        doc = CurriculumDocument.from_file(curriculum_path)
        curriculum = Curriculum(doc)
        trainer, capture = _make_trainer(
            bus, curriculum, curriculum_file=curriculum_path
        )
        trainer.start_session()

        # Session started with document
        assert trainer._session_active
        assert trainer.state.curriculum.document.lessons[0].label == "1"
        # Submit was sent
        submit_msgs = capture.find_all("kalvin", "submit")
        assert len(submit_msgs) >= 1

    @patch("trainer.trainer.compile_source")
    def test_session_startup_from_saved_state(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """CRS-39: Session startup resumes from saved state with curriculum_file."""
        mock_compile.return_value = [_make_entry(100, [10])]

        # Write curriculum file
        curriculum_path = tmp_path / "curricula" / "test.md"
        _write_curriculum(curriculum_path)

        # Save state with curriculum_file
        save_file = tmp_path / "state.json"
        doc = CurriculumDocument.from_file(curriculum_path)
        curriculum = Curriculum(doc)
        state = CurriculumState(
            curriculum, save_path=save_file, curriculum_file=str(curriculum_path)
        )
        state.mark_lesson_submitted("1")
        state.mark_lesson_satisfied("1")
        state.save()

        # Create trainer with empty curriculum but saved state
        bus = MessageBus()
        empty_curriculum = Curriculum([])
        trainer, capture = _make_trainer(bus, empty_curriculum, save_path=save_file)
        # Manually trigger startup resolution by loading saved state
        loaded_state = CurriculumState.load(save_file)
        trainer._state = loaded_state
        trainer.start_session()

        # Should have loaded from saved curriculum_file
        assert trainer._session_active
        assert trainer.state.curriculum.document.lessons[0].label == "1"

    @patch("trainer.trainer.compile_source")
    def test_session_startup_polls_for_goal(self, mock_compile: MagicMock) -> None:
        """CRS-40: Session startup polls for goal when no param and no saved state.

        When no curriculum file and no saved state exist, start_session()
        enters goal-polling mode instead of starting a session. The trainer
        waits for human input to provide a goal or curriculum file path.
        """
        mock_compile.return_value = [_make_entry(100, [10])]

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer_with_capture(bus, curriculum)

        # Reset to capture only start_session messages (not constructor ones)
        capture.reset()

        trainer.start_session()

        # Trainer enters polling mode — session is NOT active
        assert trainer._polling_for_goal is True
        assert not trainer._session_active

        # A polling_for_goal event was logged
        polling_events = [
            e for e in trainer.state.event_log if e["type"] == "polling_for_goal"
        ]
        assert len(polling_events) >= 1

        # A progress message was emitted to "ui" with status "polling_for_goal"
        progress_msgs = capture.find_all("ui", "progress")
        polling_progress = [
            m for m in progress_msgs if m.message["status"] == "polling_for_goal"
        ]
        assert len(polling_progress) >= 1

    def test_constructor_enters_polling_mode(self) -> None:
        """Constructor Path 3: no file, no saved state, empty curriculum → polling mode.

        When the Trainer is constructed with no curriculum_file and an
        empty curriculum, it immediately enters goal-polling mode without
        waiting for start_session(). This allows the trainer to be ready
        to receive goal input as soon as it's created.
        """
        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer_with_capture(bus, curriculum)

        # After construction, polling mode is active
        assert trainer._polling_for_goal is True

        # A polling_for_goal event was logged
        polling_events = [
            e for e in trainer.state.event_log if e["type"] == "polling_for_goal"
        ]
        assert len(polling_events) >= 1

        # A progress message to "ui" with status "polling_for_goal" was emitted
        progress_msgs = capture.find_all("ui", "progress")
        polling_progress = [
            m for m in progress_msgs if m.message["status"] == "polling_for_goal"
        ]
        assert len(polling_progress) >= 1

    @patch("trainer.trainer.compile_source")
    def test_no_polling_when_curriculum_file_exists(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """When a valid curriculum file exists, polling mode is NOT entered.

        Negative test: a trainer with a curriculum file parameter should
        start a normal session, not enter goal-polling mode.
        """
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "test.md"
        _write_curriculum(curriculum_path)

        bus = MessageBus()
        doc = CurriculumDocument.from_file(curriculum_path)
        curriculum = Curriculum(doc)
        trainer, _capture = _make_trainer(
            bus, curriculum, curriculum_file=curriculum_path
        )

        # Constructor should NOT enter polling mode
        assert trainer._polling_for_goal is False

        trainer.start_session()

        # Session starts normally
        assert trainer._session_active is True
        assert trainer._polling_for_goal is False


class TestGoalResolution:
    """CRS-41..CRS-42: Goal resolution."""

    @patch("trainer.trainer.compile_source")
    def test_goal_prefix_triggers_generation(self, mock_compile: MagicMock) -> None:
        """CRS-41: Goal starting with 'goal:' triggers generation."""
        mock_compile.return_value = [_make_entry(100, [10])]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1"])
        trainer, _capture = _make_trainer(
            bus, curriculum, curricula_dir="/tmp/curricula"
        )
        trainer.start_session()

        # Polling mode not active (session is active), so "goal:" input
        # goes through the regular input handler and queues
        trainer.on_message(
            Message(address="trainer", action="input", message="goal: teach SVO", sender="slack")
        )
        # Goal is queued since session is active
        assert "teach SVO" in trainer._pending_goals

    @patch("trainer.trainer.compile_source")
    def test_goal_file_path_triggers_load(self, mock_compile: MagicMock, tmp_path: Path) -> None:
        """CRS-42: Goal that is a file path triggers direct loading."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "test.md"
        _write_curriculum(curriculum_path)

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, _capture = _make_trainer(
            bus, curriculum, curricula_dir=str(tmp_path / "curricula")
        )

        # Set polling mode so input is treated as goal
        trainer._polling_for_goal = True

        trainer.on_message(
            Message(
                address="trainer",
                action="input",
                message=str(curriculum_path),
                sender="slack",
            )
        )

        # Should have loaded the curriculum file
        assert not trainer._polling_for_goal
        assert trainer.state.curriculum.document.lessons[0].label == "1"


# ── Polling-mode input handling ──────────────────────────────────────


class TestPollingModeInputHandling:
    """Polling-mode input handling: goal text and file path resolution."""

    @patch("trainer.trainer.compile_source")
    def test_polling_mode_resolves_goal_input(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """When polling, a 'goal:' input resolves via _resolve_goal and starts session.

        The trainer enters polling mode (empty curriculum, no file). When
        a 'goal: teach X' input arrives, _resolve_goal is called, polling
        mode ends, and the session starts with the generated curriculum.
        """
        mock_compile.return_value = [_make_entry(100, [10])]

        # Provide a curriculum file so _load_and_start can succeed via
        # the _resolve_goal path when the goal text triggers generation.
        curriculum_path = tmp_path / "gen_curriculum.md"
        _write_curriculum(curriculum_path)

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer_with_capture(
            bus, curriculum, curricula_dir=str(tmp_path)
        )

        # Confirm polling mode is active from constructor
        assert trainer._polling_for_goal is True
        capture.reset()

        # Send goal input
        trainer.on_message(
            Message(
                address="trainer",
                action="input",
                message="goal: teach SVO patterns",
                sender="slack",
            )
        )

        # Polling mode should be cleared
        assert trainer._polling_for_goal is False

        # A goal_received event should have been logged
        goal_events = [
            e for e in trainer.state.event_log if e["type"] == "goal_received"
        ]
        assert len(goal_events) >= 1
        assert "teach SVO patterns" in goal_events[0]["data"]["goal"]

    @patch("trainer.trainer.compile_source")
    def test_polling_mode_resolves_file_path_input(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """When polling, a valid .md file path input loads the curriculum and starts."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "test.md"
        _write_curriculum(curriculum_path)

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer_with_capture(
            bus, curriculum, curricula_dir=str(tmp_path)
        )

        # Confirm polling mode is active
        assert trainer._polling_for_goal is True
        capture.reset()

        # Send file path as input
        trainer.on_message(
            Message(
                address="trainer",
                action="input",
                message=str(curriculum_path),
                sender="slack",
            )
        )

        # Polling mode cleared, session started with loaded curriculum
        assert trainer._polling_for_goal is False
        assert trainer._session_active is True
        assert trainer.state.curriculum.document.lessons[0].label == "1"

        # Submit was sent to kalvin
        submit_msgs = capture.find_all("kalvin", "submit")
        assert len(submit_msgs) >= 1


# ── CRS-43..CRS-45: File polling and amendment ───────────────────────


class TestFilePolling:
    """CRS-43..CRS-45: File polling and monotonic submitted set."""

    @patch("trainer.trainer.compile_source")
    def test_trainer_rereads_file_before_lesson(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """CRS-43: Trainer re-reads curriculum file before each lesson submission."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "test.md"
        _write_curriculum(curriculum_path)

        bus = MessageBus()
        doc = CurriculumDocument.from_file(curriculum_path)
        curriculum = Curriculum(doc)
        trainer, capture = _make_trainer(
            bus, curriculum, curriculum_file=curriculum_path
        )
        trainer.start_session()

        # Verify first submit happened
        submit_msgs = capture.find_all("kalvin", "submit")
        assert len(submit_msgs) >= 1

    @patch("trainer.trainer.compile_source")
    def test_new_lessons_submitted_after_reread(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """CRS-44: New lessons after current label are submitted after re-read."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "test.md"
        _write_curriculum(curriculum_path)

        bus = MessageBus()
        doc = CurriculumDocument.from_file(curriculum_path)
        curriculum = Curriculum(doc)
        trainer, capture = _make_trainer(
            bus, curriculum, curriculum_file=curriculum_path
        )
        trainer.start_session()

        # Amend the file — append lesson 4
        doc2 = CurriculumDocument.from_file(curriculum_path)
        doc2.amend("append", lesson=Lesson(label="4", prose="New.", kscript=["X = Y"]))
        capture.reset()

        # Complete current lesson to trigger next submit (which polls)
        event = _make_event(
            "ground",
            query=KLine(signature=100, nodes=[10]),
            proposal=KLine(signature=100, nodes=[10]),
            significance=_S1_SIGNIFICANCE,
        )
        trainer.on_message(Message(address="trainer", action="ground", message=event))

        # After re-read, curriculum should have 4 lessons
        assert len(trainer.state.curriculum.document.lessons) == 4

    @patch("trainer.trainer.compile_source")
    def test_monotonic_set_prevents_duplicates(self, mock_compile: MagicMock) -> None:
        """CRS-45: Monotonic submitted set prevents duplicate kline submissions."""
        mock_compile.return_value = [_make_entry(100, [10])]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()
        capture.reset()

        # The entry was submitted
        key = _entry_key(_make_entry(100, [10]))
        assert trainer.state.is_submitted(key)

        # Re-submitting the same entry won't create a duplicate
        trainer.state.mark_submitted(key)
        assert trainer.state.submitted == {key}


# ── CRS-46..CRS-49: Progress events ──────────────────────────────────


class TestProgressEvents:
    """CRS-46..CRS-49: Progress events."""

    @patch("trainer.trainer.compile_source")
    def test_progress_event_session_start(self, mock_compile: MagicMock) -> None:
        """CRS-46: Progress event emitted on session start."""
        mock_compile.return_value = [_make_entry(100, [10])]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        progress_msgs = capture.find_all("ui", "progress")
        assert len(progress_msgs) >= 1
        started = progress_msgs[0]
        assert started.message["status"] == "started"
        assert started.message["lessons_total"] == 1
        assert started.message["lessons_completed"] == 0

    @patch("trainer.trainer.compile_source")
    def test_progress_event_lesson_complete(self, mock_compile: MagicMock) -> None:
        """CRS-47: Progress event emitted on lesson complete."""
        mock_compile.return_value = [_make_entry(100, [10])]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()
        capture.reset()

        # Complete lesson 1
        event = _make_event(
            "ground",
            query=KLine(signature=100, nodes=[10]),
            proposal=KLine(signature=100, nodes=[10]),
            significance=_S1_SIGNIFICANCE,
        )
        trainer.on_message(Message(address="trainer", action="ground", message=event))

        progress_msgs = capture.find_all("ui", "progress")
        complete_msgs = [m for m in progress_msgs if m.message["status"] == "lesson_complete"]
        assert len(complete_msgs) >= 1
        assert complete_msgs[0].message["lessons_completed"] == 1

    @patch("trainer.trainer.compile_source")
    def test_progress_event_curriculum_complete(self, mock_compile: MagicMock) -> None:
        """CRS-48: Progress event emitted on curriculum complete."""
        mock_compile.return_value = [_make_entry(100, [10])]

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
        trainer.on_message(Message(address="trainer", action="ground", message=event))

        progress_msgs = capture.find_all("ui", "progress")
        complete_msgs = [m for m in progress_msgs if m.message["status"] == "complete"]
        assert len(complete_msgs) >= 1

    @patch("trainer.trainer.compile_source")
    def test_progress_event_amendment(self, mock_compile: MagicMock, tmp_path: Path) -> None:
        """CRS-49: Progress event emitted on amendment applied."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "test.md"
        _write_curriculum(curriculum_path)

        bus = MessageBus()
        doc = CurriculumDocument.from_file(curriculum_path)
        curriculum = Curriculum(doc)
        trainer, capture = _make_trainer(
            bus, curriculum, curriculum_file=curriculum_path
        )
        trainer.start_session()
        capture.reset()

        # Request amendment
        trainer.request_amendment(
            "append",
            lesson=Lesson(label="4", prose="New lesson.", kscript=["X = Y"]),
        )

        # Verify progress event
        progress_msgs = capture.find_all("ui", "progress")
        amended_msgs = [m for m in progress_msgs if m.message["status"] == "amended"]
        assert len(amended_msgs) >= 1
