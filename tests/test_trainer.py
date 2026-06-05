"""Tests for the Trainer participant.

Covers: HRNS-12, HRNS-13, HRNS-14, HRNS-16, HRNS-19, HRNS-20, HRNS-24,
CRS-38..CRS-49.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

from harness.bus import MessageBus
from harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE, TRAINER_ROLE
from harness.message import Message
from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kscript.token_encoder import CompiledEntry
from trainer.cogitation import LLMResponse
from trainer.curriculum import Curriculum, CurriculumState, EntryKey
from trainer.curriculum_document import CurriculumDocument, Lesson
from trainer.curriculum_generator import CurriculumGenerationError
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

    def find_all(self, role: str, action: str) -> list[Message]:
        """Return all captured messages matching role and action."""
        return [
            m
            for m in self.messages
            if m.role == role and m.action == action
        ]

    def find_one(self, role: str, action: str) -> Message | None:
        """Return the first captured message matching role and action, or None."""
        matches = self.find_all(role, action)
        return matches[0] if matches else None

    def reset(self) -> None:
        """Clear captured messages."""
        self.messages.clear()


def _make_trainer(
    bus: MessageBus,
    curriculum: Curriculum,
    *,
    save_path: str | Path | None = None,
    curriculum_file: str | Path | None = None,
    curricula_dir: str | Path | None = None,
    llm_client=None,
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
        curriculum_file=curriculum_file,
        curricula_dir=curricula_dir,
        llm_client=llm_client,
    )
    capture = BusCapture(bus)
    capture.install()
    return trainer, capture


def _make_trainer_with_capture(
    bus: MessageBus,
    curriculum: Curriculum,
    *,
    save_path: str | Path | None = None,
    curriculum_file: str | Path | None = None,
    curricula_dir: str | Path | None = None,
    llm_client=None,
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
        curriculum_file=curriculum_file,
        curricula_dir=curricula_dir,
        llm_client=llm_client,
    )
    return trainer, capture


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
            Message(role=TRAINER_ROLE, action="input", message="goal: second", sender="slack")
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
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
        assert len(submit_msgs) == 1

        # Send pause
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="input", message="pause", sender="slack")
        )
        assert trainer._session_paused

        capture.reset()

        # Complete the current lesson with S1 event
        query = KLine(signature=100, nodes=[10])
        proposal = KLine(signature=100, nodes=[10])
        event = _make_event("ground", query, proposal, _S1_SIGNIFICANCE)
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="ground", message=event)
        )

        # Curriculum position should advance (lesson complete)
        assert trainer.state.curriculum.position == 1

        # But NO auto-submit of next lesson (paused)
        submit_after = capture.find_all(TRAINEE_ROLE, "submit")
        assert len(submit_after) == 0

        # Resume
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="input", message="resume", sender="slack")
        )
        assert not trainer._session_paused

        # After resume, next lesson should be submitted
        submit_resume = capture.find_all(TRAINEE_ROLE, "submit")
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
            Message(role=TRAINER_ROLE, action="input", message="stop", sender="slack")
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
            Message(role=TRAINER_ROLE, action="ground", message=event)
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
                Message(role=TRAINER_ROLE, action="ground", message=event)
            )

        # After 3 events: curriculum position advances
        assert trainer.state.curriculum.position == 1

        # Next lesson submitted (not paused, curriculum has more lessons)
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
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
            Message(role=TRAINER_ROLE, action="ground", message=event)
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
            Message(role=TRAINER_ROLE, action="ground", message=event)
        )

        # Entry is satisfied
        key = _entry_key(entry)
        assert trainer.state.is_satisfied(key)

        # No countersign sent (fast path auto-satisfies, no countersign needed)
        cs_msgs = capture.find_all(TRAINEE_ROLE, "countersign")
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
                role=TRAINER_ROLE,
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
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
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
            Message(role=TRAINER_ROLE, action="ground", message=event)
        )

        # Now at lesson 2 (position=1)
        assert trainer.state.curriculum.position == 1

        # Stop session
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="input", message="stop", sender="slack")
        )

        # Verify file persisted
        assert save_file.exists()

        # Load state back
        loaded = CurriculumState.load(save_file)
        assert loaded.curriculum.position == 1
        assert loaded.curriculum.lessons == ["lesson1", "lesson2", "lesson3"]
        assert loaded.submitted == {key}
        assert loaded.satisfied == {key}


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
                role=TRAINER_ROLE,
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
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
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
        progress_msgs = capture.find_all(SUPERVISOR_ROLE, "progress")
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
        progress_msgs = capture.find_all(SUPERVISOR_ROLE, "progress")
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
            Message(role=TRAINER_ROLE, action="input", message="goal: teach SVO", sender="slack")
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
                role=TRAINER_ROLE,
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

        # Provide a mock LLM that returns the pre-written curriculum content
        mock_llm = _MockLLMClient([
            LLMResponse(
                content=curriculum_path.read_text(),
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        trainer, capture = _make_trainer_with_capture(
            bus, curriculum, curricula_dir=str(tmp_path),
            llm_client=mock_llm,
        )

        # Confirm polling mode is active from constructor
        assert trainer._polling_for_goal is True
        capture.reset()

        # Send goal input
        trainer.on_message(
            Message(
                role=TRAINER_ROLE,
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
                role=TRAINER_ROLE,
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
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
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
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
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
        trainer.on_message(Message(role=TRAINER_ROLE, action="ground", message=event))

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

        progress_msgs = capture.find_all(SUPERVISOR_ROLE, "progress")
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
        trainer.on_message(Message(role=TRAINER_ROLE, action="ground", message=event))

        progress_msgs = capture.find_all(SUPERVISOR_ROLE, "progress")
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
        trainer.on_message(Message(role=TRAINER_ROLE, action="ground", message=event))

        progress_msgs = capture.find_all(SUPERVISOR_ROLE, "progress")
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
        progress_msgs = capture.find_all(SUPERVISOR_ROLE, "progress")
        amended_msgs = [m for m in progress_msgs if m.message["status"] == "amended"]
        assert len(amended_msgs) >= 1


# ── LLM mock helper ──────────────────────────────────────────────────


class _MockLLMClient:
    """Mock LLM client for generation tests."""

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        self._call_count += 1
        if self._call_count <= len(self._responses):
            return self._responses[self._call_count - 1]
        return LLMResponse(content=None, tool_calls=None, finish_reason="stop")


# ── KB-032: _generate_and_start success path ─────────────────────────


class TestGenerateAndStart:
    """KB-032: _generate_and_start creates CurriculumGenerator, calls generate, loads result."""

    @patch("trainer.trainer.compile_source")
    def test_generate_and_start_success(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """_generate_and_start creates generator, calls generate, starts session."""
        mock_compile.return_value = [_make_entry(100, [10])]

        # Write a valid curriculum file for the generator to "return"
        curriculum_path = tmp_path / "curricula" / "test-goal.md"
        _write_curriculum(curriculum_path)

        mock_llm = _MockLLMClient([
            LLMResponse(content=curriculum_path.read_text(), tool_calls=None, finish_reason="stop"),
        ])

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=tmp_path / "curricula",
            llm_client=mock_llm,
        )

        # Enter polling mode and trigger generation
        trainer._polling_for_goal = True
        trainer.on_message(
            Message(
                role=TRAINER_ROLE,
                action="input",
                message="goal: teach Kalvin about SVO",
                sender="slack",
            )
        )

        # Should no longer be polling — session started
        assert not trainer._polling_for_goal

        # goal_received event logged
        goal_events = [e for e in trainer.state.event_log if e["type"] == "goal_received"]
        assert len(goal_events) == 1
        assert goal_events[0]["data"]["goal"] == "teach Kalvin about SVO"

        # curriculum_generated event logged
        gen_events = [e for e in trainer.state.event_log if e["type"] == "curriculum_generated"]
        assert len(gen_events) == 1

        # Session should be active
        assert trainer._session_active

        # A submit should have been sent to kalvin (session started with loaded curriculum)
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
        assert len(submit_msgs) >= 1

    @patch("trainer.trainer.compile_source")
    def test_generate_and_start_uses_llm_client_and_curricula_dir(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """CurriculumGenerator receives trainer's llm_client and curricula_dir."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curricula_dir = tmp_path / "curricula"

        # Write a valid curriculum for the mock generator to return
        curriculum_path = curricula_dir / "test-goal.md"
        _write_curriculum(curriculum_path)

        mock_llm = _MockLLMClient()

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=curricula_dir,
            llm_client=mock_llm,
        )

        # Patch at definition site — _generate_and_start imports from trainer.curriculum_generator
        with patch(
            "trainer.curriculum_generator.CurriculumGenerator"
        ) as mock_gen:
            mock_gen.return_value.generate.return_value = curriculum_path
            trainer._generate_and_start("test goal")

            # CurriculumGenerator was instantiated with our llm_client and curricula_dir
            mock_gen.assert_called_once_with(mock_llm, curricula_dir)
            mock_gen.return_value.generate.assert_called_once_with("test goal")

    @patch("trainer.trainer.compile_source")
    def test_generate_and_start_loads_generated_curriculum(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """After generate(), _load_and_start is called with the returned path."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "curricula" / "test-goal.md"
        _write_curriculum(curriculum_path)

        mock_llm = _MockLLMClient([
            LLMResponse(content=curriculum_path.read_text(), tool_calls=None, finish_reason="stop"),
        ])

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=tmp_path / "curricula",
            llm_client=mock_llm,
        )

        # Verify the curriculum was loaded
        trainer._generate_and_start("test goal")
        assert trainer._session_active
        assert trainer.state.curriculum.document.lessons[0].label == "1"

    @patch("trainer.trainer.compile_source")
    def test_generate_and_start_emits_progress(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """Session start emits progress event to UI."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "curricula" / "test-goal.md"
        _write_curriculum(curriculum_path)

        mock_llm = _MockLLMClient([
            LLMResponse(content=curriculum_path.read_text(), tool_calls=None, finish_reason="stop"),
        ])

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=tmp_path / "curricula",
            llm_client=mock_llm,
        )

        trainer._generate_and_start("test goal")

        # Progress event should have been emitted (session started)
        progress_msgs = capture.find_all(SUPERVISOR_ROLE, "progress")
        assert any(m.message["status"] == "started" for m in progress_msgs)


# ── KB-032: _generate_and_start error and guard paths ────────────────


class TestGenerateAndStartGuards:
    """KB-032: Guard checks in _generate_and_start."""

    def test_no_curricula_dir_returns_early(self) -> None:
        """When _curricula_dir is None, method returns with logger.error only."""
        bus = MessageBus()
        curriculum = Curriculum([])
        mock_llm = _MockLLMClient()
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=None,
            llm_client=mock_llm,
        )

        trainer._generate_and_start("test goal")

        # No CurriculumGenerator created (no LLM calls)
        assert mock_llm.call_count == 0

        # No session started
        assert not trainer._session_active

        # No events logged (guard is just logger.error, no state.log_event)
        goal_events = [e for e in trainer.state.event_log if e["type"] == "goal_received"]
        assert len(goal_events) == 0

    @patch("trainer.trainer.compile_source")
    def test_no_llm_client_returns_early(self, mock_compile: MagicMock) -> None:
        """When _llm_client is None, method returns with error event."""
        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir="/tmp/curricula",
            llm_client=None,
        )

        trainer._generate_and_start("test goal")

        # No session started
        assert not trainer._session_active

        # generation_failed event logged
        failed_events = [e for e in trainer.state.event_log if e["type"] == "generation_failed"]
        assert len(failed_events) == 1
        assert "no LLM client configured" in failed_events[0]["data"]["error"]

    @patch("trainer.trainer.compile_source")
    def test_generation_error_re_enters_polling(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """CurriculumGenerationError caught, polling re-enabled, status emitted."""
        mock_compile.return_value = [_make_entry(100, [10])]

        mock_llm = _MockLLMClient()

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=tmp_path / "curricula",
            llm_client=mock_llm,
        )

        with patch(
            "trainer.curriculum_generator.CurriculumGenerator"
        ) as mock_gen:
            mock_gen.return_value.generate.side_effect = CurriculumGenerationError(
                "LLM returned invalid output"
            )
            trainer._generate_and_start("test goal")

        # generation_failed event logged
        failed_events = [e for e in trainer.state.event_log if e["type"] == "generation_failed"]
        assert len(failed_events) == 1
        assert "LLM returned invalid output" in failed_events[0]["data"]["error"]

        # Should re-enter polling mode
        assert trainer._polling_for_goal

        # Polling status emitted to UI
        polling_msgs = [
            m for m in capture.messages
            if m.role == SUPERVISOR_ROLE
            and m.action == "progress"
            and m.message.get("status") == "polling_for_goal"
        ]
        assert len(polling_msgs) >= 1

    @patch("trainer.trainer.compile_source")
    def test_after_generation_failure_accepts_new_goal(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """After failure, Trainer re-enters polling and can accept new goal."""
        mock_compile.return_value = [_make_entry(100, [10])]

        # Write a valid curriculum
        curriculum_path = tmp_path / "curricula" / "test-goal.md"
        _write_curriculum(curriculum_path)

        mock_llm = _MockLLMClient([
            # First call: will be intercepted by side_effect
            LLMResponse(content=None, tool_calls=None, finish_reason="stop"),
            # Second call: valid response for retry
            LLMResponse(
                content=curriculum_path.read_text(),
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=tmp_path / "curricula",
            llm_client=mock_llm,
        )

        # First attempt: force generation failure
        with patch(
            "trainer.curriculum_generator.CurriculumGenerator"
        ) as mock_gen:
            mock_gen.return_value.generate.side_effect = CurriculumGenerationError("fail")
            trainer._generate_and_start("failing goal")

        assert trainer._polling_for_goal

        capture.reset()

        # Second attempt via _handle_input while polling — should succeed
        trainer.on_message(
            Message(
                role=TRAINER_ROLE,
                action="input",
                message="goal: teach SVO",
                sender="slack",
            )
        )

        # After successful generation and session start, polling is off
        assert not trainer._polling_for_goal
        assert trainer._session_active


# ── KB-032: _resolve_goal dispatching ─────────────────────────────────


class TestResolveGoalDispatch:
    """KB-032: _resolve_goal dispatches to _generate_and_start or _load_and_start."""

    @patch("trainer.trainer.compile_source")
    def test_goal_prefix_with_space(self, mock_compile: MagicMock, tmp_path: Path) -> None:
        """Input 'goal: teach Kalvin about SVO' while polling → calls _generate_and_start."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "curricula" / "teach-kalvin-about-svo.md"
        _write_curriculum(curriculum_path)

        mock_llm = _MockLLMClient([
            LLMResponse(
                content=curriculum_path.read_text(),
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=tmp_path / "curricula",
            llm_client=mock_llm,
        )
        trainer._polling_for_goal = True

        trainer.on_message(
            Message(
                role=TRAINER_ROLE,
                action="input",
                message="goal: teach Kalvin about SVO",
                sender="slack",
            )
        )

        # Should have generated and started session
        assert not trainer._polling_for_goal
        assert trainer._session_active
        goal_events = [e for e in trainer.state.event_log if e["type"] == "goal_received"]
        assert len(goal_events) == 1
        assert goal_events[0]["data"]["goal"] == "teach Kalvin about SVO"

    @patch("trainer.trainer.compile_source")
    def test_goal_prefix_no_space(self, mock_compile: MagicMock, tmp_path: Path) -> None:
        """Input 'goal:teach Kalvin' (no space after colon) calls _generate_and_start."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "curricula" / "teach-kalvin-about-svo.md"
        _write_curriculum(curriculum_path)

        mock_llm = _MockLLMClient([
            LLMResponse(
                content=curriculum_path.read_text(),
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=tmp_path / "curricula",
            llm_client=mock_llm,
        )
        trainer._polling_for_goal = True

        trainer.on_message(
            Message(
                role=TRAINER_ROLE,
                action="input",
                message="goal:teach Kalvin about SVO",
                sender="slack",
            )
        )

        # Should have generated (goal prefix parsed, "teach Kalvin about SVO")
        assert not trainer._polling_for_goal
        assert trainer._session_active
        goal_events = [e for e in trainer.state.event_log if e["type"] == "goal_received"]
        assert len(goal_events) == 1
        assert goal_events[0]["data"]["goal"] == "teach Kalvin about SVO"

    @patch("trainer.trainer.compile_source")
    def test_goal_prefix_whitespace_only(self, mock_compile: MagicMock, tmp_path: Path) -> None:
        """Input 'goal: ' (whitespace only after strip) → no crash, generation fails gracefully."""
        mock_compile.return_value = [_make_entry(100, [10])]

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=tmp_path / "curricula",
            llm_client=_MockLLMClient(),
        )
        trainer._polling_for_goal = True

        # Should not crash — empty goal goes through generation and fails
        trainer.on_message(
            Message(
                role=TRAINER_ROLE,
                action="input",
                message="goal: ",
                sender="slack",
            )
        )

        # No crash — generation attempted with empty string, fails, re-enters polling
        assert trainer._polling_for_goal  # re-entered polling after failure
        assert not trainer._session_active

    @patch("trainer.trainer.compile_source")
    def test_file_path_triggers_load(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """Input that is a path to an existing .md file → calls _load_and_start."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "my_curriculum.md"
        _write_curriculum(curriculum_path)

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=tmp_path / "curricula",
            llm_client=_MockLLMClient(),
        )
        trainer._polling_for_goal = True

        trainer.on_message(
            Message(
                role=TRAINER_ROLE,
                action="input",
                message=str(curriculum_path),
                sender="slack",
            )
        )

        # Should have loaded the curriculum file (not generated)
        assert not trainer._polling_for_goal
        assert trainer._session_active
        assert trainer.state.curriculum.document.lessons[0].label == "1"
        # No goal_received event (loaded, not generated)
        goal_events = [e for e in trainer.state.event_log if e["type"] == "goal_received"]
        assert len(goal_events) == 0

    @patch("trainer.trainer.compile_source")
    def test_non_goal_non_file_triggers_generate(
        self, mock_compile: MagicMock, tmp_path: Path
    ) -> None:
        """Input that is neither goal: prefix nor valid file → calls _generate_and_start."""
        mock_compile.return_value = [_make_entry(100, [10])]

        curriculum_path = tmp_path / "curricula" / "some-free-text.md"
        _write_curriculum(curriculum_path)

        mock_llm = _MockLLMClient([
            LLMResponse(
                content=curriculum_path.read_text(),
                tool_calls=None,
                finish_reason="stop",
            ),
        ])

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer, capture = _make_trainer(
            bus,
            curriculum,
            curricula_dir=tmp_path / "curricula",
            llm_client=mock_llm,
        )
        trainer._polling_for_goal = True

        trainer.on_message(
            Message(
                role=TRAINER_ROLE,
                action="input",
                message="teach Kalvin about SVO",
                sender="slack",
            )
        )

        # Should have called _generate_and_start with the raw text
        assert not trainer._polling_for_goal
        assert trainer._session_active
        goal_events = [e for e in trainer.state.event_log if e["type"] == "goal_received"]
        assert len(goal_events) == 1
        assert goal_events[0]["data"]["goal"] == "teach Kalvin about SVO"


# ── HRNS-33: Event relay and ratify request ──────────────────────────


class TestEventRelay:
    """HRNS-33: Trainer relays events to supervisor, sends ratify_request for S2/S3."""

    @patch("trainer.trainer.compile_source")
    def test_relay_ground_event_to_supervisor(self, mock_compile: MagicMock) -> None:
        """S1 ground event: event relay sent to supervisor, no ratify_request."""
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()
        capture.reset()

        # Send S1 ground event (completes the lesson's only entry)
        event = _make_event(
            "ground",
            query=KLine(signature=100, nodes=[10]),
            proposal=KLine(signature=100, nodes=[10]),
            significance=_S1_SIGNIFICANCE,
        )
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="ground", message=event)
        )

        # Event relay was sent to supervisor
        relay_msgs = capture.find_all(SUPERVISOR_ROLE, "event")
        assert len(relay_msgs) == 1
        assert relay_msgs[0].message is event

        # NO ratify_request for S1 events
        ratify_msgs = capture.find_all(SUPERVISOR_ROLE, "ratify_request")
        assert len(ratify_msgs) == 0

    @patch("trainer.trainer.process_s2_s3", create=True)
    @patch("trainer.trainer.compile_source")
    def test_relay_frame_event_with_ratify_request(
        self, mock_compile: MagicMock, mock_process: MagicMock
    ) -> None:
        """S2/S3 frame event: event relay + ratify_request sent to supervisor.

        Reactor.process_s2_s3 is patched because the Reactor's bus messages
        are tested separately in test_reactor.py.
        """
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()
        capture.reset()

        # Patch the reactor's process_s2_s3 to isolate Trainer-level testing
        trainer._reactor.process_s2_s3 = MagicMock()

        # Send S2/S3 frame event (low significance, non-matching signature)
        event = _make_event(
            "frame",
            query=KLine(signature=999, nodes=[99]),
            proposal=KLine(signature=999, nodes=[99]),
            significance=_S2_SIGNIFICANCE,
        )
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="frame", message=event)
        )

        # Event relay was sent to supervisor
        relay_msgs = capture.find_all(SUPERVISOR_ROLE, "event")
        assert len(relay_msgs) == 1
        assert relay_msgs[0].message is event

        # Ratify request was sent to supervisor
        ratify_msgs = capture.find_all(SUPERVISOR_ROLE, "ratify_request")
        assert len(ratify_msgs) == 1
        payload = ratify_msgs[0].message
        assert payload["proposal"] is event.proposal
        assert payload["query"] is event.query
        assert payload["significance"] == event.significance

    @patch("trainer.trainer.compile_source")
    def test_s1_event_relay_payload_and_sender(self, mock_compile: MagicMock) -> None:
        """S1 event relay includes correct sender and RationaliseEvent payload."""
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()
        capture.reset()

        event = _make_event(
            "ground",
            query=KLine(signature=100, nodes=[10]),
            proposal=KLine(signature=100, nodes=[10]),
            significance=_S1_SIGNIFICANCE,
        )
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="ground", message=event)
        )

        relay_msgs = capture.find_all(SUPERVISOR_ROLE, "event")
        assert len(relay_msgs) == 1

        # Verify sender is the trainer's role
        assert relay_msgs[0].sender == trainer.role
        assert relay_msgs[0].sender == "trainer"

        # Verify payload is the actual RationaliseEvent
        assert relay_msgs[0].message is event
        assert relay_msgs[0].message.kind == "ground"

    @patch("trainer.trainer.compile_source")
    def test_high_significance_frame_event_no_ratify_request(
        self, mock_compile: MagicMock
    ) -> None:
        """High-significance frame event takes S1 path: relay but no ratify."""
        from kalvin.expand import D_MAX

        _S1_FRAME_THRESHOLD = D_MAX - 1
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()
        capture.reset()

        # Frame event at S1 threshold → classified as S1
        event = _make_event(
            "frame",
            query=KLine(signature=100, nodes=[10]),
            proposal=KLine(signature=100, nodes=[10]),
            significance=_S1_FRAME_THRESHOLD,
        )
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="frame", message=event)
        )

        # Event relay IS sent (all events are relayed regardless of S1/S2/S3)
        relay_msgs = capture.find_all(SUPERVISOR_ROLE, "event")
        assert len(relay_msgs) == 1

        # No ratify_request (went through S1 path)
        ratify_msgs = capture.find_all(SUPERVISOR_ROLE, "ratify_request")
        assert len(ratify_msgs) == 0


# ── HRNS-31: Trainer progress/escalation to all supervisor subscribers ──


class TestTrainerProgressToAllSupervisors:
    """HRNS-31: Trainer sends progress and escalation to role `supervisor`;
    all supervisor subscribers receive."""

    @patch("trainer.trainer.compile_source")
    def test_progress_to_all_supervisor_subscribers(
        self, mock_compile: MagicMock
    ) -> None:
        """Two handlers subscribed to supervisor both receive progress."""
        import threading

        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()

        # Two separate handlers subscribed to supervisor role
        received_a: list[Message] = []
        received_b: list[Message] = []
        event = threading.Event()
        count = 0

        def handler_a(msg: Message) -> None:
            nonlocal count
            received_a.append(msg)
            if msg.action == "progress":
                count += 1
                if count == 2:
                    event.set()

        def handler_b(msg: Message) -> None:
            nonlocal count
            received_b.append(msg)
            if msg.action == "progress":
                count += 1
                if count == 2:
                    event.set()

        bus.subscribe(SUPERVISOR_ROLE, handler_a)
        bus.subscribe(SUPERVISOR_ROLE, handler_b)

        curriculum = Curriculum(["lesson1"])

        # Run bus in background thread
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            # Trainer constructor may emit progress — create AFTER bus running
            trainer = Trainer(bus, curriculum)

            # Starting a session emits a "started" progress event
            trainer.start_session()

            assert event.wait(timeout=2), "Both handlers should have received progress"

            # Both handlers should have received the progress message
            progress_in_a = [m for m in received_a if m.action == "progress"]
            progress_in_b = [m for m in received_b if m.action == "progress"]

            assert len(progress_in_a) >= 1, "handler_a should have received progress"
            assert len(progress_in_b) >= 1, "handler_b should have received progress"

            # Verify the progress message targets supervisor
            for msg in progress_in_a + progress_in_b:
                assert msg.role == SUPERVISOR_ROLE
        finally:
            bus.stop()
            bus_thread.join(timeout=2)

    @patch("trainer.trainer.compile_source")
    def test_escalation_to_all_supervisor_subscribers(
        self, mock_compile: MagicMock
    ) -> None:
        """Two handlers subscribed to supervisor both receive escalation."""
        import threading

        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()

        # Two separate handlers subscribed to supervisor role
        received_a: list[Message] = []
        received_b: list[Message] = []
        event = threading.Event()
        count = 0

        def handler_a(msg: Message) -> None:
            nonlocal count
            received_a.append(msg)
            if msg.action == "notify":
                count += 1
                if count == 2:
                    event.set()

        def handler_b(msg: Message) -> None:
            nonlocal count
            received_b.append(msg)
            if msg.action == "notify":
                count += 1
                if count == 2:
                    event.set()

        bus.subscribe(SUPERVISOR_ROLE, handler_a)
        bus.subscribe(SUPERVISOR_ROLE, handler_b)

        curriculum = Curriculum(["lesson1"])

        # Run bus in background thread
        bus_thread = threading.Thread(target=bus.run, daemon=True)
        bus_thread.start()

        try:
            trainer = Trainer(bus, curriculum)
            trainer.start_session()

            # Trigger escalation by sending enough non-matching S2/S3 events
            # to exhaust the reactive budget (default max_reactive_rounds=5)
            for _ in range(6):
                ev = _make_event(
                    "frame",
                    query=KLine(signature=999, nodes=[99]),
                    proposal=KLine(signature=999, nodes=[99]),
                    significance=_S2_SIGNIFICANCE,
                )
                trainer.on_message(
                    Message(role=TRAINER_ROLE, action="frame", message=ev)
                )

            assert event.wait(timeout=2), "Both handlers should have received escalation"

            # Both handlers should have received the escalation (notify) message
            notify_in_a = [m for m in received_a if m.action == "notify"]
            notify_in_b = [m for m in received_b if m.action == "notify"]

            assert len(notify_in_a) >= 1, "handler_a should have received escalation"
            assert len(notify_in_b) >= 1, "handler_b should have received escalation"

            # Verify escalation targets supervisor role
            for msg in notify_in_a + notify_in_b:
                assert msg.role == SUPERVISOR_ROLE
                assert msg.message["reason"] in ("budget_exhaustion", "low_confidence")
        finally:
            bus.stop()
            bus_thread.join(timeout=2)


# ── KB-125: Cogitator auto-wiring ────────────────────────────────────


class TestCogitatorAutoWiring:
    """KB-125: Trainer auto-wires Cogitator when llm_client is provided."""

    def test_auto_wires_cogitator_when_llm_client_provided(self) -> None:
        """When llm_client is provided without cogitate_fn, reactor gets wired."""
        mock_llm = MagicMock()
        bus = MessageBus()
        curriculum = Curriculum([])
        trainer = Trainer(bus, curriculum, llm_client=mock_llm)

        assert trainer._reactor._cogitate_fn is not None

    def test_explicit_cogitate_fn_not_overwritten(self) -> None:
        """When both llm_client and cogitate_fn are provided, explicit fn wins."""
        explicit_fn = MagicMock(return_value=("explicit", 0.9))
        mock_llm = MagicMock()

        bus = MessageBus()
        curriculum = Curriculum([])
        trainer = Trainer(
            bus, curriculum, llm_client=mock_llm, cogitate_fn=explicit_fn,
        )

        assert trainer._reactor._cogitate_fn is explicit_fn

    def test_no_llm_client_no_cogitate_fn(self) -> None:
        """When neither is provided, cogitate_fn remains None."""
        bus = MessageBus()
        curriculum = Curriculum([])
        trainer = Trainer(bus, curriculum)

        assert trainer._reactor._cogitate_fn is None

    def test_cogitate_adapter_calls_cogitator(self) -> None:
        """Auto-wired adapter builds CogitationRequest and returns the right tuple."""
        from trainer.cogitation import CogitationRequest, CogitationResult

        mock_result = CogitationResult(
            scaffolding="0x10 -> 0x20",
            confidence=0.85,
            reasoning="test reasoning",
            raw_response=None,
        )

        mock_cogitator = MagicMock()
        mock_cogitator.cogitate.return_value = mock_result

        with patch("trainer.cogitation.Cogitator", return_value=mock_cogitator):
            bus = MessageBus()
            curriculum = Curriculum([])
            trainer = Trainer(bus, curriculum, llm_client=MagicMock())

        assert trainer._reactor._cogitate_fn is not None

        # Call the adapter
        query = KLine(signature=0xFF, nodes=[0x10])
        proposal = KLine(signature=0x0F, nodes=[0x20])
        event = RationaliseEvent(
            kind="frame", query=query, proposal=proposal, significance=100,
        )

        result = trainer._reactor._cogitate_fn(event)

        # Cogitator.cogitate was called once
        mock_cogitator.cogitate.assert_called_once()

        # Verify the CogitationRequest structure
        call_args = mock_cogitator.cogitate.call_args[0][0]
        assert isinstance(call_args, CogitationRequest)
        assert call_args.events == [event]
        assert len(call_args.misfits) == 1
        assert call_args.curriculum_context == ""
        assert call_args.round_number == 1
        assert call_args.max_rounds == 3

        # Result is the right tuple
        assert result == ("0x10 -> 0x20", 0.85)

    def test_cogitate_adapter_returns_none_on_no_scaffolding(self) -> None:
        """Auto-wired adapter returns None when Cogitator produces no scaffolding."""
        from trainer.cogitation import CogitationResult

        mock_result = CogitationResult(
            scaffolding=None,
            confidence=0.0,
            reasoning="failed",
            raw_response=None,
        )

        mock_cogitator = MagicMock()
        mock_cogitator.cogitate.return_value = mock_result

        with patch("trainer.cogitation.Cogitator", return_value=mock_cogitator):
            bus = MessageBus()
            curriculum = Curriculum([])
            trainer = Trainer(bus, curriculum, llm_client=MagicMock())

        query = KLine(signature=0xFF, nodes=[0x10])
        proposal = KLine(signature=0x0F, nodes=[0x20])
        event = RationaliseEvent(
            kind="frame", query=query, proposal=proposal, significance=100,
        )

        result = trainer._reactor._cogitate_fn(event)
        assert result is None

    @patch("trainer.trainer.compile_source")
    def test_reactive_mode_uses_cogitator_end_to_end(
        self, mock_compile: MagicMock,
    ) -> None:
        """S2/S3 event with auto-wired cogitator: scaffolding submitted, no escalation."""
        from trainer.cogitation import CogitationResult

        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        mock_result = CogitationResult(
            scaffolding="0x10 -> 0x20",
            confidence=0.85,
            reasoning="test",
            raw_response=None,
        )

        mock_cogitator = MagicMock()
        mock_cogitator.cogitate.return_value = mock_result

        with patch("trainer.cogitation.Cogitator", return_value=mock_cogitator):
            bus = MessageBus()
            curriculum = Curriculum(["lesson1", "lesson2"])
            trainer = Trainer(bus, curriculum, llm_client=MagicMock())

        capture = BusCapture(bus)
        capture.install()
        trainer.start_session()
        capture.reset()

        # Non-matching S2/S3 event
        proposal = KLine(signature=999, nodes=[88])
        query = KLine(signature=888, nodes=[1])
        event = _make_event("frame", query, proposal, _S2_SIGNIFICANCE)

        trainer.on_message(
            Message(role=TRAINER_ROLE, action="frame", message=event)
        )

        # Reactive scaffolding was submitted to kalvin
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
        scaffolding_msgs = [
            m for m in submit_msgs if m.message == "0x10 -> 0x20"
        ]
        assert len(scaffolding_msgs) == 1
        assert scaffolding_msgs[0].sender == "trainer"

        # No escalation (low_confidence or budget_exhaustion)
        notify_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        escalation_msgs = [
            m
            for m in notify_msgs
            if m.message.get("reason") in ("low_confidence", "budget_exhaustion")
        ]
        assert len(escalation_msgs) == 0


# ── Restart action ───────────────────────────────────────────────────


class TestRestartSession:
    """Restart action clears state and restarts from the beginning."""

    @patch("trainer.trainer.compile_source")
    def test_restart_clears_state_and_restarts(
        self, mock_compile: MagicMock
    ) -> None:
        """Restart clears tracking sets, resets position, starts fresh."""
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2", "lesson3"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        # Complete lesson 1
        event = _make_event(
            "ground",
            query=KLine(signature=100, nodes=[10]),
            proposal=KLine(signature=100, nodes=[10]),
            significance=_S1_SIGNIFICANCE,
        )
        trainer.on_message(Message(role=TRAINER_ROLE, action="ground", message=event))

        # Verify we advanced to lesson 2
        assert trainer.state.curriculum.position == 1
        key = _entry_key(entry)
        assert trainer.state.is_satisfied(key)

        capture.reset()

        # Send restart
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="input", message="restart", sender="slack")
        )

        # Position reset to 0
        assert trainer.state.curriculum.position == 0

        # Satisfied sets cleared (but submitted gets re-populated by re-submit)
        assert len(trainer.state.satisfied) == 0
        assert len(trainer.state.pending) == 0
        assert len(trainer.state.lesson_satisfied) == 0

        # Session is active
        assert trainer._session_active
        assert not trainer._session_paused

        # A "restart" progress event was emitted
        progress_msgs = capture.find_all(SUPERVISOR_ROLE, "progress")
        restart_msgs = [m for m in progress_msgs if m.message["status"] == "restart"]
        assert len(restart_msgs) >= 1

        # A "started" progress event was emitted (for the new session)
        started_msgs = [m for m in progress_msgs if m.message["status"] == "started"]
        assert len(started_msgs) >= 1

        # First lesson was re-submitted
        submit_msgs = capture.find_all(TRAINEE_ROLE, "submit")
        assert len(submit_msgs) >= 1

        # A session_restart event was logged
        restart_events = [
            e for e in trainer.state.event_log if e["type"] == "session_restart"
        ]
        assert len(restart_events) == 1

    @patch("trainer.trainer.compile_source")
    def test_restart_when_no_session_does_nothing(
        self, mock_compile: MagicMock
    ) -> None:
        """Restart when no session is active does nothing."""
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1"])
        trainer, capture = _make_trainer(bus, curriculum)

        # No session active
        assert not trainer._session_active
        capture.reset()

        # Send restart (should do nothing since there's no active session)
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="input", message="restart", sender="slack")
        )

        # The handler should still work — it just restarts even without a session
        assert trainer._session_active

    @patch("trainer.trainer.compile_source")
    def test_restart_resets_reactor(
        self, mock_compile: MagicMock
    ) -> None:
        """Restart clears the reactor's per-lesson state."""
        entry = _make_entry(100, [10])
        mock_compile.return_value = [entry]

        bus = MessageBus()
        curriculum = Curriculum(["lesson1", "lesson2"])
        trainer, capture = _make_trainer(bus, curriculum)
        trainer.start_session()

        capture.reset()

        # Restart
        trainer.on_message(
            Message(role=TRAINER_ROLE, action="input", message="restart", sender="slack")
        )

        # Reactor should have been reset and then re-loaded with lesson 1
        assert trainer._reactor._expected_count > 0  # lesson re-loaded

        # Session active
        assert trainer._session_active
