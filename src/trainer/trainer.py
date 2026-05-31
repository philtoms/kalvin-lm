"""Trainer participant — embedded harness component that drives the training loop.

The Trainer submits curriculum lessons to the KAgent, auto-countersigns
structurally matching proposals, enters reactive mode on S2/S3 events
(delegating to a cogitation function from KB-024), escalates to the human
when stuck, and persists curriculum state for restart recovery.

Supports document-based curriculum with file polling, label-based tracking,
session startup resolution, amendment handling, and progress events.

This module is synchronous — it receives messages from the bus dispatch
thread via ``on_message()``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness.bus import MessageBus
from harness.message import Message
from kalvin.events import RationaliseEvent
from kalvin.expand import D_MAX
from kalvin.kline import KLine
from kscript.compiler import compile_source
from kscript.token_encoder import CompiledEntry
from trainer.cogitation import LLMClient
from trainer.curriculum import Curriculum, CurriculumState, EntryKey
from trainer.curriculum_document import (
    CurriculumDocument,
    CurriculumParseError,
    Lesson,
)

logger = logging.getLogger(__name__)

# S1 significance boundary for frame events. A frame event with
# significance at or above this threshold is considered S1 (fast path).
_S1_FRAME_THRESHOLD = D_MAX - 1


class Trainer:
    """Embedded harness participant that drives the training loop.

    Parameters
    ----------
    bus:
        The message bus to subscribe to and send messages on.
    curriculum:
        The :class:`Curriculum` instance with ordered lessons.
    address:
        Bus address for this participant (default ``"trainer"``).
    max_reactive_rounds:
        Maximum reactive scaffolding rounds before budget-exhaustion escalation.
    cogitate_fn:
        Optional cogitation function injected by KB-024. Signature:
        ``(RationaliseEvent) -> tuple[str, float] | None``.
        Returns ``(kscript_source, confidence)`` or ``None`` if no
        scaffolding can be generated.
    llm_client:
        Optional LLM client for curriculum generation and reactive
        scaffolding. Must satisfy the :class:`~trainer.cogitation.LLMClient`
        protocol. When provided, enables goal-based curriculum
        generation via the CurriculumGenerator.
    save_path:
        Optional file path for curriculum state persistence.
    curriculum_file:
        Optional path to the curriculum markdown file for file polling
        and persistence.
    curricula_dir:
        Optional directory for generated curriculum files.
    """

    def __init__(
        self,
        bus: MessageBus,
        curriculum: Curriculum,
        *,
        address: str = "trainer",
        max_reactive_rounds: int = 5,
        cogitate_fn: Callable[[RationaliseEvent], tuple[str, float] | None] | None = None,
        llm_client: Any | None = None,
        save_path: str | Path | None = None,
        curriculum_file: str | Path | None = None,
        curricula_dir: str | Path | None = None,
    ) -> None:
        self._address = address
        self._bus = bus
        self._state = CurriculumState(
            curriculum,
            save_path=save_path,
            curriculum_file=str(curriculum_file) if curriculum_file else None,
        )
        self._max_reactive_rounds = max_reactive_rounds
        self._cogitate_fn = cogitate_fn
        self._llm_client = llm_client
        self._curriculum_file = Path(curriculum_file) if curriculum_file else None
        self._curricula_dir = Path(curricula_dir) if curricula_dir else None

        # Session model fields
        self._session_active: bool = False
        self._session_paused: bool = False
        self._current_entries: list[CompiledEntry] = []
        self._expected_count: int = 0
        self._received_count: int = 0
        self._reactive_rounds: int = 0
        self._pending_goals: list[str] = []
        self._conversation_history: list[str] = []
        self._polling_for_goal: bool = False

        # Register on the bus
        bus.subscribe(self._address, self.on_message)

        # Constructor Path 3: no curriculum file and empty curriculum →
        # enter goal-polling mode immediately.
        if self._curriculum_file is None and self._state.curriculum.total() == 0:
            self._polling_for_goal = True
            self._state.log_event("polling_for_goal", {})
            self._emit_progress("polling_for_goal")

    # ── Participant protocol ──────────────────────────────────────────

    @property
    def address(self) -> str:
        """Bus address for this participant."""
        return self._address

    @property
    def state(self) -> CurriculumState:
        """The curriculum state (for test inspection)."""
        return self._state

    # ── Event classification ──────────────────────────────────────────

    @staticmethod
    def _is_s1(event: RationaliseEvent) -> bool:
        """Return ``True`` if *event* is a fast-path S1 event.

        - ``"ground"`` events are always S1.
        - ``"frame"`` events are S1 if significance >= S1 boundary.
        """
        if event.kind == "ground":
            return True
        if event.kind == "frame" and event.significance >= _S1_FRAME_THRESHOLD:
            return True
        return False

    # ── Message handler ───────────────────────────────────────────────

    def on_message(self, msg: Message) -> None:
        """Route incoming messages by action.

        Note: We route by ``msg.action`` rather than ``msg.sender`` because
        the KAgentAdapter constructs forwarded event messages without setting
        ``sender`` (it defaults to ``None``). Using action as the discriminator
        is robust regardless of sender field population.
        """
        action = msg.action

        # KAgent events (ground/frame) and errors — routed by action
        if action in ("ground", "frame"):
            if not self._session_active:
                logger.debug("Ignoring %s event — no active session", action)
                return
            self._handle_kagent_event(msg)
        elif action == "error":
            if not self._session_active:
                return
            self._handle_kagent_error(msg)
        elif action == "input":
            self._handle_input(msg)
        else:
            logger.warning("Unknown action %r from %s", action, msg.sender)

    # ── KAgent event handling ─────────────────────────────────────────

    def _handle_kagent_event(self, msg: Message) -> None:
        """Process a KAgent ground or frame event."""
        event: RationaliseEvent = msg.message
        self._received_count += 1

        if self._is_s1(event):
            # S1 fast path: auto-satisfy by query match
            key = _entry_key(event.query)
            self._state.mark_satisfied(key)
            self._check_lesson_complete()
        else:
            # S2/S3 slow path: try auto-countersign first, then reactive
            if not self._auto_countersign(event.proposal):
                self._handle_reactive(event)
            self._check_lesson_complete()

    def _handle_kagent_error(self, msg: Message) -> None:
        """Log KAgent error and count toward lesson completion."""
        self._state.log_event("kagent_error", {"message": str(msg.message)})
        self._received_count += 1
        self._check_lesson_complete()

    # ── Input handling (from Slack / human) ───────────────────────────

    def _handle_input(self, msg: Message) -> None:
        """Process human input from Slack."""
        text = str(msg.message).strip()

        if self._polling_for_goal:
            self._resolve_goal(text)
            return

        if text.startswith("goal:") or "=" in text:
            # Goal or KScript source — start or queue session
            goal = text[5:].strip() if text.startswith("goal:") else text
            self.start_session(goal=goal)
        elif text == "pause":
            self._session_paused = True
            self._state.log_event("pause", {})
        elif text == "stop":
            self._end_session()
        elif text == "resume":
            self._session_paused = False
            self._state.log_event("resume", {})
            if not self._session_paused and self._session_active:
                self._submit_next_lesson()
        else:
            # Guidance text for reactive mode context
            self._conversation_history.append(text)

    # ── Goal resolution ───────────────────────────────────────────────

    def _resolve_goal(self, text: str) -> None:
        """Resolve a goal text to a curriculum and start session."""
        self._polling_for_goal = False

        if text.startswith("goal:"):
            # Generate curriculum via LLM
            goal = text[5:].strip()
            self._generate_and_start(goal)
        else:
            # Try as file path
            path = Path(text)
            if path.exists() and path.suffix == ".md":
                self._load_and_start(path)
            else:
                # Treat as goal: prefix for generation
                self._generate_and_start(text)

    def _generate_and_start(self, goal: str) -> None:
        """Generate a curriculum from a goal and start the session."""
        if self._curricula_dir is None:
            logger.error("Cannot generate curriculum: no curricula_dir configured")
            return

        if self._llm_client is None:
            logger.error("Cannot generate curriculum: no LLM client configured")
            self._state.log_event("generation_failed", {
                "goal": goal,
                "error": "no LLM client configured",
            })
            return

        logger.info("Generating curriculum for goal: %s", goal)
        self._state.log_event("goal_received", {"goal": goal})

        try:
            from trainer.curriculum_generator import (
                CurriculumGenerationError,
                CurriculumGenerator,
            )

            generator = CurriculumGenerator(self._llm_client, self._curricula_dir)
            curriculum_path = generator.generate(goal)

            logger.info("Curriculum generated: %s", curriculum_path)
            self._state.log_event("curriculum_generated", {"path": str(curriculum_path)})

            self._load_and_start(curriculum_path)
        except CurriculumGenerationError as exc:
            logger.error("Curriculum generation failed: %s", exc)
            self._state.log_event("generation_failed", {
                "goal": goal,
                "error": str(exc),
            })
        except CurriculumParseError as exc:
            logger.error("Generated curriculum failed to parse: %s", exc)
            self._state.log_event("generation_failed", {
                "goal": goal,
                "error": str(exc),
            })
        except Exception as exc:
            logger.error("Unexpected error during curriculum generation: %s", exc)
            self._state.log_event("generation_failed", {
                "goal": goal,
                "error": str(exc),
            })

        # Re-enter polling mode on any failure so the human can try again
        self._polling_for_goal = True
        self._emit_polling_status()

    def _load_and_start(self, path: Path) -> None:
        """Load a curriculum from a file and start the session."""
        try:
            doc = CurriculumDocument.from_file(path)
            curriculum = Curriculum(doc)
            self._state.curriculum = curriculum
            self._curriculum_file = path
            self._state.curriculum_file = str(path)

            # Start session directly (curriculum is now loaded)
            self._session_active = True
            self._session_paused = False
            self._state.log_event("session_start", {
                "goal": None,
                "curriculum_file": str(path),
            })
            self._emit_progress("started")
            self._submit_next_lesson()

            # Persist state with the curriculum file path
            try:
                self._state.save()
            except ValueError:
                logger.debug("No save path configured — skipping state persistence")
        except CurriculumParseError as exc:
            logger.error("Failed to load curriculum from %s: %s", path, exc)
            self._state.log_event("curriculum_load_error", {"path": str(path), "error": str(exc)})

    # ── Session lifecycle ─────────────────────────────────────────────

    def start_session(self, goal: str | None = None) -> None:
        """Begin a new training session.

        If a session is already active, the goal is queued instead
        (one session at a time — HRNS-16).

        Implements three-path startup resolution:
        1. curriculum_file parameter → load and start
        2. No file but saved state has curriculum_file → resume
        3. No file and no saved state → poll for goal
        """
        if self._session_active:
            if goal:
                self._pending_goals.append(goal)
                self._state.log_event("goal_queued", {"goal": goal})
            return

        # Session startup resolution
        if self._curriculum_file and self._curriculum_file.exists():
            # Path 1: curriculum_file provided
            try:
                doc = CurriculumDocument.from_file(self._curriculum_file)
                self._state.curriculum = Curriculum(doc)
            except CurriculumParseError:
                logger.error("Failed to load curriculum from %s", self._curriculum_file)
                return
        elif self._state.curriculum_file:
            # Path 2: saved state has curriculum_file
            saved_path = Path(self._state.curriculum_file)
            if saved_path.exists():
                try:
                    doc = CurriculumDocument.from_file(saved_path)
                    self._state.curriculum = Curriculum(doc)
                    self._curriculum_file = saved_path
                except CurriculumParseError:
                    logger.error("Failed to resume from %s", saved_path)
        elif not self._state.curriculum.lessons:
            # Path 3: no curriculum resolved — poll for goal instead
            self._session_active = False
            self._polling_for_goal = True
            logger.info("No curriculum resolved — polling for goal")
            self._state.log_event("polling_for_goal", {})
            self._emit_polling_status()
            return

        # Path 3: no curriculum file and no saved state → poll for goal.
        # After path resolution, if the curriculum is still empty,
        # enter polling mode instead of starting a session.
        if self._state.curriculum.total() == 0:
            self._polling_for_goal = True
            self._state.log_event("polling_for_goal", {})
            self._emit_progress("polling_for_goal")
            return

        self._session_active = True
        self._session_paused = False
        if goal:
            self._state.log_event("session_start", {"goal": goal})
        else:
            self._state.log_event("session_start", {"goal": None})

        self._emit_progress("started")
        self._submit_next_lesson()

    # ── File polling ──────────────────────────────────────────────────

    def _poll_curriculum_file(self) -> None:
        """Re-read the curriculum file from disk before each lesson.

        Picks up amendments and new lessons without restart.
        """
        if self._curriculum_file and self._curriculum_file.exists():
            try:
                doc = CurriculumDocument.from_file(self._curriculum_file)
                new_curriculum = Curriculum(doc)

                # Check for new lessons
                old_labels = set(self._state.curriculum.document.all_labels())
                new_labels = set(new_curriculum.document.all_labels())

                if new_labels != old_labels:
                    added = new_labels - old_labels
                    if added:
                        self._state.log_event("amendment_detected", {
                            "new_labels": sorted(added),
                        })
                        self._emit_progress("amended")

                # Update curriculum, preserving position
                pos = self._state.curriculum.position
                new_curriculum.position = pos
                self._state.curriculum = new_curriculum

            except CurriculumParseError as exc:
                logger.warning("Failed to re-read curriculum file: %s", exc)

    # ── Curriculum-driven mode ────────────────────────────────────────

    def _submit_next_lesson(self) -> None:
        """Submit the next lesson from the curriculum to the KAgent."""
        # File polling: re-read before each lesson
        self._poll_curriculum_file()

        lesson = self._state.curriculum.current()
        if lesson is None:
            # Curriculum complete — end session
            self._emit_progress("complete")
            self._end_session()
            return

        # Track lesson label
        current_lesson = self._state.curriculum.current_lesson()
        if current_lesson:
            self._state.mark_lesson_submitted(current_lesson.label)

        # Compile locally to obtain CompiledEntry objects for structural matching
        entries = compile_source(lesson)
        self._current_entries = entries
        self._received_count = 0
        self._expected_count = len(entries)
        self._reactive_rounds = 0

        # Mark each entry as submitted
        for entry in entries:
            key = _entry_key(entry)
            self._state.mark_submitted(key)

        # Send raw KScript source — the adapter compiles independently
        self._bus.send(
            Message(
                address="kalvin",
                action="submit",
                message=lesson,
                sender=self._address,
            )
        )

    # ── Auto-countersign ──────────────────────────────────────────────

    def _auto_countersign(self, proposal: KLine) -> bool:
        """Check structural match and auto-countersign if found.

        Uses ``KLine.__eq__`` (signature + nodes) for structural matching.
        Returns ``True`` if a match was found and countersigned.
        """
        for entry in self._current_entries:
            if entry == proposal:
                key = _entry_key(entry)
                # Guard against duplicate countersigns on already-satisfied entries
                if self._state.is_satisfied(key):
                    return True

                self._bus.send(
                    Message(
                        address="kalvin",
                        action="countersign",
                        message=proposal,
                        sender=self._address,
                    )
                )
                self._state.mark_satisfied(key)
                return True
        return False

    # ── Reactive mode ─────────────────────────────────────────────────

    def _handle_reactive(self, event: RationaliseEvent) -> None:
        """Reactive mode on S2/S3 proposals.

        Increments reactive round counter. Escalates on budget exhaustion.
        Otherwise attempts cogitation for reactive scaffolding.
        """
        self._reactive_rounds += 1

        if self._reactive_rounds >= self._max_reactive_rounds:
            self._escalate("budget_exhaustion")
            return

        scaffolding = self._cogitate(event)
        if scaffolding is not None:
            kscript_source, confidence = scaffolding
            self._state.log_event(
                "reactive_scaffolding",
                {"confidence": confidence, "source": kscript_source},
            )
            # Submit reactive scaffolding to KAgent
            self._bus.send(
                Message(
                    address="kalvin",
                    action="submit",
                    message=kscript_source,
                    sender=self._address,
                )
            )
        else:
            self._escalate("low_confidence")

    def _cogitate(self, event: RationaliseEvent) -> tuple[str, float] | None:
        """Attempt to generate reactive scaffolding.

        Delegates to the injected ``cogitate_fn`` if available.
        Returns ``None`` when no cogitation is available (triggers escalation).
        """
        if self._cogitate_fn is not None:
            return self._cogitate_fn(event)
        return None

    # ── Escalation ────────────────────────────────────────────────────

    def _escalate(self, reason: str, detail: str = "") -> None:
        """Escalate to the human via Slack notify message."""
        self._bus.send(
            Message(
                address="slack",
                action="notify",
                message={
                    "reason": reason,
                    "detail": detail,
                    "lesson_position": self._state.curriculum.position,
                },
                sender=self._address,
            )
        )
        self._state.log_event("escalation", {"reason": reason, "detail": detail})

    # ── Progress events ───────────────────────────────────────────────

    def _emit_polling_status(self) -> None:
        """Emit a polling status event to the UI participant."""
        self._bus.send(
            Message(
                address="ui",
                action="progress",
                message={
                    "lesson_label": None,
                    "lessons_total": 0,
                    "lessons_completed": 0,
                    "status": "polling_for_goal",
                },
                sender=self._address,
            )
        )

    def _emit_progress(self, status: str) -> None:
        """Emit a progress event to the UI participant."""
        current_lesson = self._state.curriculum.current_lesson()
        lesson_label = current_lesson.label if current_lesson else None
        total = self._state.curriculum.total()
        completed = len(self._state.lesson_satisfied)

        self._bus.send(
            Message(
                address="ui",
                action="progress",
                message={
                    "lesson_label": lesson_label,
                    "lessons_total": total,
                    "lessons_completed": completed,
                    "status": status,
                },
                sender=self._address,
            )
        )

    def _emit_polling_status(self) -> None:
        """Emit a polling status event to the UI participant.

        Signals that the Trainer is waiting for a goal input.
        """
        self._bus.send(
            Message(
                address="ui",
                action="progress",
                message={
                    "lesson_label": None,
                    "lessons_total": self._state.curriculum.total(),
                    "lessons_completed": len(self._state.lesson_satisfied),
                    "status": "polling_for_goal",
                },
                sender=self._address,
            )
        )

    # ── Amendment ─────────────────────────────────────────────────────

    def request_amendment(self, action: str, **kwargs: object) -> None:
        """Request an amendment to the running curriculum.

        Delegates to ``CurriculumDocument.amend()`` and re-reads the file.
        """
        if self._curriculum_file and self._curriculum_file.exists():
            try:
                doc = CurriculumDocument.from_file(self._curriculum_file)
                lesson = kwargs.get("lesson")
                if lesson and isinstance(lesson, Lesson):
                    doc.amend(action, **kwargs)
                    self._state.log_event("amendment_applied", {
                        "action": action,
                    })
                    self._emit_progress("amended")
                    # Re-read to pick up changes
                    self._poll_curriculum_file()
                    # Restart from first unsatisfied lesson
                    if not self._session_paused and self._session_active:
                        self._submit_next_lesson()
            except (CurriculumParseError, ValueError) as exc:
                logger.error("Amendment failed: %s", exc)
        else:
            logger.warning("Cannot amend: no curriculum file")

    # ── Lesson completion ─────────────────────────────────────────────

    def _check_lesson_complete(self) -> bool:
        """Return ``True`` if the current lesson is complete.

        A lesson is complete when all submitted entries have received
        responses (``_received_count == _expected_count``). Uses exact
        equality to guard against duplicate events causing double-advance.
        """
        if self._received_count == self._expected_count and self._expected_count > 0:
            # Mark lesson satisfied (also advances curriculum position)
            current_lesson = self._state.curriculum.current_lesson()
            old_position = self._state.curriculum.position
            if current_lesson:
                self._state.mark_lesson_satisfied(current_lesson.label)

            self._state.log_event(
                "lesson_complete",
                {"position": old_position},
            )

            self._emit_progress("lesson_complete")

            if not self._session_paused:
                self._submit_next_lesson()
            return True
        return False

    # ── Session end ───────────────────────────────────────────────────

    def _end_session(self) -> None:
        """End the current session, persist state, process queued goals."""
        self._session_active = False
        self._session_paused = False
        self._polling_for_goal = False
        self._state.log_event("session_end", {})

        # Persist state
        try:
            self._state.save()
        except ValueError:
            # No save path configured — skip persistence silently
            logger.debug("No save path configured — skipping state persistence")

        # Process queued goals
        if self._pending_goals:
            next_goal = self._pending_goals.pop(0)
            self.start_session(goal=next_goal)


# ── Module-level helpers ──────────────────────────────────────────────


def _entry_key(kline: KLine) -> EntryKey:
    """Return a hashable identity key for a KLine / CompiledEntry."""
    return (kline.signature, tuple(kline.nodes))
