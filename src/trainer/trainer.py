"""Trainer participant — embedded harness component that drives the training loop.

The Trainer submits curriculum lessons to the KAgent, auto-countersigns
structurally matching proposals, enters reactive mode on S2/S3 events
(delegating to a cogitation function from KB-024), escalates to the human
when stuck, and persists curriculum state for restart recovery.

This module is synchronous — it receives messages from the bus dispatch
thread via ``on_message()``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from harness.bus import MessageBus
from harness.message import Message
from kalvin.events import RationaliseEvent
from kalvin.expand import D_MAX
from kalvin.kline import KLine
from kscript.compiler import compile_source
from kscript.token_encoder import CompiledEntry
from trainer.curriculum import Curriculum, CurriculumState, EntryKey

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
    save_path:
        Optional file path for curriculum state persistence.
    """

    def __init__(
        self,
        bus: MessageBus,
        curriculum: Curriculum,
        *,
        address: str = "trainer",
        max_reactive_rounds: int = 5,
        cogitate_fn: Callable[[RationaliseEvent], tuple[str, float] | None] | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        self._address = address
        self._bus = bus
        self._state = CurriculumState(curriculum, save_path=save_path)
        self._max_reactive_rounds = max_reactive_rounds
        self._cogitate_fn = cogitate_fn

        # Session model fields
        self._session_active: bool = False
        self._session_paused: bool = False
        self._current_entries: list[CompiledEntry] = []
        self._expected_count: int = 0
        self._received_count: int = 0
        self._reactive_rounds: int = 0
        self._pending_goals: list[str] = []
        self._conversation_history: list[str] = []

        # Register on the bus
        bus.subscribe(self._address, self.on_message)

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

    # ── Session lifecycle ─────────────────────────────────────────────

    def start_session(self, goal: str | None = None) -> None:
        """Begin a new training session.

        If a session is already active, the goal is queued instead
        (one session at a time — HRNS-16).
        """
        if self._session_active:
            if goal:
                self._pending_goals.append(goal)
                self._state.log_event("goal_queued", {"goal": goal})
            return

        self._session_active = True
        self._session_paused = False
        if goal:
            self._state.log_event("session_start", {"goal": goal})
        else:
            self._state.log_event("session_start", {"goal": None})

        self._submit_next_lesson()

    # ── Curriculum-driven mode ────────────────────────────────────────

    def _submit_next_lesson(self) -> None:
        """Submit the next lesson from the curriculum to the KAgent."""
        lesson = self._state.curriculum.current()
        if lesson is None:
            # Curriculum complete — end session
            self._end_session()
            return

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

    # ── Lesson completion ─────────────────────────────────────────────

    def _check_lesson_complete(self) -> bool:
        """Return ``True`` if the current lesson is complete.

        A lesson is complete when all submitted entries have received
        responses (``_received_count == _expected_count``). Uses exact
        equality to guard against duplicate events causing double-advance.
        """
        if self._received_count == self._expected_count and self._expected_count > 0:
            self._state.curriculum.advance()
            self._state.log_event(
                "lesson_complete",
                {"position": self._state.curriculum.position - 1},
            )

            if not self._session_paused:
                self._submit_next_lesson()
            return True
        return False

    # ── Session end ───────────────────────────────────────────────────

    def _end_session(self) -> None:
        """End the current session, persist state, process queued goals."""
        self._session_active = False
        self._session_paused = False
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
