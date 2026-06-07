"""Reactor — S2/S3 event processing for the Trainer.

The Reactor owns all reactive behaviour during a training session:
auto-countersign matching, reactive scaffolding via an injected
``cogitate_fn``, budget tracking, and escalation to the supervisor
when stuck.

Per-lesson state (current entries, expected/received counts,
reactive round budget) lives here rather than on the Trainer
session driver, keeping the session lifecycle and event-reaction
concerns cleanly separated.

This module is synchronous — the Reactor receives events from the
Trainer driver (which itself runs on the bus dispatch thread).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from harness.bus import MessageBus
from harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE
from harness.message import Message
from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kscript.token_encoder import CompiledEntry
from trainer.curriculum import CurriculumState, EntryKey

logger = logging.getLogger(__name__)


# ── Action dataclass ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Action:
    """Represents a side-effect the reactor wants the driver to perform.

    Currently unused — the reactor sends bus messages directly.
    Retained for potential future use when decoupling reactor
    decisions from bus-side-effects.
    """

    kind: str  # "countersign", "submit", "escalate"
    payload: object  # KLine, kscript source, or escalation reason string
    confidence: float | None = None  # only for "submit" actions from scaffolding


# ── Module-level helpers ─────────────────────────────────────────────


def _entry_key(kline: KLine) -> EntryKey:
    """Return a hashable identity key for a KLine / CompiledEntry."""
    return (kline.signature, tuple(kline.nodes))


# ── Reactor class ────────────────────────────────────────────────────


class Reactor:
    """Owns S2/S3 event processing during a training session.

    Parameters
    ----------
    bus:
        The message bus for sending countersign, submit, and escalation messages.
    state:
        The shared curriculum state for marking entries satisfied and logging events.
    role:
        Bus role of the owning Trainer (used as ``sender`` in bus messages).
    max_reactive_rounds:
        Maximum reactive scaffolding rounds before budget-exhaustion escalation.
    cogitate_fn:
        Optional cogitation function. Signature:
        ``(RationaliseEvent) -> tuple[str, float] | None``.
        Returns ``(kscript_source, confidence)`` or ``None`` if no
        scaffolding can be generated.
    """

    def __init__(
        self,
        bus: MessageBus,
        state: CurriculumState,
        *,
        role: str = "trainer",
        max_reactive_rounds: int = 5,
        cogitate_fn: Callable[[RationaliseEvent], tuple[str, float] | None] | None = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._role = role
        self._max_reactive_rounds = max_reactive_rounds
        self._cogitate_fn = cogitate_fn

        # Per-lesson state
        self._current_entries: list[CompiledEntry] = []
        self._expected_count: int = 0
        self._received_count: int = 0
        self._reactive_rounds: int = 0

    # ── Lesson lifecycle ──────────────────────────────────────────────

    def load_lesson(self, entries: list[CompiledEntry]) -> None:
        """Reset per-lesson state: set entries, zero counters."""
        self._current_entries = entries
        self._received_count = 0
        self._expected_count = len(entries)
        self._reactive_rounds = 0

    # ── Event processing ──────────────────────────────────────────────

    def process_s2_s3(self, event: RationaliseEvent) -> None:
        """Handle an S2/S3 event.

        Tries auto-countersign first; falls through to reactive
        handling (scaffolding or escalation) on no match.
        """
        if not self._auto_countersign(event.proposal):
            self._handle_reactive(event)

    def record_response(self) -> None:
        """Increment the received-event counter."""
        self._received_count += 1

    # ── Completion tracking ───────────────────────────────────────────

    @property
    def is_lesson_complete(self) -> bool:
        """True when received_count == expected_count and > 0."""
        return self._received_count == self._expected_count and self._expected_count > 0

    @property
    def received_count(self) -> int:
        return self._received_count

    @property
    def expected_count(self) -> int:
        return self._expected_count

    @property
    def current_entries(self) -> list[CompiledEntry]:
        return list(self._current_entries)

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
                    logger.debug("Auto-countersign: already satisfied %s", entry)
                    return True

                logger.info("Auto-countersign: proposal matched expectation")
                self._bus.send(
                    Message(
                        role=TRAINEE_ROLE,
                        action="countersign",
                        message=proposal,
                        sender=self._role,
                    )
                )
                self._state.mark_satisfied(key)
                return True
        logger.debug("Auto-countersign: no match for proposal")
        return False

    # ── Reactive mode ─────────────────────────────────────────────────

    def _handle_reactive(self, event: RationaliseEvent) -> None:
        """Reactive mode on S2/S3 proposals.

        Increments reactive round counter. Escalates on budget exhaustion.
        Otherwise attempts cogitation for reactive scaffolding.
        Silently drops events after budget exhaustion to prevent spinning.
        """
        self._reactive_rounds += 1

        if self._reactive_rounds > self._max_reactive_rounds:
            # Already past budget — drop silently (first over-budget event
            # escalated; no need to re-escalate on every subsequent event).
            return

        if self._reactive_rounds >= self._max_reactive_rounds:
            logger.warning(
                "Reactive budget exhausted (%d rounds) — escalating",
                self._reactive_rounds,
            )
            self._escalate("budget_exhaustion")
            return

        scaffolding = self._cogitate(event)
        if scaffolding is not None:
            kscript_source, confidence = scaffolding
            logger.info(
                "Reactive scaffolding (round %d, confidence=%.2f): %s",
                self._reactive_rounds,
                confidence,
                kscript_source[:100],
            )
            self._state.log_event(
                "reactive_scaffolding",
                {"confidence": confidence, "source": kscript_source},
            )
            # Submit reactive scaffolding to KAgent
            self._bus.send(
                Message(
                    role=TRAINEE_ROLE,
                    action="submit",
                    message=kscript_source,
                    sender=self._role,
                )
            )
            logger.info("submitted reactive scaffolding")
        else:
            logger.warning("Cogitation produced no scaffolding — escalating")
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
        """Escalate to supervisor subscribers via notify message."""
        logger.error("Escalation: %s %s", reason, f"({detail})" if detail else "")
        self._bus.send(
            Message(
                role=SUPERVISOR_ROLE,
                action="notify",
                message={
                    "reason": reason,
                    "detail": detail,
                    "lesson_position": self._state.curriculum.position,
                },
                sender=self._role,
            )
        )
        self._state.log_event("escalation", {"reason": reason, "detail": detail})
