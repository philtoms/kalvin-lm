"""Reactor — S2/S3 event processing for the Trainer.

The Reactor owns all reactive behaviour during a training session:
auto-countersign matching, reactive scaffolding via an injected
``cogitate_fn``, budget tracking, and escalation to the supervisor
when stuck.

Per-lesson state (current entries, reactive round budget) lives here
rather than on the Trainer session driver, keeping the session
lifecycle and event-reaction concerns cleanly separated.

Loaded lesson entries and proposal events are KValues (@kvalue spec
§Exchange): the reactor matches structurally (kline-only equality,
ignoring significance — KV-2) and posts the proposal KValue on the
countersign bus.

This module is synchronous — the Reactor receives events from the
Trainer driver (which itself runs on the bus dispatch thread).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S4
from kalvin.kvalue import KValue
from training.harness.bus import MessageBus
from training.harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE
from training.harness.message import Message
from training.trainer.curriculum import CurriculumState, EntryKey

logger = logging.getLogger(__name__)


# Action dataclass


@dataclass(frozen=True)
class Action:
    """Represents a side-effect the reactor wants the driver to perform.

    Currently unused — the reactor sends bus messages directly.
    Retained for potential future use when decoupling reactor
    decisions from bus-side-effects.
    """

    kind: str  # "countersign", "submit", "escalate"
    payload: object  # KValue, KLine, kscript source, or escalation reason string
    confidence: float | None = None  # only for "submit" actions from scaffolding


# Module-level helpers


def _entry_key(value: KValue) -> EntryKey:
    """Return a hashable identity key for a KValue (from its kline)."""
    return (value.kline.signature, tuple(value.kline.nodes))


# Reactor class


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
    delegate_reactive:
        When ``True``, the Reactor enters **delegated mode**: it still
        auto-countersigns structurally matching proposals, but any S2/S3
        proposal that does not auto-countersign produces zero side effects
        — no reactive round increment, no cogitation, no scaffolding
        submission, and no escalation — and ``process_s2_s3`` returns
        ``False`` immediately so the Trainer can defer the decision to the
        supervisor. Distinct from ``cogitate_fn is None`` (which still
        escalates ``low_confidence``); default ``False`` preserves today's
        behaviour.
    """

    def __init__(
        self,
        bus: MessageBus,
        state: CurriculumState,
        *,
        role: str = "trainer",
        max_reactive_rounds: int = 5,
        cogitate_fn: Callable[[RationaliseEvent], tuple[str, float] | None] | None = None,
        delegate_reactive: bool = False,
    ) -> None:
        self._bus = bus
        self._state = state
        self._role = role
        self._max_reactive_rounds = max_reactive_rounds
        self._cogitate_fn = cogitate_fn
        self._delegate_reactive = delegate_reactive

        self._current_entries: list[KValue] = []
        self._reactive_rounds: int = 0
        # Proposals that failed auto-countersign in this lesson, keyed by
        # structural identity (KV-2). A second sighting of the same proposal
        # kline is intra-expectation recurrence — the trainer re-submits it
        # at a declared S4 so Kalvin's rationalise drops it instead of
        # re-cogitating it indefinitely. Reset per lesson (load_lesson).
        self._seen_proposals: set[EntryKey] = set()

    # Lesson lifecycle

    def load_lesson(self, entries: list[KValue]) -> None:
        """Reset per-lesson state: set entries (KValues), zero reactive rounds,
        and clear the seen-proposals set (recurrence is scoped to this lesson).
        """
        self._current_entries = entries
        self._reactive_rounds = 0
        self._seen_proposals = set()

    # Event processing

    def process_s2_s3(self, event: RationaliseEvent) -> bool:
        """Handle an S2/S3 event.

        Order of precedence:
        1. Auto-countersign on a structural match → return True.
        2. Recurrence (second sighting of a proposal this lesson) → increment
           the reactive round, run the shared budget guard, re-submit the
           proposal to Kalvin at a declared S4 so rationalise drops it, and
           return True (no supervisor needed). The first sighting records the
           proposal; the second is the recurrence this branch catches.
        3. Delegated mode (``delegate_reactive=True``) → return False with no
           side effects so the trainer defers to the supervisor.
        4. Reactive handling (scaffolding or escalation) on no match.

        Returns ``True`` if auto-countersign succeeded or recurrence dropped
        the proposal (no supervisor interaction needed). Returns ``False`` if
        reactive handling was invoked (supervisor ratification may be
        required) or delegated mode deferred the decision.
        """
        if self._auto_countersign(event.proposal):
            return True

        # Recurrence: the same proposal kline has already failed
        # auto-countersign this lesson (intra-expectation fan-out — one
        # expectation against two candidates yielding the same reshaped
        # proposal). Re-submit at a declared S4 so Kalvin drops it, and
        # count it toward the reactive budget so the escalation safety net
        # survives a pure-recurrence stall.
        key = _entry_key(event.proposal)
        if key in self._seen_proposals:
            if self._delegate_reactive:
                # Delegated mode: the supervisor already saw this proposal
                # on its first sighting (which returned False → a
                # ratify_request). A recurrence is Kalvin re-emitting the
                # same kline; dedup by dropping it at a declared S4 WITHOUT
                # touching the reactive-round budget. RD-6 forbids
                # budget_exhaustion escalation in delegated mode — the
                # supervisor is the sole decision-maker, so a recurrence
                # must neither re-surface nor escalate. The supervisor's
                # decision on the first sighting stands.
                self._bus.send(
                    Message(
                        role=TRAINEE_ROLE,
                        action="rationalise",
                        message=KValue(event.proposal.kline, SIG_S4),
                        sender=self._role,
                    )
                )
                logger.info(
                    "Recurring proposal re-submitted at declared S4 (delegated dedup)"
                )
                return True
            self._reactive_rounds += 1
            if self._check_budget():
                # At or over the cliff — escalation already sent by
                # _check_budget; drop the proposal without re-submitting.
                return True
            self._bus.send(
                Message(
                    role=TRAINEE_ROLE,
                    action="rationalise",
                    message=KValue(event.proposal.kline, SIG_S4),
                    sender=self._role,
                )
            )
            logger.info("Recurring proposal re-submitted at declared S4 (drop signal)")
            return True
        self._seen_proposals.add(key)

        if self._delegate_reactive:
            return False  # defer to supervisor; no round/cogitate/escalate side effects
        self._handle_reactive(event)
        return False

    # Entry access

    @property
    def current_entries(self) -> list[KValue]:
        return list(self._current_entries)

    # Auto-countersign

    def _auto_countersign(self, proposal: KValue) -> bool:
        """Check structural match and auto-countersign if found.

        Matches the proposal KValue against a loaded expectation using KValue
        structural equality (kline-only, ignoring significance — KV-2). The
        countersign bus message carries the proposal KValue itself (the agreed
        payload contract with the adapter, KB-355).

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

    # Reactive mode

    def _check_budget(self) -> bool:
        """Increment the reactive round counter and guard the budget.

        Shared by the reactive and recurrence paths. Must be called after
        ``self._reactive_rounds`` is incremented by the caller (kept here for
        a single source of truth on the cliff logic).

        Returns ``True`` if the round is at or over the budget cliff — the
        escalation has already been sent and the caller must drop the proposal
        without scaffolding or re-submitting. Returns ``False`` if the round
        is under budget and processing may proceed. Silently drops events past
        the cliff (the first over-budget event escalated; no need to
        re-escalate on every subsequent event).
        """
        if self._reactive_rounds > self._max_reactive_rounds:
            # Already past budget — the first over-budget event escalated.
            return True
        if self._reactive_rounds >= self._max_reactive_rounds:
            logger.warning(
                "Reactive budget exhausted (%d rounds) — escalating",
                self._reactive_rounds,
            )
            self._escalate("budget_exhaustion")
            return True
        return False

    def _handle_reactive(self, event: RationaliseEvent) -> None:
        """Reactive mode on S2/S3 proposals.

        Increments reactive round counter (via the shared budget guard),
        escalates on budget exhaustion, and otherwise attempts cogitation for
        reactive scaffolding.
        """
        self._reactive_rounds += 1
        if self._check_budget():
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

    # Escalation

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
