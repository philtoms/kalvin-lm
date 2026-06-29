"""Reactor — S2/S3 event processing for the Trainer.

The Reactor owns the Trainer's *mechanical* S2/S3 handling: auto-countersign
of structurally matching proposals and within-lesson recurrence dedup. Every
proposal it cannot resolve itself is surfaced to the Trainer, which emits a
decision request to the supervisor and gates the run until answered
(`@specs/supervisor-decision.md`). The Reactor never cogitates, never submits
reactive scaffolding, and never escalates — those are decider concerns, owned
by a supervisor participant.

Loaded lesson entries and proposal events are KValues (`@kvalue` §Exchange):
the reactor matches structurally (kline-only equality, ignoring significance —
KV-2) and posts the proposal KValue on the countersign bus.

This module is synchronous — the Reactor receives events from the Trainer
driver (which itself runs on the bus dispatch thread).
"""

from __future__ import annotations

import logging

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S4
from kalvin.kvalue import KValue
from training.harness.bus import MessageBus
from training.harness.constants import TRAINEE_ROLE
from training.harness.message import Message
from training.trainer.curriculum import CurriculumState, EntryKey

logger = logging.getLogger(__name__)


# Module-level helpers


def _entry_key(value: KValue) -> EntryKey:
    """Return a hashable identity key for a KValue (from its kline)."""
    return (value.kline.signature, tuple(value.kline.nodes))


# Reactor class


class Reactor:
    """Owns the Trainer's mechanical S2/S3 event handling.

    Two proposals the Reactor resolves itself (never reaching a decider):

    - **Auto-countersign** — a proposal structurally matching a loaded
      expectation. The Reactor sends the countersign and marks the entry
      satisfied (`@specs/supervisor-decision.md` SD-13).
    - **Recurrence** — the same proposal kline seen twice in one lesson
      (intra-expectation fan-out). The second sighting is re-submitted to
      Kalvin at a declared ``SIG_S4`` so rationalise drops it instead of
      re-cogitating indefinitely (SD-14).

    Every other proposal returns ``False`` from :meth:`process_s2_s3`; the
    Trainer then surfaces it as a decision request and gates the run.

    Parameters
    ----------
    bus:
        The message bus for sending countersign and rationalise messages.
    state:
        The shared curriculum state for marking entries satisfied and logging events.
    role:
        Bus role of the owning Trainer (used as ``sender`` in bus messages).
    """

    def __init__(
        self,
        bus: MessageBus,
        state: CurriculumState,
        *,
        role: str = "trainer",
    ) -> None:
        self._bus = bus
        self._state = state
        self._role = role

        self._current_entries: list[KValue] = []
        # Proposals that failed auto-countersign in this lesson, keyed by
        # structural identity (KV-2). A second sighting of the same proposal
        # kline is intra-expectation recurrence — the trainer re-submits it
        # at a declared S4 so Kalvin's rationalise drops it instead of
        # re-cogitating it indefinitely. Reset per lesson (load_lesson).
        self._seen_proposals: set[EntryKey] = set()

    # Lesson lifecycle

    def load_lesson(self, entries: list[KValue]) -> None:
        """Reset per-lesson state: set entries (KValues) and clear the
        seen-proposals set (recurrence is scoped to this lesson).
        """
        self._current_entries = entries
        self._seen_proposals = set()

    # Event processing

    def process_s2_s3(self, event: RationaliseEvent) -> bool:
        """Handle an S2/S3 event.

        Order of precedence:
        1. Auto-countersign on a structural match → return ``True``
           (resolved; no decider needed).
        2. Recurrence (second sighting of a proposal this lesson) → re-submit
           the proposal at a declared ``SIG_S4`` so Kalvin drops it, and
           return ``True``. The first sighting records the proposal; the
           second is the recurrence this branch catches.
        3. Otherwise → return ``False`` (let the Trainer surface the decision
           request and arm the decision gate).

        Returns ``True`` if the proposal was auto-resolved (no supervisor
        interaction needed); ``False`` if it must be escalated to the
        supervisor as a decision (`@specs/supervisor-decision.md` SD-1).
        """
        if self._auto_countersign(event.proposal):
            return True

        key = _entry_key(event.proposal)
        if key in self._seen_proposals:
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
