"""Trainer participant — embedded harness component that drives the training loop.

The Trainer submits curriculum lessons to the KAgent and manages session
lifecycle (start/stop/pause, curriculum loading, file polling, progress
emission, message routing). Reactive decisions — what to do when a proposal
cannot be auto-ratified — are owned by a supervisor participant; the Trainer
surfaces them as decision requests, gates the run until answered, and
applies the answer. The Reactor handles the Trainer's mechanical S2/S3
work (auto-countersign, recurrence dedup).

Supports document-based curriculum with file polling, label-based tracking,
session startup resolution, amendment handling, and progress events.

This module is synchronous — it receives messages from the bus dispatch
thread via ``on_message()``.
"""

from __future__ import annotations

import logging
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

from kalvin.events import RationaliseEvent
from kalvin.expand import D_MAX, normalise_significance
from kalvin.kline import kline_display
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from training.harness.bus import MessageBus
from training.harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE
from training.harness.message import Message
from training.trainer.curriculum import Curriculum, CurriculumState, EntryKey
from training.trainer.curriculum_document import (
    CurriculumDocument,
    CurriculumParseError,
    Lesson,
)
from training.trainer.reactor import Reactor

logger = logging.getLogger(__name__)

# S1 significance boundary for frame events. A frame event with
# significance at or above this threshold is considered S1 (fast path).
_S1_FRAME_THRESHOLD = D_MAX


@lru_cache(maxsize=1)
def _display_tokenizer() -> NLPTokenizer:
    """Lazily-built kalvin tokenizer for kline display (cached; data required)."""
    return NLPTokenizer()


@lru_cache(maxsize=1)
def _display_signifier() -> NLPSignifier:
    """Lazily-built kalvin signifier for kline display (cached)."""
    return NLPSignifier()


class Trainer:
    """Embedded harness participant that drives the training loop.

    Parameters
    ----------
    bus:
        The message bus to subscribe to and send messages on.
    curriculum:
        The :class:`Curriculum` instance with ordered lessons.
    role:
        Bus role for this participant (default ``"trainer"``).
    llm_client:
        Optional LLM client for curriculum generation (goal resolution).
        Must satisfy the :class:`~training.harness.llm.LLMClient` protocol.
        Enables goal-based curriculum generation via the
        :class:`~trainer.curriculum_generator.CurriculumGenerator`. Reactive
        decisions are owned by a supervisor participant (the LLMSupervisor
        is one such participant); the Trainer surfaces decisions and gates
        the run — it never decides reactively (`@specs/supervisor-decision.md`).
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
        role: str = "trainer",
        llm_client: Any | None = None,
        save_path: str | Path | None = None,
        curriculum_file: str | Path | None = None,
        curricula_dir: str | Path | None = None,
        tokenizer: Any | None = None,
        signifier: Any | None = None,
    ) -> None:
        self._role = role
        self._bus = bus
        self._tokenizer = tokenizer
        self._signifier = signifier or NLPSignifier()
        self._state = CurriculumState(
            curriculum,
            save_path=save_path,
            curriculum_file=str(curriculum_file) if curriculum_file else None,
        )
        self._llm_client = llm_client
        self._curriculum_file = Path(curriculum_file) if curriculum_file else None
        self._curricula_dir = Path(curricula_dir) if curricula_dir else None

        # The Reactor owns the Trainer's mechanical S2/S3 handling
        # (auto-countersign + recurrence dedup). Every proposal it cannot
        # resolve itself is surfaced to the supervisor as a decision
        # (`@specs/supervisor-decision.md`). The Trainer never cogitates.
        self._reactor = Reactor(
            bus,
            self._state,
            role=role,
        )

        self._session_active: bool = False
        self._session_paused: bool = False
        self._pending_goals: list[str] = []
        self._conversation_history: list[str] = []
        self._polling_for_goal: bool = False
        self._drain_pending: bool = False

        # Decision gate (SD-7/9). When a ratify_request is emitted
        # the Trainer holds subsequent KAgent events until the supervisor
        # replies (``supervisor_decision`` action). This makes the supervisor
        # the gating decision-maker: the lesson cannot advance
        # (``_check_lesson_complete`` / next-lesson submit are among the held
        # events) until the pending decision is resolved. Each replayed event
        # may raise a new decision, yielding a multi-turn loop. The bus itself
        # never blocks (the handler returns immediately, stashing the event).
        self._pending_decision: RationaliseEvent | None = None
        self._held_messages: deque[Message] = deque()

        bus.subscribe(self._role, self.on_message)

        if self._curriculum_file is not None and self._state.curriculum.total() > 0:
            # Curriculum loaded but not started
            self._emit_progress("ready")
        elif self._curriculum_file is None and self._state.curriculum.total() == 0:
            # No curriculum — enter goal-polling mode
            self._polling_for_goal = True
            self._state.log_event("polling_for_goal", {})
            self._emit_progress("polling_for_goal")

    # Participant protocol

    @property
    def role(self) -> str:
        """Bus role for this participant."""
        return self._role

    @property
    def state(self) -> CurriculumState:
        """The curriculum state (for test inspection)."""
        return self._state

    # Event classification

    @staticmethod
    def _is_s1(event: RationaliseEvent) -> bool:
        """Return ``True`` if *event* is a fast-path S1 event.

        - ``"ground"`` events are always S1.
        - ``"frame"`` events are S1 if significance >= S1 boundary.
        """
        if event.kind == "ground":
            return True
        if event.kind == "frame" and event.proposal.significance >= _S1_FRAME_THRESHOLD:
            return True
        return False

    # Misfit diagnosis

    def _compute_misfit(self, event: RationaliseEvent) -> dict:
        """Compute the misfit diagnosis for an S2/S3 event.

        Operates on the proposal kline (``event.proposal.kline``). The
        ``candidate`` field is gone (KB-354 D5), so misfit is always computed
        on the proposal — the objective structure Kalvin is assessing.

        Returns a dict matching the spec's Decision Request ``misfit``
        field: ``{underfit, overfit, underfit_gap, overfit_mask}``.
        ``underfit``/``overfit`` are ``bool``; the gap/mask are the
        bit differences between the kline signature and its nodes.
        """
        target = event.proposal.kline
        target_underfit, target_overfit = self._signifier.classify_misfit(
            target.signature, target.nodes
        )
        target_nodes_sig = self._signifier.make_signature(target.nodes)
        underfit_gap = self._signifier.residual(target.signature, target_nodes_sig)
        overfit_mask = self._signifier.residual(target_nodes_sig, target.signature)
        return {
            "underfit": target_underfit,
            "overfit": target_overfit,
            "underfit_gap": underfit_gap,
            "overfit_mask": overfit_mask,
        }

    def _compute_curriculum_context(self) -> dict | str:
        """Derive the curriculum context for a decision request.

        Returns ``{objective, approach, lesson_prose}`` mirroring
        :class:`~training.supervisors.llm_supervisor.CogitationRequest`, or a legacy empty
        string when no document is available or all three fields are empty.
        """
        document = self._state.curriculum.document
        objective = document.objective
        approach = document.approach
        current_lesson = self._state.curriculum.current_lesson()
        lesson_prose = current_lesson.prose if current_lesson is not None else ""

        if not (objective or approach or lesson_prose):
            return ""
        return {
            "objective": objective,
            "approach": approach,
            "lesson_prose": lesson_prose,
        }

    # Message handler

    def on_message(self, msg: Message) -> None:
        """Route incoming messages by action.

        Routed by ``msg.action`` (not ``sender``) because the KAgentAdapter
        forwards event messages without setting ``sender``.
        """
        action = msg.action

        # Delegated-decision gate: the supervisor's answer to a pending
        # ratify_request. Resolves the decision and replays held events.
        if action == "supervisor_decision":
            self._handle_supervisor_decision(msg)
            return

        # While a decision is pending, hold KAgent events (and the
        # drained/lesson-advance they trigger) until the supervisor replies.
        # This is what makes the supervisor gating: the run cannot advance
        # past the pending proposal. ``supervisor_decision`` above bypasses
        # the hold so the reply is always processed immediately.
        if self._pending_decision is not None and action in (
            "ground",
            "frame",
            "error",
            "drained",
        ):
            self._held_messages.append(msg)
            return

        if action in ("ground", "frame"):
            if not self._session_active:
                logger.debug("Ignoring %s event — no active session", action)
                return
            self._handle_kagent_event(msg)
        elif action == "error":
            if not self._session_active:
                return
            self._handle_kagent_error(msg)
        elif action == "drained":
            self._handle_drained(msg)
        elif action == "input":
            self._handle_input(msg)
        else:
            logger.warning("Unknown action %r from %s", action, msg.sender)

    # KAgent event handling

    def _handle_kagent_event(self, msg: Message) -> None:
        """Process a KAgent ground or frame event."""
        event: RationaliseEvent = msg.message

        _log_tok = _display_tokenizer()
        _log_sig = _display_signifier()
        try:
            query_src = kline_display(event.query.kline, _log_tok, _log_sig)
        except Exception:
            query_src = repr(event.query)
        try:
            proposal_src = kline_display(event.proposal.kline, _log_tok, _log_sig)
        except Exception:
            proposal_src = repr(event.proposal)

        if event.proposal.significance:
            distance = (~event.proposal.significance) & D_MAX
            sig_norm = normalise_significance(event.proposal.significance)
        else:
            distance = 0
            sig_norm = 0.0

        if self._is_s1(event):
            logger.info(
                "%s %s → S1 (fast path) ← %s",
                event.kind.upper(),
                query_src,
                proposal_src,
            )
        else:
            logger.info(
                "%s %s → %.2f (d=%d) | proposal: %s",
                event.kind.upper(),
                query_src,
                sig_norm,
                distance,
                proposal_src,
            )

        # Relay event to supervisor (HRNS-33)
        self._bus.send(
            Message(
                role=SUPERVISOR_ROLE,
                action="event",
                message=event,
                sender=self._role,
            )
        )

        if self._is_s1(event):
            # S1: the Trainer resolves this itself — auto-ratify by query match.
            key = _entry_key(event.query)
            self._state.mark_satisfied(key)
        else:
            # A proposal the Trainer may not be able to auto-ratify. The
            # Reactor resolves what it can (auto-countersign, recurrence);
            # anything else is escalated to the supervisor as a decision
            # (`@specs/supervisor-decision.md`). The decision gate is
            # unconditional (SD-8): every decision request arms the hold.
            #
            # We do NOT short-circuit on ``is_satisfied(query)``: the
            # Reactor's ``_auto_countersign`` already absorbs genuine
            # structural matches (returning True), so letting a proposal
            # whose query entry happens to be satisfied flow through is
            # safe — matches are absorbed, non-matches surface as
            # decisions the supervisor needs to see.
            auto_matched = self._reactor.process_s2_s3(event)

            # Escalation: a proposal the Reactor could not resolve. The
            # decision request is always enriched with ``misfit`` and
            # ``curriculum_context`` so every decider receives the same
            # context (SD-1). A context-gathering failure must never block
            # the request itself.
            if not auto_matched:
                payload: dict = {
                    "proposal": event.proposal,
                    "query": event.query,
                    "significance": event.proposal.significance,
                }

                try:
                    payload["misfit"] = self._compute_misfit(event)
                except Exception:
                    logger.warning(
                        "Failed to compute misfit for ratify_request",
                        exc_info=True,
                    )
                try:
                    payload["curriculum_context"] = self._compute_curriculum_context()
                except Exception:
                    logger.warning(
                        "Failed to derive curriculum context for ratify_request",
                        exc_info=True,
                    )

                self._bus.send(
                    Message(
                        role=SUPERVISOR_ROLE,
                        action="ratify_request",
                        message=payload,
                        sender=self._role,
                    )
                )

                # Decision gate (SD-4/5/6/7/8): hold subsequent trainee
                # events until the supervisor resolves this decision.
                # "Progress is bounded only by the supervisor's responses"
                # is a runtime guarantee. The bus never blocks —
                # on_message stashes later events into _held_messages
                # and returns immediately.
                self._pending_decision = event

        self._check_lesson_complete()

    def _handle_kagent_error(self, msg: Message) -> None:
        """Log KAgent error and abandon the current lesson.

        An error means the lesson source could not be processed (e.g.
        ParseError). Mark every submitted-but-unsatisfied entry as
        satisfied so the lesson completes — errors must not stall the
        curriculum.
        """
        self._state.log_event("kagent_error", {"message": str(msg.message)})
        # Satisfy all pending entries so _check_lesson_complete can fire.
        unsatisfied = self._state.submitted - self._state.satisfied
        for key in unsatisfied:
            self._state.mark_satisfied(key)
        self._check_lesson_complete()

    # Cogitator drain

    def _handle_drained(self, msg: Message) -> None:
        """Handle the drained response: all previous-lesson cogitation work
        is done, so submit the next lesson."""
        if not self._drain_pending:
            logger.debug("Ignoring unexpected drained event")
            return
        self._drain_pending = False
        logger.info("Cogitator drained — submitting next lesson")
        self._do_submit_lesson()

    def _handle_supervisor_decision(self, msg: Message) -> None:
        """Resolve a pending decision and replay held events.

        The supervisor routes ratify/scaffold/continue to the trainer
        as a ``supervisor_decision`` message when a ratify_request is
        pending. This is the gating point ("progress is bounded only by
        the supervisor's responses"): the decision is applied and the
        held event stream resumes. Replaying a held event may raise a new
        ratify_request, which re-arms the gate (remaining events stay held)
        — yielding the multi-turn decision loop. The bus never blocks.
        """
        if self._pending_decision is None:
            logger.debug("supervisor_decision with no pending decision — ignoring")
            return

        payload = msg.message if isinstance(msg.message, dict) else {}
        decision = payload.get("decision")
        pending = self._pending_decision
        # Clear before applying/replaying so replayed events flow normally
        # (and so a decision with no held events cleanly returns to ready).
        self._pending_decision = None

        if decision == "ratify":
            # Accept the pending proposal: countersign via KAgent (KP-2, S1).
            self._bus.send(
                Message(
                    role=TRAINEE_ROLE,
                    action="countersign",
                    message=pending.proposal,
                    sender=self._role,
                )
            )
            logger.info("Delegated decision: ratified proposal %s", pending.proposal)
        elif decision == "scaffold":
            text = payload.get("text", "")
            if text:
                self._bus.send(
                    Message(
                        role=TRAINEE_ROLE,
                        action="submit",
                        message=text,
                        sender=self._role,
                    )
                )
                logger.info("Delegated decision: scaffolded %d chars of KScript", len(text))
            else:
                logger.warning("Delegated scaffold decision with empty text — skipping")
        elif decision == "continue":
            logger.info("Delegated decision: skipped proposal %s", pending.proposal)
        else:
            logger.warning("Unknown decision %r — treating as skip", decision)

        # Replay held events until the gate re-arms (a new ratify_request
        # sets _pending_decision) or the hold drains. Each replayed event
        # dispatches through on_message, which stashes again if a new
        # decision is now pending.
        while self._held_messages and self._pending_decision is None:
            held = self._held_messages.popleft()
            self.on_message(held)

    # Input handling (from Slack / supervisor)

    def _handle_input(self, msg: Message) -> None:
        """Process supervisor input from the TUI or Slack."""
        text = str(msg.message).strip()

        if self._polling_for_goal:
            self._resolve_goal(text)
            return

        if text == "start":
            if not self._session_active:
                self.start_session()
            else:
                logger.info("Session already active")
        elif text.startswith("goal:") or "=" in text:
            # Goal or KScript source — start or queue session
            goal = text[5:].strip() if text.startswith("goal:") else text
            self.start_session(goal=goal)
        elif text == "pause":
            self._session_paused = True
            self._state.log_event("pause", {})
        elif text == "restart":
            self._restart_session()
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

    # Goal resolution

    def _resolve_goal(self, text: str) -> None:
        """Resolve a goal text to a curriculum and start session."""
        self._polling_for_goal = False

        if text.startswith("goal:"):
            goal = text[5:].strip()
            self._generate_and_start(goal)
        else:
            path = Path(text)
            if path.exists() and path.suffix == ".md":
                self._load_and_start(path)
            else:
                # Otherwise treat as a goal for generation
                self._generate_and_start(text)

    def _generate_and_start(self, goal: str) -> None:
        """Generate a curriculum from a goal and start the session."""
        if self._curricula_dir is None:
            logger.error("Cannot generate curriculum: no curricula_dir configured")
            return

        if self._llm_client is None:
            logger.error("Cannot generate curriculum: no LLM client configured")
            self._state.log_event(
                "generation_failed",
                {
                    "goal": goal,
                    "error": "no LLM client configured",
                },
            )
            return

        logger.info("Generating curriculum for goal: %s", goal)
        self._state.log_event("goal_received", {"goal": goal})

        try:
            from training.trainer.curriculum_generator import (
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
            self._state.log_event(
                "generation_failed",
                {
                    "goal": goal,
                    "error": str(exc),
                },
            )
            self._polling_for_goal = True
            self._emit_polling_status()
        except CurriculumParseError as exc:
            logger.error("Generated curriculum failed to parse: %s", exc)
            self._state.log_event(
                "generation_failed",
                {
                    "goal": goal,
                    "error": str(exc),
                },
            )
            self._polling_for_goal = True
            self._emit_polling_status()
        except Exception as exc:
            logger.error("Unexpected error during curriculum generation: %s", exc)
            self._state.log_event(
                "generation_failed",
                {
                    "goal": goal,
                    "error": str(exc),
                },
            )
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

            self._session_active = True
            self._session_paused = False
            self._state.log_event(
                "session_start",
                {
                    "goal": None,
                    "curriculum_file": str(path),
                },
            )
            self._emit_progress("started")
            self._submit_next_lesson()

            try:
                self._state.save()
            except ValueError:
                logger.debug("No save path configured — skipping state persistence")
        except CurriculumParseError as exc:
            logger.error("Failed to load curriculum from %s: %s", path, exc)
            self._state.log_event("curriculum_load_error", {"path": str(path), "error": str(exc)})

    # Session lifecycle

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

        # Startup resolution: curriculum_file → load; else saved state →
        # resume; else poll for goal.
        if self._curriculum_file and self._curriculum_file.exists():
            try:
                doc = CurriculumDocument.from_file(self._curriculum_file)
                self._state.curriculum = Curriculum(doc)
            except CurriculumParseError:
                logger.error("Failed to load curriculum from %s", self._curriculum_file)
                return
        elif self._state.curriculum_file:
            saved_path = Path(self._state.curriculum_file)
            if saved_path.exists():
                try:
                    doc = CurriculumDocument.from_file(saved_path)
                    self._state.curriculum = Curriculum(doc)
                    self._curriculum_file = saved_path
                except CurriculumParseError:
                    logger.error("Failed to resume from %s", saved_path)
        elif not self._state.curriculum.lessons:
            self._session_active = False
            self._polling_for_goal = True
            logger.info("No curriculum resolved — polling for goal")
            self._state.log_event("polling_for_goal", {})
            self._emit_polling_status()
            return

        # If still empty after resolution, poll for goal instead.
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

        logger.info(
            "Session started — %d lessons, curriculum: %s",
            self._state.curriculum.total(),
            self._curriculum_file or "(none)",
        )
        self._emit_progress("started")
        self._submit_next_lesson()

    # File polling

    def _poll_curriculum_file(self) -> None:
        """Re-read the curriculum file from disk before each lesson.

        Picks up amendments and new lessons without restart.
        """
        if self._curriculum_file and self._curriculum_file.exists():
            try:
                doc = CurriculumDocument.from_file(self._curriculum_file)
                new_curriculum = Curriculum(doc)

                old_labels = set(self._state.curriculum.document.all_labels())
                new_labels = set(new_curriculum.document.all_labels())

                if new_labels != old_labels:
                    added = new_labels - old_labels
                    if added:
                        self._state.log_event(
                            "amendment_detected",
                            {
                                "new_labels": sorted(added),
                            },
                        )
                        self._emit_progress("amended")

                pos = self._state.curriculum.position
                new_curriculum.position = pos
                self._state.curriculum = new_curriculum

            except CurriculumParseError as exc:
                logger.warning("Failed to re-read curriculum file: %s", exc)

    # Curriculum-driven mode

    def _submit_next_lesson(self) -> None:
        """Submit the next lesson from the curriculum.

        Drains the Cogitator between lessons so late-arriving S2/S3 events
        from lesson N can't consume lesson N+1's reactive budget.
        """
        if self._drain_pending:
            logger.debug("Drain already pending — deferring lesson submit")
            return

        self._drain_pending = True
        self._bus.send(
            Message(
                role=TRAINEE_ROLE,
                action="drain",
                message=30.0,  # timeout in seconds
                sender=self._role,
            )
        )

    def _do_submit_lesson(self) -> None:
        """Actually compile and submit the next lesson."""
        self._poll_curriculum_file()

        lesson = self._state.curriculum.current()
        if lesson is None:
            logger.info("Curriculum complete — all lessons submitted")
            self._emit_progress("complete")
            self._end_session()
            return

        current_lesson = self._state.curriculum.current_lesson()
        if current_lesson:
            self._state.mark_lesson_submitted(current_lesson.label)
            logger.info(
                "Submitting lesson %s (%d/%d)",
                current_lesson.label,
                len(self._state.lesson_satisfied) + 1,
                self._state.curriculum.total(),
            )
            logger.debug("Lesson %s kscript: %s", current_lesson.label, lesson.strip())

        entries = compile_source(lesson, tokenizer=self._tokenizer, signifier=self._signifier)
        self._reactor.load_lesson(entries)

        logger.info(
            "Compiled %d entries for lesson %s",
            len(entries),
            current_lesson.label if current_lesson else "?",
        )

        for entry in entries:
            key = _entry_key(entry)
            self._state.mark_submitted(key)

        # Adapter compiles the source independently.
        self._bus.send(
            Message(
                role=TRAINEE_ROLE,
                action="submit",
                message=lesson,
                sender=self._role,
            )
        )

    # Progress events

    def _emit_progress(self, status: str, *, label_override: str | None = None) -> None:
        """Emit a progress event to the UI participant."""
        if label_override is not None:
            lesson_label = label_override
        else:
            current_lesson = self._state.curriculum.current_lesson()
            lesson_label = current_lesson.label if current_lesson else None
        total = self._state.curriculum.total()
        completed = len(self._state.lesson_satisfied)

        self._bus.send(
            Message(
                role=SUPERVISOR_ROLE,
                action="progress",
                message={
                    "lesson_label": lesson_label,
                    "lessons_total": total,
                    "lessons_completed": completed,
                    "status": status,
                },
                sender=self._role,
            )
        )

    def _emit_polling_status(self) -> None:
        """Signal to the UI that the Trainer is waiting for a goal input."""
        self._bus.send(
            Message(
                role=SUPERVISOR_ROLE,
                action="progress",
                message={
                    "lesson_label": None,
                    "lessons_total": self._state.curriculum.total(),
                    "lessons_completed": len(self._state.lesson_satisfied),
                    "status": "polling_for_goal",
                },
                sender=self._role,
            )
        )

    # Amendment

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
                    self._state.log_event(
                        "amendment_applied",
                        {
                            "action": action,
                        },
                    )
                    self._emit_progress("amended")
                    self._poll_curriculum_file()
                    if not self._session_paused and self._session_active:
                        self._submit_next_lesson()
            except (CurriculumParseError, ValueError) as exc:
                logger.error("Amendment failed: %s", exc)
        else:
            logger.warning("Cannot amend: no curriculum file")

    # Lesson completion

    def _check_lesson_complete(self) -> bool:
        """Return ``True`` if the current lesson is complete (every submitted
        entry satisfied). Uses ``satisfied`` vs ``submitted`` rather than
        reactor event counts, since cogitation can yield multiple events per
        entry."""
        if (
            len(self._state.satisfied) >= len(self._state.submitted)
            and len(self._state.submitted) > 0
        ):
            # Don't re-fire if already complete (late cogitation events
            # arrive after completion).
            current_lesson = self._state.curriculum.current_lesson()
            if current_lesson and self._state.is_lesson_satisfied(current_lesson.label):
                return False
            if not current_lesson:
                return False

            completed_label = current_lesson.label
            self._state.mark_lesson_satisfied(current_lesson.label)

            satisfied = len(self._state.satisfied)
            total_entries = len(self._state.submitted)
            logger.info(
                "Lesson %s complete — entries: %d/%d satisfied, %d/%d lessons done",
                completed_label or "?",
                satisfied,
                total_entries,
                len(self._state.lesson_satisfied),
                self._state.curriculum.total(),
            )

            self._state.log_event(
                "lesson_complete",
                {"position": self._state.curriculum.position},
            )

            self._emit_progress("lesson_complete", label_override=completed_label)

            if not self._session_paused:
                self._submit_next_lesson()
            return True
        return False

    # Session end

    def _end_session(self) -> None:
        """End the current session, persist state, process queued goals."""
        self._session_active = False
        self._session_paused = False
        self._polling_for_goal = False
        self._state.log_event("session_end", {})

        try:
            self._state.save()
        except ValueError:
            logger.debug("No save path configured — skipping state persistence")

        if self._pending_goals:
            next_goal = self._pending_goals.pop(0)
            self.start_session(goal=next_goal)

    def _restart_session(self) -> None:
        """Clear training state and restart from the first lesson."""
        was_active = self._session_active
        if not was_active:
            return

        self._session_active = False
        self._session_paused = False

        self._state.curriculum.position = 0
        self._state.submitted.clear()
        self._state.satisfied.clear()
        self._state.pending.clear()
        self._state.lesson_submitted.clear()
        self._state.lesson_satisfied.clear()

        self._reactor.load_lesson([])

        self._state.log_event("session_restart", {})
        self._emit_progress("restart")

        self._session_active = True
        self._session_paused = False
        self._state.log_event("session_start", {"goal": None})
        self._emit_progress("started")
        self._submit_next_lesson()


# Module-level helpers


def _entry_key(value: KValue) -> EntryKey:
    """Return a hashable identity key for a KValue (from its kline)."""
    return (value.kline.signature, tuple(value.kline.nodes))
