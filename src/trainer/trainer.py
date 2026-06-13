"""Trainer participant — embedded harness component that drives the training loop.

The Trainer submits curriculum lessons to the KAgent and manages session
lifecycle (start/stop/pause, curriculum loading, file polling, progress
emission, message routing). Reactive handling of S2/S3 events — auto-
countersign matching, scaffolding via ``cogitate_fn``, budget tracking,
and escalation — is delegated to the :class:`~trainer.reactor.Reactor`.

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
from harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE
from harness.message import Message
from kalvin.events import RationaliseEvent
from kalvin.expand import D_MAX
from kalvin.kline import KLine, kline_display
from kalvin.mod_tokenizer import Mod32Tokenizer
from ks.compiler import compile_source
from trainer.curriculum import Curriculum, CurriculumState, EntryKey
from trainer.curriculum_document import (
    CurriculumDocument,
    CurriculumParseError,
    Lesson,
)
from trainer.reactor import Reactor

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
    role:
        Bus role for this participant (default ``"trainer"``).
    max_reactive_rounds:
        Maximum reactive scaffolding rounds before budget-exhaustion escalation.
    cogitate_fn:
        Optional cogitation function. Signature:
        ``(RationaliseEvent) -> tuple[str, float] | None``.
        Returns ``(kscript_source, confidence)`` or ``None`` if no
        scaffolding can be generated. When ``llm_client`` is provided
        but ``cogitate_fn`` is not, a :class:`~trainer.cogitation.Cogitator`
        is automatically constructed and its ``cogitate()`` method is
        adapted into the ``cogitate_fn`` callable the
        :class:`~trainer.reactor.Reactor` expects.
    llm_client:
        Optional LLM client for curriculum generation and reactive
        scaffolding. Must satisfy the :class:`~trainer.cogitation.LLMClient`
        protocol. When provided without an explicit ``cogitate_fn``,
        a :class:`~trainer.cogitation.Cogitator` is auto-wired for
        reactive scaffolding on S2/S3 events. Also enables goal-based
        curriculum generation via the CurriculumGenerator.
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
        max_reactive_rounds: int = 5,
        cogitate_fn: Callable[[RationaliseEvent], tuple[str, float] | None] | None = None,
        llm_client: Any | None = None,
        save_path: str | Path | None = None,
        curriculum_file: str | Path | None = None,
        curricula_dir: str | Path | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self._role = role
        self._bus = bus
        self._tokenizer = tokenizer
        self._state = CurriculumState(
            curriculum,
            save_path=save_path,
            curriculum_file=str(curriculum_file) if curriculum_file else None,
        )
        self._llm_client = llm_client
        self._curriculum_file = Path(curriculum_file) if curriculum_file else None
        self._curricula_dir = Path(curricula_dir) if curricula_dir else None

        # Auto-wire Cogitator when llm_client is provided without an
        # explicit cogitate_fn.  The adapter closes over a local
        # Cogitator instance (not ``self``) so that cogitate_fn
        # remains a plain callable.
        if llm_client is not None and cogitate_fn is None:
            from kalvin.misfit import classify_misfit
            from kalvin.signature import make_signature
            from trainer.cogitation import CogitationRequest, Cogitator, MisfitInfo

            _cogitator = Cogitator(client=llm_client)
            _display_tok = Mod32Tokenizer()

            def _cogitate_adapter(
                event: RationaliseEvent,
            ) -> tuple[str, float] | None:
                # Compute misfit diagnosis on the ORIGINAL candidate
                # (the one with the gap/excess), not the expansion proposal.
                # event.candidate carries the original misfit candidate
                # for S2/S3 expansion events; fall back to event.proposal
                # for legacy events without the candidate field.
                target = event.candidate if event.candidate is not None else event.proposal
                target_underfit, target_overfit = classify_misfit(target)
                target_nodes_sig = make_signature(target.nodes)
                underfit_gap = target.signature & ~target_nodes_sig
                overfit_mask = target_nodes_sig & ~target.signature

                # Display klines as human-readable KScript for the LLM
                try:
                    query_src = kline_display(event.query, _display_tok)
                except Exception:
                    query_src = repr(event.query)
                try:
                    proposal_src = (
                        kline_display(event.proposal, _display_tok) if event.proposal else "None"
                    )
                except Exception:
                    proposal_src = repr(event.proposal) if event.proposal else "None"

                logger.info(
                    "Cogitate adapter: event query=%r (%s), candidate=%r, proposal=%r (%s)",
                    event.query,
                    query_src,
                    target,
                    event.proposal,
                    proposal_src,
                )
                logger.info(
                    "Cogitate misfit: underfit=%s, overfit=%s, gap=%#x, mask=%#x",
                    target_underfit,
                    target_overfit,
                    underfit_gap,
                    overfit_mask,
                )

                misfit_info = MisfitInfo(
                    underfit=target_underfit,
                    overfit=target_overfit,
                    underfit_gap=underfit_gap,
                    overfit_mask=overfit_mask,
                    expectation_summary=query_src,
                    proposal_summary=proposal_src,
                )

                request = CogitationRequest(
                    events=[event],
                    misfits=[misfit_info],
                    curriculum_context="",
                    conversation_history=[],
                    round_number=1,
                    max_rounds=3,
                )

                result = _cogitator.cogitate(request)
                logger.info(
                    "Cogitate result: scaffolding=%s, confidence=%.2f, reasoning=%s",
                    "None" if result.scaffolding is None else result.scaffolding[:80],
                    result.confidence,
                    result.reasoning[:80] if result.reasoning else "",
                )
                if result.scaffolding is not None:
                    return (result.scaffolding, result.confidence)
                return None

            cogitate_fn = _cogitate_adapter

        # Reactor handles S2/S3 event processing
        self._reactor = Reactor(
            bus,
            self._state,
            role=role,
            max_reactive_rounds=max_reactive_rounds,
            cogitate_fn=cogitate_fn,
        )

        # Session model fields
        self._session_active: bool = False
        self._session_paused: bool = False
        self._pending_goals: list[str] = []
        self._conversation_history: list[str] = []
        self._polling_for_goal: bool = False
        self._drain_pending: bool = False

        # Register on the bus
        bus.subscribe(self._role, self.on_message)

        if self._curriculum_file is not None and self._state.curriculum.total() > 0:
            # Curriculum loaded but not started — tell the UI we're ready
            self._emit_progress("ready")
        elif self._curriculum_file is None and self._state.curriculum.total() == 0:
            # No curriculum at all — enter goal-polling mode
            self._polling_for_goal = True
            self._state.log_event("polling_for_goal", {})
            self._emit_progress("polling_for_goal")

    # ── Participant protocol ──────────────────────────────────────────

    @property
    def role(self) -> str:
        """Bus role for this participant."""
        return self._role

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
        elif action == "drained":
            self._handle_drained(msg)
        elif action == "input":
            self._handle_input(msg)
        else:
            logger.warning("Unknown action %r from %s", action, msg.sender)

    # ── KAgent event handling ─────────────────────────────────────────

    def _handle_kagent_event(self, msg: Message) -> None:
        """Process a KAgent ground or frame event."""
        event: RationaliseEvent = msg.message

        # Display for log readability
        _log_tok = Mod32Tokenizer()
        try:
            query_src = kline_display(event.query, _log_tok)
        except Exception:
            query_src = repr(event.query)
        try:
            proposal_src = kline_display(event.proposal, _log_tok) if event.proposal else None
        except Exception:
            proposal_src = repr(event.proposal) if event.proposal else None

        if event.significance:
            distance = (~event.significance) & D_MAX
            sig_norm = distance
        else:
            sig_norm = 0

        if self._is_s1(event):
            logger.info(
                "%s %s → S1 (fast path)%s",
                event.kind.upper(),
                query_src,
                f" ← {proposal_src}" if proposal_src else "",
            )
        else:
            logger.info(
                "%s %s → %d%s",
                event.kind.upper(),
                query_src,
                sig_norm,
                f" | proposal: {proposal_src}" if proposal_src else "",
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
            # S1 fast path: auto-satisfy by query match
            key = _entry_key(event.query)
            self._state.mark_satisfied(key)
        else:
            # S2/S3 slow path: skip if entry already satisfied
            # (e.g. S1 cogitation event arrived before this S3 expansion)
            key = _entry_key(event.query)
            if self._state.is_satisfied(key):
                logger.debug("S2/S3 event for already-satisfied entry — skipping")
            else:
                auto_matched = self._reactor.process_s2_s3(event)

                # Ratify request for S2/S3 proposals (HRNS-33)
                # Only sent when auto-countersign did NOT handle it
                if not auto_matched:
                    self._bus.send(
                        Message(
                            role=SUPERVISOR_ROLE,
                            action="ratify_request",
                            message={
                                "proposal": event.proposal,
                                "query": event.query,
                                "significance": event.significance,
                            },
                            sender=self._role,
                        )
                    )

        self._check_lesson_complete()

    def _handle_kagent_error(self, msg: Message) -> None:
        """Log KAgent error and abandon the current lesson.

        A KAgent error means the lesson source could not be processed at
        all (e.g. ParseError). Mark every submitted-but-unsatisfied entry
        as satisfied so the lesson completes and training advances to the
        next one — errors must not stall the curriculum.
        """
        self._state.log_event("kagent_error", {"message": str(msg.message)})
        # An error abandons the lesson: satisfy all pending entries so the
        # satisfaction-vs-submitted gate in _check_lesson_complete can fire.
        unsatisfied = self._state.submitted - self._state.satisfied
        for key in unsatisfied:
            self._state.mark_satisfied(key)
        self._check_lesson_complete()

    # ── Cogitator drain ─────────────────────────────────────────────────

    def _handle_drained(self, msg: Message) -> None:
        """Handle the drained response from the KAgent adapter.

        When a drain was pending, this means all cogitation work items
        from the previous lesson have been processed. Now we can safely
        submit the next lesson.
        """
        if not self._drain_pending:
            logger.debug("Ignoring unexpected drained event")
            return
        self._drain_pending = False
        logger.info("Cogitator drained — submitting next lesson")
        self._do_submit_lesson()

    # ── Input handling (from Slack / supervisor) ───────────────────────────

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

            # Start session directly (curriculum is now loaded)
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

        logger.info(
            "Session started — %d lessons, curriculum: %s",
            self._state.curriculum.total(),
            self._curriculum_file or "(none)",
        )
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
                        self._state.log_event(
                            "amendment_detected",
                            {
                                "new_labels": sorted(added),
                            },
                        )
                        self._emit_progress("amended")

                # Update curriculum, preserving position
                pos = self._state.curriculum.position
                new_curriculum.position = pos
                self._state.curriculum = new_curriculum

            except CurriculumParseError as exc:
                logger.warning("Failed to re-read curriculum file: %s", exc)

    # ── Curriculum-driven mode ────────────────────────────────────────

    def _submit_next_lesson(self) -> None:
        """Submit the next lesson from the curriculum to the KAgent.

        Between lessons, drains the Cogitator to ensure all work items
        from the previous lesson have been processed. This prevents
        cross-lesson cogitation spillover where late-arriving S2/S3
        events from lesson N consume lesson N+1's reactive budget.
        """
        # Drain cogitator between lessons to prevent spillover.
        # On the first lesson there's nothing to drain, but the no-op
        # cost is negligible so we don't special-case it.
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
        # File polling: re-read before each lesson
        self._poll_curriculum_file()

        lesson = self._state.curriculum.current()
        if lesson is None:
            # Curriculum complete — end session
            logger.info("Curriculum complete — all lessons submitted")
            self._emit_progress("complete")
            self._end_session()
            return

        # Track lesson label
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

        # Compile locally to obtain KLine objects for structural matching
        entries = compile_source(lesson, tokenizer=self._tokenizer)
        self._reactor.load_lesson(entries)

        logger.info(
            "Compiled %d entries for lesson %s",
            len(entries),
            current_lesson.label if current_lesson else "?",
        )

        # Mark each entry as submitted
        for entry in entries:
            key = _entry_key(entry)
            self._state.mark_submitted(key)

        # Send raw KScript source — the adapter compiles independently
        self._bus.send(
            Message(
                role=TRAINEE_ROLE,
                action="submit",
                message=lesson,
                sender=self._role,
            )
        )

    # ── Progress events ───────────────────────────────────────────────

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
        """Emit a polling status event to the UI participant.

        Signals that the Trainer is waiting for a goal input.
        """
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
                    self._state.log_event(
                        "amendment_applied",
                        {
                            "action": action,
                        },
                    )
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

        A lesson is complete when every submitted entry has been marked
        satisfied. Uses ``CurriculumState.satisfied`` vs ``submitted``
        rather than reactor event counts, because cogitation can produce
        multiple events per entry (initial + expansions).
        """
        if (
            len(self._state.satisfied) >= len(self._state.submitted)
            and len(self._state.submitted) > 0
        ):
            # Guard: skip if lesson already completed (cogitation events
            # arrive after lesson completion fires — don't re-fire)
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

    def _restart_session(self) -> None:
        """Clear training state and restart the session from the beginning.

        Resets curriculum position, all tracking sets, and the reactor,
        then starts a fresh session with the current curriculum.
        """

        # End current session (without processing queued goals)
        self._session_active = False
        self._session_paused = False

        # Reset curriculum position and tracking sets
        self._state.curriculum.position = 0
        self._state.submitted.clear()
        self._state.satisfied.clear()
        self._state.pending.clear()
        self._state.lesson_submitted.clear()
        self._state.lesson_satisfied.clear()

        # Reset reactor
        self._reactor.load_lesson([])

        self._state.log_event("session_restart", {})
        self._emit_progress("restart")

        # Start fresh session
        self._session_active = True
        self._session_paused = False
        self._state.log_event("session_start", {"goal": None})
        self._emit_progress("started")
        self._submit_next_lesson()


# ── Module-level helpers ──────────────────────────────────────────────


def _entry_key(kline: KLine) -> EntryKey:
    """Return a hashable identity key for a KLine."""
    return (kline.signature, tuple(kline.nodes))
