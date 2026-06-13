"""Tests for Cogitator inter-lesson drain — spec specs/cogitator-drain.md.

Validates that the cogitator drain mechanism prevents cross-lesson
spillover of S2/S3 events.
"""

from __future__ import annotations

import threading
import time

from kalvin.agent import Cogitator, KAgent, WorkItem
from kalvin.events import EventBus
from kalvin.kline import KLine
from kalvin.model import Model
from kalvin.signature import make_signature

# ── Helpers ──────────────────────────────────────────────────────────


class _StubHandler:
    """Minimal CogitationHandler that records calls."""

    def __init__(self):
        self.s1_calls = []
        self.expansion_calls = []

    def on_s1(self, query, candidate):
        self.s1_calls.append((query, candidate))

    def on_expansion(self, query, proposal, significance, original_candidate=None):
        self.expansion_calls.append((query, proposal, significance))


def _make_cogitator(model=None, adapter=None, handler=None):
    """Create a Cogitator with sensible defaults."""
    model = model or Model()
    adapter = adapter or EventBus()
    handler = handler or _StubHandler()
    return Cogitator(model, adapter, handler)


# ── DRN-3: Empty-backlog drain completes fast ────────────────────────


class TestDrainEmptyBacklog:
    def test_empty_backlog_drain_returns_true(self):
        """DRN-3: drain() on empty backlog returns True immediately."""
        cog = _make_cogitator()
        try:
            start = time.monotonic()
            result = cog.drain(timeout=2.0)
            elapsed = time.monotonic() - start
            assert result is True
            assert elapsed < 0.5  # Should be nearly instant
        finally:
            cog.join(timeout=2.0)

    def test_empty_backlog_drain_under_10ms(self):
        """DRN-3: Empty-backlog drain completes in <10ms."""
        cog = _make_cogitator()
        try:
            start = time.monotonic()
            cog.drain(timeout=2.0)
            elapsed = time.monotonic() - start
            assert elapsed < 0.01  # <10ms
        finally:
            cog.join(timeout=2.0)


# ── DRN-4: Drain timeout ────────────────────────────────────────────


class TestDrainTimeout:
    def test_drain_timeout_returns_false(self):
        """DRN-4: drain() returns False on timeout when work item is slow."""
        model = Model()
        adapter = EventBus()
        handler = _StubHandler()

        # Create a work item that will block
        blocker_event = threading.Event()
        original_run = Cogitator._run_work_item

        def slow_run(self, item):
            blocker_event.wait(timeout=5.0)

        Cogitator._run_work_item = slow_run
        try:
            cog = Cogitator(model, adapter, handler)
            # Submit a work item
            item = WorkItem(
                query=KLine(0x1, []),
                candidate=KLine(0x2, []),
                level="S3",
            )
            cog.submit(item)
            # Small delay to ensure item is picked up
            time.sleep(0.3)

            result = cog.drain(timeout=0.5)
            assert result is False
        finally:
            blocker_event.set()
            Cogitator._run_work_item = original_run
            cog.join(timeout=2.0)

    def test_drain_timeout_does_not_stop_thread(self):
        """DRN-4: timed-out drain does not stop the cogitator thread."""
        cog = _make_cogitator()
        try:
            cog.drain(timeout=0.1)
            # Thread should still be alive
            assert cog._thread.is_alive()
        finally:
            cog.join(timeout=2.0)


# ── DRN-5: Processing flag ──────────────────────────────────────────


class TestProcessingFlag:
    def test_processing_flag_guards_drain(self):
        """DRN-5: drain() waits for processing flag to clear."""
        model = Model()
        adapter = EventBus()
        handler = _StubHandler()

        processed = threading.Event()
        original_run = Cogitator._run_work_item

        def blocking_run(self, item):
            time.sleep(0.3)
            processed.set()

        Cogitator._run_work_item = blocking_run
        try:
            cog = Cogitator(model, adapter, handler)
            item = WorkItem(
                query=KLine(0x1, []),
                candidate=KLine(0x2, []),
                level="S3",
            )
            cog.submit(item)
            # Drain should wait for the work item to finish
            result = cog.drain(timeout=5.0)
            assert result is True
            assert processed.is_set()
        finally:
            Cogitator._run_work_item = original_run
            cog.join(timeout=2.0)

    def test_processing_flag_not_set_when_idle(self):
        """DRN-5: _processing is False when cogitator is idle."""
        cog = _make_cogitator()
        try:
            assert cog._processing is False
        finally:
            cog.join(timeout=2.0)


# ── DRN-6: No cross-lesson spillover ────────────────────────────────


class TestNoCrossLessonSpillover:
    def test_drain_between_lessons_prevents_spillover(self):
        """DRN-6: Lesson-N cogitation drains fully before lesson N+1 begins.

        Submits real S2 work items in "lesson 1", drains the cogitator, then
        verifies the backlog is empty and the cogitator remains healthy for a
        subsequent lesson whose signature shares zero bits with lesson 1.
        """
        events: list = []
        bus = EventBus()
        agent = KAgent(adapter=bus)
        bus.subscribe(lambda e: events.append(e))

        try:
            # ── Lesson 1: add a candidate, rationalise a query that routes S2 ──
            # Candidate signature 5 = 0b00101.
            agent.model.add_ltm(KLine(5, [10, 30]))

            # Query signature 30 = 10 | 20 = 0b11110 — shares bit 2 with the
            # candidate, so Model.where() finds it and routing classifies S2
            # (node 10 overlaps, node 20 doesn't).
            q = KLine(0, [10, 20])
            q.signature = make_signature([10, 20])
            agent.rationalise(q)

            # The S2 candidate was submitted to the cogitator (not an
            # empty-backlog no-op like the old stub).
            assert len(agent.cogitator._backlog) > 0

            result = agent.cogitate_drain(timeout=5.0)
            assert result is True

            # Drain emptied the backlog and cleared the processing flag —
            # no lesson-N work remains to spill into lesson N+1.
            assert len(agent.cogitator._backlog) == 0
            assert agent.cogitator._processing is False

            # Real cogitation happened (events were captured), proving the
            # drain waited for actual work to complete.
            assert len(events) > 0

            # ── Lesson 2: verify cogitator health post-drain ──
            # Signature 64 = 0b1000000 shares zero bits with lesson-1
            # signatures (5 | 30 = 31 = 0b11111), so it routes S4 (no
            # candidates) — a clean frame event with no S2/S3 cogitation.
            events.clear()
            agent.rationalise(KLine(64, [64]))

            result = agent.cogitate_drain(timeout=5.0)
            assert result is True
            assert len(agent.cogitator._backlog) == 0
            assert len(events) > 0  # S4 frame event emitted
        finally:
            agent.cogitate_join(timeout=2.0)


# ── DRN-6 (unit): drain empties the backlog ─────────────────────────


class TestDrainEmptiesBacklog:
    def test_drain_empties_backlog_after_work(self):
        """DRN-6: drain() empties the backlog after processing real work.

        Isolates the core guarantee — a submitted work item is processed and
        the backlog is empty after drain — at the Cogitator level.
        """
        cog = _make_cogitator()
        try:
            cog.submit(WorkItem(query=KLine(0x1, []), candidate=KLine(0x2, []), level="S3"))

            result = cog.drain(timeout=5.0)
            assert result is True
            assert len(cog._backlog) == 0
            assert cog._processing is False
            # Drain does not stop the thread — ready for the next lesson.
            assert cog._thread.is_alive()
        finally:
            cog.join(timeout=2.0)


# ── KAgent.cogitate_drain ────────────────────────────────────────────


class TestKAgentDrain:
    def test_cogitate_drain_on_fresh_agent(self):
        """cogitate_drain returns True on fresh agent (no work items)."""
        bus = EventBus()
        agent = KAgent(adapter=bus)
        try:
            result = agent.cogitate_drain(timeout=2.0)
            assert result is True
        finally:
            agent.cogitate_join(timeout=2.0)

    def test_cogitate_drain_after_rationalise(self):
        """cogitate_drain returns True after all rationalise work completes."""
        bus = EventBus()
        agent = KAgent(adapter=bus)

        # Add identities
        agent.rationalise(KLine(0x2, []))  # A
        agent.rationalise(KLine(0x4, []))  # B

        # Submit an entry that may trigger cogitation
        agent.rationalise(KLine(0x6, [0x2, 0x4]))  # AB -> A, B

        try:
            result = agent.cogitate_drain(timeout=5.0)
            assert result is True
        finally:
            agent.cogitate_join(timeout=2.0)
