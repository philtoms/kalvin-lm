"""Tests for cascade control — reactor silent-drop past budget.

Spec: specs/reactive-delegation.md §Reactive-Round Budget (Default Mode).
"""

from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from training.harness.bus import MessageBus
from training.harness.constants import SUPERVISOR_ROLE
from training.harness.message import Message
from training.trainer.curriculum import Curriculum, CurriculumState
from training.trainer.reactor import Reactor

# ── Helpers ────────────────────────────────────────────────────────────


class BusCapture:
    """Captures messages sent via bus.send() for test assertions."""

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self.messages: list[Message] = []
        self._original_send = bus.send

    def install(self) -> None:
        capture = self

        def capturing_send(msg: Message) -> None:
            capture.messages.append(msg)

        self._bus.send = capturing_send  # type: ignore[assignment]

    def find_all(self, role: str, action: str) -> list[Message]:
        return [m for m in self.messages if m.role == role and m.action == action]


# ── Reactor silent-drop tests ─────────────────────────────────────────


class TestReactorSilentDrop:
    """CC-4, CC-5."""

    def test_first_budget_exhaustion_escalates(self):
        """CC-4: First budget exhaustion event escalates."""
        bus = MessageBus()
        capture = BusCapture(bus)
        capture.install()
        state = CurriculumState(Curriculum(["A", "B"]))

        reactor = Reactor(
            bus,
            state,
            role="trainer",
            max_reactive_rounds=3,
            cogitate_fn=lambda e: None,
        )
        reactor.load_lesson([])

        # Process events up to budget
        for i in range(3):
            event = RationaliseEvent(
                "frame",
                KLine(1, [2]),
                KLine(3, [4]),
                100,
            )
            reactor.process_s2_s3(event)

        # The third event should have triggered escalation
        escalation_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        budget_msgs = [m for m in escalation_msgs if m.message.get("reason") == "budget_exhaustion"]
        assert len(budget_msgs) >= 1

    def test_subsequent_events_silently_dropped(self):
        """CC-5: Subsequent events after budget exhaustion are silently dropped."""
        bus = MessageBus()
        capture = BusCapture(bus)
        capture.install()
        state = CurriculumState(Curriculum(["A", "B"]))

        call_count = 0

        def counting_cogitate(event):
            nonlocal call_count
            call_count += 1
            return None

        reactor = Reactor(
            bus,
            state,
            role="trainer",
            max_reactive_rounds=3,
            cogitate_fn=counting_cogitate,
        )
        reactor.load_lesson([])

        # Process 10 events — only first 3 should trigger cogitate
        for i in range(10):
            event = RationaliseEvent(
                "frame",
                KLine(1, [2]),
                KLine(3, [4]),
                100,
            )
            reactor.process_s2_s3(event)

        # Cogitate should have been called at most max_reactive_rounds times
        assert call_count <= 3

        # Only one budget_exhaustion escalation (not 7)
        escalation_msgs = capture.find_all(SUPERVISOR_ROLE, "notify")
        budget_msgs = [m for m in escalation_msgs if m.message.get("reason") == "budget_exhaustion"]
        assert len(budget_msgs) == 1
