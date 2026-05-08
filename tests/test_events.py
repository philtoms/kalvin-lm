"""Tests for EventBus and RationaliseEvent."""

import threading

from kalvin.events import EventBus, RationaliseEvent
from kalvin.kline import KLine


class TestRationaliseEvent:
    def test_attributes(self) -> None:
        k1 = KLine(5, [1, 2])
        k2 = KLine(10, [3])
        e = RationaliseEvent("frame", k1, k2, 42)
        assert e.kind == "frame"
        assert e.query is k1
        assert e.proposal is k2
        assert e.significance == 42

    def test_repr(self) -> None:
        e = RationaliseEvent("ground", KLine(0, []), KLine(0, []), 0xFF)
        r = repr(e)
        assert "ground" in r
        assert "0xff" in r


class TestEventBus:
    def test_subscribe_and_publish(self) -> None:
        bus = EventBus()
        received = []
        bus.subscribe(lambda e: received.append(e))

        k = KLine(0, [])
        event = RationaliseEvent("frame", k, k, 0)
        bus.publish(event)

        assert len(received) == 1
        assert received[0] is event

    def test_multiple_subscribers(self) -> None:
        bus = EventBus()
        a, b = [], []
        bus.subscribe(lambda e: a.append(e))
        bus.subscribe(lambda e: b.append(e))

        k = KLine(0, [])
        event = RationaliseEvent("frame", k, k, 0)
        bus.publish(event)

        assert len(a) == 1
        assert len(b) == 1

    def test_no_subscribers(self) -> None:
        bus = EventBus()
        k = KLine(0, [])
        event = RationaliseEvent("frame", k, k, 0)
        # Should not raise
        bus.publish(event)

    def test_thread_safety(self) -> None:
        bus = EventBus()
        received = []
        lock = threading.Lock()
        bus.subscribe(lambda e: (lock.acquire(), received.append(e), lock.release()))

        results = []
        for i in range(10):
            k = KLine(i, [])
            results.append(RationaliseEvent("frame", k, k, i))

        threads = [threading.Thread(target=bus.publish, args=(e,)) for e in results]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 10
