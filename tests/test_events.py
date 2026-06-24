"""Tests for EventBus and RationaliseEvent."""

import threading

from kalvin.events import EventBus, RationaliseEvent
from kalvin.kline import KLine
from kalvin.kvalue import KValue


class TestRationaliseEvent:
    def test_attributes(self) -> None:
        k1 = KLine(5, [1, 2])
        k2 = KLine(10, [3])
        q = KValue(k1, 0x10)
        p = KValue(k2, 42)
        e = RationaliseEvent("frame", q, p)
        assert e.kind == "frame"
        assert e.query is q
        assert e.proposal is p
        # Significance lives on the KValues, not the event (KE-3).
        assert e.proposal.significance == 42
        assert e.query.significance == 0x10

    def test_repr(self) -> None:
        # __repr__ reports Kalvin's assessment via proposal.significance.
        e = RationaliseEvent("ground", KValue(KLine(0, []), 0), KValue(KLine(0, []), 0xFF))
        r = repr(e)
        assert "ground" in r
        assert "0xff" in r


class TestEventBus:
    def test_subscribe_and_publish(self) -> None:
        bus = EventBus()
        received = []
        bus.subscribe(lambda e: received.append(e))

        k = KLine(0, [])
        kv = KValue(k, 0)
        event = RationaliseEvent("frame", kv, kv)
        bus.publish(event)

        assert len(received) == 1
        assert received[0] is event

    def test_multiple_subscribers(self) -> None:
        bus = EventBus()
        a, b = [], []
        bus.subscribe(lambda e: a.append(e))
        bus.subscribe(lambda e: b.append(e))

        k = KLine(0, [])
        kv = KValue(k, 0)
        event = RationaliseEvent("frame", kv, kv)
        bus.publish(event)

        assert len(a) == 1
        assert len(b) == 1

    def test_no_subscribers(self) -> None:
        bus = EventBus()
        k = KLine(0, [])
        kv = KValue(k, 0)
        event = RationaliseEvent("frame", kv, kv)
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
            kv = KValue(k, i)
            results.append(RationaliseEvent("frame", kv, kv))

        threads = [threading.Thread(target=bus.publish, args=(e,)) for e in results]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 10
