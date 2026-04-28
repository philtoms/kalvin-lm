"""Tests for Agent — openspec/agent.md conformance."""

import pytest
from kalvin.kline import KLine
from kalvin.agent import Agent
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.model import Model
from kalvin.signature import make_signature

MASK64 = 0xFFFF_FFFF_FFFF_FFFF


class TestAgentInit:
    def test_default_init(self):
        a = Agent()
        assert a.model is not None
        assert a.tokenizer is not None

    def test_custom_tokenizer(self):
        t = Mod32Tokenizer()
        a = Agent(tokenizer=t)
        assert a.tokenizer is t

    def test_custom_model(self):
        t = Mod32Tokenizer()
        m = Model(is_literal_fn=t.is_literal)
        a = Agent(tokenizer=t, model=m)
        assert a.model is m

    def test_frame_size_empty(self):
        a = Agent()
        assert a.frame_size() == 0


class TestAgentRationalise:
    def test_all_literal_s1(self):
        """All-literal kline → S1 (fast path)."""
        a = Agent()
        t = a.tokenizer
        lit_nodes = t.encode("ABC", pack=False)
        k = KLine(1, lit_nodes)
        result = a.rationalise(k)
        assert result is True

    def test_unsigned_s4(self):
        """Empty kline → S4."""
        a = Agent()
        k = KLine(0, [])
        result = a.rationalise(k)
        assert result is True

    def test_ground_check(self):
        """Already exists → ground event."""
        a = Agent()
        k = KLine(5, [1, 2])
        a.rationalise(k)
        # Second rationalise should return True (grounded)
        result = a.rationalise(KLine(5, [1, 2]))
        assert result is True

    def test_novel_kline(self):
        """Novel kline with no candidates → S4."""
        a = Agent()
        # Use a unique signature unlikely to overlap
        t = a.tokenizer
        packed = t.encode("XYZ", pack=True)[0]
        k = KLine(packed, [packed])
        result = a.rationalise(k)
        assert result is True

    def test_rationalise_adds_to_model(self):
        a = Agent()
        k = KLine(5, [1, 2])
        a.rationalise(k)
        assert a.model.find(5) is not None

    def test_rationalise_with_external_encode(self):
        """Caller encodes text, builds kline, rationalises."""
        a = Agent()
        t = a.tokenizer

        # Caller-side encoding workflow
        nodes = t.encode("hello", pack=True)
        sig = make_signature(nodes, t.is_literal)
        kline = KLine(sig, nodes, dbg_text="hello")

        result = a.rationalise(kline)
        assert result is True
        assert a.model.find(sig) is not None


class TestAgentEvents:
    def test_subscribe(self):
        a = Agent()
        events = []
        a.events.subscribe(lambda e: events.append(e))

        k = KLine(0, [])
        a.rationalise(k)
        assert len(events) >= 1

    def test_ground_event(self):
        a = Agent()
        events = []
        a.events.subscribe(lambda e: events.append(e))

        k = KLine(5, [1, 2])
        a.rationalise(k)
        a.rationalise(KLine(5, [1, 2]))
        # First: frame, second: ground
        kinds = [e.kind for e in events]
        assert "ground" in kinds

    def test_frame_event(self):
        a = Agent()
        events = []
        a.events.subscribe(lambda e: events.append(e))

        k = KLine(0, [])
        a.rationalise(k)
        assert any(e.kind == "frame" for e in events)


class TestAgentCogitate:
    def test_cogitate_join(self):
        a = Agent()
        a.cogitate_join(timeout=1.0)
        # Should not raise

    def test_rationalise_after_join(self):
        a = Agent()
        a.cogitate_join(timeout=1.0)
        k = KLine(5, [1, 2])
        result = a.rationalise(k)
        # Should still work without cogitation thread
        assert isinstance(result, bool)
