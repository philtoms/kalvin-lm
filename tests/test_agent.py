"""Tests for Agent — openspec/agent.md conformance."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from kalvin.kline import KLine
from kalvin.agent import Agent, Cogitator, WorkItem
from kalvin.model import QueryCandidate
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

    def test_cogitator_accessible(self):
        a = Agent()
        assert isinstance(a.cogitator, Cogitator)


# ── Routing Tests ─────────────────────────────────────────────────────

class TestRoute:
    """Agent._route: fast node-membership classification. No model call."""

    def test_all_nodes_match_s1(self):
        q = KLine(5, [10, 20])
        c = KLine(99, [10, 20, 30])
        assert Agent._route(q, c) == "S1"

    def test_some_nodes_match_s2(self):
        q = KLine(5, [10, 20, 99])
        c = KLine(99, [10, 20, 30])
        assert Agent._route(q, c) == "S2"

    def test_no_nodes_match_s3(self):
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        assert Agent._route(q, c) == "S3"

    def test_single_node_match_s1(self):
        q = KLine(5, [10])
        c = KLine(99, [10, 20])
        assert Agent._route(q, c) == "S1"

    def test_empty_query_s4(self):
        q = KLine(0, [])
        c = KLine(10, [1])
        assert Agent._route(q, c) == "S4"

    def test_routing_independent_of_signature(self):
        """Routing only cares about candidate's node sequence."""
        q = KLine(5, [42])
        c = KLine(999, [42, 100])
        assert Agent._route(q, c) == "S1"

    def test_duplicate_nodes_in_query(self):
        """Duplicate query nodes are counted per-occurrence for membership."""
        q = KLine(5, [10, 10])
        c = KLine(99, [10, 20])
        assert Agent._route(q, c) == "S1"


# ── Rationalisation Tests ─────────────────────────────────────────────

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
        result = a.rationalise(KLine(5, [1, 2]))
        assert result is True

    def test_novel_kline(self):
        """Novel kline with no candidates → S4."""
        a = Agent()
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
        nodes = t.encode("hello", pack=True)
        sig = make_signature(nodes, t.is_literal)
        kline = KLine(sig, nodes, dbg_text="hello")
        result = a.rationalise(kline)
        assert result is True
        assert a.model.find(sig) is not None

    def test_s2_kline_returns_false(self):
        """Kline that routes S2 against all candidates → returns False."""
        a = Agent()
        # Add a candidate that partially overlaps
        candidate = KLine(5, [10, 30])
        a.rationalise(candidate)
        # Query overlaps on [10] but not [20] → S2
        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20], a.tokenizer.is_literal)
        result = a.rationalise(q)
        assert result is False

    def test_s3_kline_returns_false(self):
        """Kline that routes S3 against all candidates → returns False."""
        a = Agent()
        candidate = KLine(5, [100, 200])
        a.rationalise(candidate)
        q = KLine(0, [1, 2])
        q.signature = make_signature([1, 2], a.tokenizer.is_literal)
        result = a.rationalise(q)
        assert result is False


# ── Short-Circuit Tests ───────────────────────────────────────────────

class TestShortCircuit:
    """S1 short-circuits — no model.expand() called."""

    def test_s1_skips_remaining_candidates(self):
        """First candidate is S1 → no further candidates processed."""
        a = Agent()
        # Add two candidates to the model
        c1 = KLine(5, [10, 20])       # S1 match for query
        c2 = KLine(6, [10, 20, 30])   # Also S1 but shouldn't be reached
        a.rationalise(c1)
        a.rationalise(c2)

        # Query that matches c1 fully
        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20], a.tokenizer.is_literal)

        # Patch model.expand to track calls
        with patch.object(a.model, 'expand', side_effect=AssertionError("expand should not be called for S1")):
            result = a.rationalise(q)

        assert result is True

    def test_s1_after_s2_still_short_circuits(self):
        """If second candidate is S1, no expand called."""
        a = Agent()
        # c1 will route S2, c2 will route S1
        c1 = KLine(5, [10, 30])
        c2 = KLine(6, [10, 20])
        a.rationalise(c1)
        a.rationalise(c2)

        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20], a.tokenizer.is_literal)

        result = a.rationalise(q)
        assert result is True

    def test_no_candidates_no_expand(self):
        """No candidates → S4 directly, no expand call."""
        a = Agent()
        q = KLine(0, [999])
        q.signature = make_signature([999], a.tokenizer.is_literal)

        with patch.object(a.model, 'expand', side_effect=AssertionError("expand should not be called for S4")):
            result = a.rationalise(q)

        assert result is True


# ── WorkItem Tests ────────────────────────────────────────────────────

class TestWorkItem:
    def test_work_item_fields(self):
        q = KLine(5, [1, 2])
        c = KLine(10, [3, 4])
        item = WorkItem(q, c, "S2")
        assert item.query is q
        assert item.candidate is c
        assert item.level == "S2"

    def test_work_item_equality(self):
        q = KLine(5, [1])
        c = KLine(10, [3])
        assert WorkItem(q, c, "S2") == WorkItem(q, c, "S2")
        assert WorkItem(q, c, "S2") != WorkItem(q, c, "S3")


# ── Event Tests ───────────────────────────────────────────────────────

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
        kinds = [e.kind for e in events]
        assert "ground" in kinds

    def test_frame_event(self):
        a = Agent()
        events = []
        a.events.subscribe(lambda e: events.append(e))
        k = KLine(0, [])
        a.rationalise(k)
        assert any(e.kind == "frame" for e in events)


# ── Cogitator Tests ───────────────────────────────────────────────────

class TestCogitator:
    def test_cogitate_join(self):
        a = Agent()
        a.cogitate_join(timeout=1.0)
        # Should not raise

    def test_rationalise_after_join(self):
        a = Agent()
        a.cogitate_join(timeout=1.0)
        k = KLine(5, [1, 2])
        result = a.rationalise(k)
        assert isinstance(result, bool)

    def test_s2_submits_work_item(self):
        """S2 kline submits a work item to the cogitator."""
        a = Agent()
        candidate = KLine(5, [10, 30])
        a.rationalise(candidate)

        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20], a.tokenizer.is_literal)

        # Capture submitted work items
        submitted = []
        original_submit = a._cogitator.submit

        def capture_submit(item):
            submitted.append(item)
            original_submit(item)

        a._cogitator.submit = capture_submit
        result = a.rationalise(q)

        assert result is False
        assert len(submitted) == 1
        assert submitted[0].level == "S2"
        assert submitted[0].query is q
        assert submitted[0].candidate is candidate


# ── Serialization Tests ───────────────────────────────────────────────

class TestAgentSerialization:
    def _make_agent_with_klines(self) -> Agent:
        a = Agent()
        a.rationalise(KLine(5, [1, 2]))
        a.rationalise(KLine(10, [3, 4]))
        a.rationalise(KLine(0, []))
        return a

    def test_to_bytes_roundtrip(self):
        a = self._make_agent_with_klines()
        data = a.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0
        loaded = Agent.from_bytes(data)
        assert len(loaded.model) == len(a.model)

    def test_to_dict_roundtrip(self):
        a = self._make_agent_with_klines()
        d = a.to_dict()
        assert isinstance(d, dict)
        assert "klines" in d
        assert "activity" in d
        loaded = Agent.from_dict(d)
        assert len(loaded.model) == len(a.model)

    def test_to_dict_structure(self):
        a = Agent()
        a.rationalise(KLine(5, [1, 2]))
        d = a.to_dict()
        assert len(d["klines"]) == 1
        assert d["klines"][0]["signature"] == 5
        assert d["klines"][0]["nodes"] == [1, 2]

    def test_save_and_load_json(self):
        a = self._make_agent_with_klines()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            a.save(path)
            loaded = Agent.load(path)
            assert len(loaded.model) == len(a.model)
        finally:
            path.unlink(missing_ok=True)

    def test_save_and_load_bin(self):
        a = self._make_agent_with_klines()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = Path(f.name)
        try:
            a.save(path, format="bin")
            loaded = Agent.load(path, format="bin")
            assert len(loaded.model) == len(a.model)
        finally:
            path.unlink(missing_ok=True)

    def test_save_auto_detect_json(self):
        a = Agent()
        a.rationalise(KLine(5, [1]))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            a.save(path)
            content = path.read_text()
            data = json.loads(content)
            assert isinstance(data, dict)
        finally:
            path.unlink(missing_ok=True)

    def test_empty_agent_serialization(self):
        a = Agent()
        data = a.to_bytes()
        loaded = Agent.from_bytes(data)
        assert len(loaded.model) == 0

    def test_empty_agent_dict(self):
        a = Agent()
        d = a.to_dict()
        loaded = Agent.from_dict(d)
        assert len(loaded.model) == 0
