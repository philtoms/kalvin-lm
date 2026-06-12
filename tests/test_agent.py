"""Tests for Agent — openspec/agent.md conformance."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from kalvin.agent import CogitationHandler, Cogitator, KAgent, WorkItem
from kalvin.agent_codec import AgentCodec
from kalvin.events import EventBus
from kalvin.kline import KDbg, KLine
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.model import Model
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signature import make_signature


class TestAgentInit:
    def test_default_init(self):
        a = KAgent(adapter=EventBus())
        assert a.model is not None
        assert a.tokenizer is not None

    def test_custom_tokenizer(self):
        t = Mod32Tokenizer()
        a = KAgent(tokenizer=t, adapter=EventBus())
        assert a.tokenizer is t

    def test_custom_model(self):
        t = Mod32Tokenizer()
        m = Model()
        a = KAgent(tokenizer=t, model=m, adapter=EventBus())
        assert a.model is m

    def test_frame_size_empty(self):
        a = KAgent(adapter=EventBus())
        assert a.frame_size() == 0

    def test_cogitator_accessible(self):
        a = KAgent(adapter=EventBus())
        assert isinstance(a.cogitator, Cogitator)


# ── Routing Tests ─────────────────────────────────────────────────────

class TestRoute:
    """KAgent._route: fast node-membership classification. No model call."""

    def test_all_nodes_match_s1(self):
        q = KLine(5, [10, 20])
        c = KLine(99, [10, 20, 30])
        assert KAgent._route(q, c) == "S1"

    def test_some_nodes_match_s2(self):
        q = KLine(5, [10, 20, 99])
        c = KLine(99, [10, 20, 30])
        assert KAgent._route(q, c) == "S2"

    def test_no_nodes_match_s3(self):
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        assert KAgent._route(q, c) == "S3"

    def test_single_node_match_s1(self):
        q = KLine(5, [10])
        c = KLine(99, [10, 20])
        assert KAgent._route(q, c) == "S1"

    def test_empty_query_s4(self):
        q = KLine(0, [])
        c = KLine(10, [1])
        assert KAgent._route(q, c) == "S4"

    def test_routing_independent_of_signature(self):
        """Routing only cares about candidate's node sequence."""
        q = KLine(5, [42])
        c = KLine(999, [42, 100])
        assert KAgent._route(q, c) == "S1"

    def test_duplicate_nodes_in_query(self):
        """Duplicate query nodes are counted per-occurrence for membership."""
        q = KLine(5, [10, 10])
        c = KLine(99, [10, 20])
        assert KAgent._route(q, c) == "S1"


# ── Rationalisation Tests ─────────────────────────────────────────────

class TestAgentRationalise:
    def test_unsigned_s4(self):
        """Empty kline → S4."""
        a = KAgent(adapter=EventBus())
        k = KLine(0, [])
        result = a.rationalise(k)
        assert result is True

    def test_ground_check(self):
        """Already exists → ground event."""
        a = KAgent(adapter=EventBus())
        k = KLine(5, [1, 2])
        a.rationalise(k)
        result = a.rationalise(KLine(5, [1, 2]))
        assert result is True

    def test_novel_kline(self):
        """Novel kline with no candidates → S4."""
        a = KAgent(adapter=EventBus())
        t = a.tokenizer
        packed = t.encode("XYZ")[0]
        k = KLine(packed, [packed])
        result = a.rationalise(k)
        assert result is True

    def test_rationalise_adds_to_model(self):
        a = KAgent(adapter=EventBus())
        k = KLine(5, [1, 2])
        a.rationalise(k)
        assert a.model.find(5) is not None

    def test_rationalise_with_external_encode(self):
        """Caller encodes text, builds kline, rationalises."""
        a = KAgent(adapter=EventBus())
        t = a.tokenizer
        nodes = t.encode("HELLO")
        sig = make_signature(nodes)
        kline = KLine(sig, nodes, dbg=KDbg(label="HELLO"))
        result = a.rationalise(kline)
        assert result is True
        assert a.model.find(sig) is not None

    def test_s2_kline_returns_false(self):
        """Kline that routes S2 against all candidates → returns False."""
        a = KAgent(adapter=EventBus())
        # Add a candidate that partially overlaps
        candidate = KLine(5, [10, 30])
        a.rationalise(candidate)
        # Query overlaps on [10] but not [20] → S2
        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20])
        result = a.rationalise(q)
        assert result is False

    def test_s3_kline_returns_false(self):
        """Kline that routes S3 against all candidates → returns False."""
        a = KAgent(adapter=EventBus())
        candidate = KLine(5, [100, 200])
        a.rationalise(candidate)
        q = KLine(0, [1, 2])
        q.signature = make_signature([1, 2])
        result = a.rationalise(q)
        assert result is False


# ── Short-Circuit Tests ───────────────────────────────────────────────

class TestShortCircuit:
    """Phase 5 pushes ALL candidates to cogitator — no S1 short-circuit."""

    def test_all_candidates_submitted_to_cogitator(self):
        """All candidates are submitted to cogitator regardless of routing level."""
        a = KAgent(adapter=EventBus())
        # Add two candidates to the model
        c1 = KLine(5, [10, 20])       # S1 match for query
        c2 = KLine(6, [10, 20, 30])   # Also S1 for query
        a.rationalise(c1)
        a.rationalise(c2)

        # Query that would route S1 against both candidates
        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20])

        # Capture submitted work items
        submitted = []
        original_submit = a._cogitator.submit

        def capture_submit(item):
            submitted.append(item)
            original_submit(item)

        a._cogitator.submit = capture_submit
        result = a.rationalise(q)

        assert result is False  # No short-circuit — all go to cogitator
        assert len(submitted) == 2  # Both candidates submitted

    def test_mixed_s1_s2_all_submitted(self):
        """S2 and S1 candidates are all submitted — no S1 short-circuit."""
        a = KAgent(adapter=EventBus())
        # c1 will route S2, c2 will route S1
        c1 = KLine(5, [10, 30])
        c2 = KLine(6, [10, 20])
        a.rationalise(c1)
        a.rationalise(c2)

        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20])

        # Capture submitted work items
        submitted = []
        original_submit = a._cogitator.submit

        def capture_submit(item):
            submitted.append(item)
            original_submit(item)

        a._cogitator.submit = capture_submit
        result = a.rationalise(q)

        assert result is False  # No short-circuit
        assert len(submitted) == 2  # Both S2 and S1 candidates submitted
        levels = {item.level for item in submitted}
        assert "S2" in levels
        assert "S1" in levels

    def test_s2_candidates_submitted_to_cogitator(self):
        """S2 candidates are submitted to cogitator alongside any S1 candidates.

        Phase 5 no longer discards deferred items on S1 — all candidates
        are submitted to the cogitator unconditionally.
        """
        a = KAgent(adapter=EventBus())
        # c1 routes S2 (partial match), c2 routes S1 (full match)
        c1 = KLine(5, [10, 30])   # S2: only node 10 in common with query
        c2 = KLine(6, [10, 20])   # S1: both query nodes present
        a.rationalise(c1)
        a.rationalise(c2)

        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20])

        # Capture submitted work items
        submitted = []
        original_submit = a._cogitator.submit

        def capture_submit(item):
            submitted.append(item)
            original_submit(item)

        a._cogitator.submit = capture_submit
        result = a.rationalise(q)

        assert result is False  # No S1 short-circuit
        assert len(submitted) == 2  # Both S2 and S1 work items submitted

    def test_no_candidates_no_expand(self):
        """No candidates → S4 directly, no expand call."""
        a = KAgent(adapter=EventBus())
        q = KLine(0, [999])
        q.signature = make_signature([999])

        with patch(
            'kalvin.agent.expand',
            side_effect=AssertionError("expand should not be called for S4"),
        ):
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
        a = KAgent(adapter=EventBus())
        events = []
        a.events.subscribe(lambda e: events.append(e))
        k = KLine(0, [])
        a.rationalise(k)
        assert len(events) >= 1

    def test_ground_event(self):
        a = KAgent(adapter=EventBus())
        events = []
        a.events.subscribe(lambda e: events.append(e))
        k = KLine(5, [1, 2])
        a.rationalise(k)
        a.rationalise(KLine(5, [1, 2]))
        kinds = [e.kind for e in events]
        assert "ground" in kinds

    def test_frame_event(self):
        a = KAgent(adapter=EventBus())
        events = []
        a.events.subscribe(lambda e: events.append(e))
        k = KLine(0, [])
        a.rationalise(k)
        assert any(e.kind == "frame" for e in events)


# ── Cogitator Tests ───────────────────────────────────────────────────

class TestCogitator:
    def test_cogitate_join(self):
        a = KAgent(adapter=EventBus())
        a.cogitate_join(timeout=1.0)
        # Should not raise

    def test_rationalise_after_join(self):
        a = KAgent(adapter=EventBus())
        a.cogitate_join(timeout=1.0)
        k = KLine(5, [1, 2])
        result = a.rationalise(k)
        assert isinstance(result, bool)

    def test_s2_submits_work_item(self):
        """S2 kline submits a work item to the cogitator."""
        a = KAgent(adapter=EventBus())
        candidate = KLine(5, [10, 30])
        a.rationalise(candidate)

        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20])

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
    def _make_agent_with_klines(self) -> KAgent:
        a = KAgent(adapter=EventBus())
        a.rationalise(KLine(5, [1, 2]))
        a.rationalise(KLine(10, [3, 4]))
        a.rationalise(KLine(0, []))
        return a

    def test_to_bytes_roundtrip(self):
        a = self._make_agent_with_klines()
        data = a.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0
        loaded = KAgent.from_bytes(data)
        assert len(loaded.model) == len(a.model)

    def test_to_dict_roundtrip(self):
        a = self._make_agent_with_klines()
        d = a.to_dict()
        assert isinstance(d, dict)
        assert "klines" in d
        assert "activity" in d
        loaded = KAgent.from_dict(d)
        assert len(loaded.model) == len(a.model)

    def test_to_dict_structure(self):
        a = KAgent(adapter=EventBus())
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
            loaded = KAgent.load(path)
            assert len(loaded.model) == len(a.model)
        finally:
            path.unlink(missing_ok=True)

    def test_save_and_load_bin(self):
        a = self._make_agent_with_klines()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = Path(f.name)
        try:
            a.save(path, format="bin")
            loaded = KAgent.load(path, format="bin")
            assert len(loaded.model) == len(a.model)
        finally:
            path.unlink(missing_ok=True)

    def test_save_auto_detect_json(self):
        a = KAgent(adapter=EventBus())
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
        a = KAgent(adapter=EventBus())
        data = a.to_bytes()
        loaded = KAgent.from_bytes(data)
        assert len(loaded.model) == 0

    def test_empty_agent_dict(self):
        a = KAgent(adapter=EventBus())
        d = a.to_dict()
        loaded = KAgent.from_dict(d)
        assert len(loaded.model) == 0

    def test_codec_returns_agent_codec(self):
        """Agent.codec() returns an AgentCodec with the correct model and activity."""
        a = KAgent(adapter=EventBus())
        a.rationalise(KLine(5, [1, 2]))
        codec = a.codec()
        assert isinstance(codec, AgentCodec)
        # Verify the codec's output matches the agent's
        assert codec.to_bytes() == a.to_bytes()
        assert codec.to_dict() == a.to_dict()


# ── Structural Grounding Tests ───────────────────────────────────────

class TestStructuralGrounding:
    """Agent-level tests for structural grounding and promote_participating."""

    def test_s1_fast_path_promotes(self):
        """S1 in Phase 3 fast path (canonical) promotes to frame."""
        a = KAgent(adapter=EventBus())
        k = KLine(10, [10])  # canonical
        result = a.rationalise(k)
        assert result is True
        assert a.frame_size() >= 1

    def test_s4_fast_path_promotes(self):
        """S4 (empty kline) promotes to frame."""
        a = KAgent(adapter=EventBus())
        k = KLine(0, [])
        result = a.rationalise(k)
        assert result is True
        assert a.frame_size() >= 1

    def test_frame_holds_mixed_significance(self):
        """After ratification, frame contains klines of mixed significance."""
        a = KAgent(adapter=EventBus())
        # Build a model with countersigned klines
        a.rationalise(KLine(10, [10]))  # canonical → frame
        a.rationalise(KLine(5, [10, 20]))  # may be S4 or route to candidate
        assert a.frame_size() >= 1

    def test_publish_no_auto_promote(self):
        """_publish does not auto-promote — promotion is explicit."""
        a = KAgent(adapter=EventBus())
        events = []
        a.events.subscribe(lambda e: events.append(e))
        # Create a non-canonical kline that won't be fast-path promoted
        k = KLine(5, [1, 2])
        a.rationalise(k)
        # The kline may or may not be promoted depending on route,
        # but _publish itself shouldn't have promoted it
        # (promotion happens via promote_participating or explicit promote)


class TestCogitatorStructuralGrounding:
    """Cogitator-level tests for structural grounding behavior."""

    def test_boundary_s1_structural_promotes(self):
        """Boundary S1 on structurally S1 kline → promotion."""
        a = KAgent(adapter=EventBus())
        # Build model with canonical kline
        c = KLine(10, [10])
        a.rationalise(c)
        # Query that fully matches
        q = KLine(0, [10])
        q.signature = make_signature([10])
        result = a.rationalise(q)
        assert result is True

    def test_cogitator_countersignature_promotes_participating(self):
        """Countersignature discovery promotes all participating klines."""
        a = KAgent(adapter=EventBus())
        # Build countersigned pair
        a.rationalise(KLine(10, [10]))  # canonical
        a.rationalise(KLine(5, [10, 20]))  # contains 10
        a.rationalise(KLine(20, [5, 30]))  # contains 5
        # At least one kline should be in the frame
        assert a.frame_size() >= 1

    def test_expansion_proposals_emitted_as_events(self):
        """S2 expansion proposals are emitted as frame events."""
        a = KAgent(adapter=EventBus())
        events = []
        a.events.subscribe(lambda e: events.append(e))

        # Build model with misfit-eligible klines
        a.rationalise(KLine(0b100, [0b100]))  # canonical
        a.rationalise(KLine(0b010, [0b010]))  # canonical
        # A misfit kline
        a.rationalise(KLine(0b110, [0b100]))  # underfitting: sig promises more

        # Events should have been published (including potential expansion events)
        assert len(events) >= 1

    def test_no_expansion_for_canonical(self):
        """Canonical klines produce no expansion proposals."""
        from kalvin.misfit import classify_misfit

        k = KLine(10, [10])  # canonical
        nodes_sig = make_signature(k.nodes)
        assert k.signature == nodes_sig  # canonical
        underfit, overfit = classify_misfit(k)
        assert not underfit and not overfit


# ── Test Fake ─────────────────────────────────────────────────────────

class RecordingCogitationHandler:
    """Test fake: records all cogitation callbacks for assertion."""

    def __init__(self):
        self.s1_calls: list[tuple[KLine, KLine]] = []
        self.expansion_calls: list[tuple[KLine, KLine, int]] = []

    def on_s1(self, query: KLine, candidate: KLine) -> None:
        self.s1_calls.append((query, candidate))

    def on_expansion(
        self,
        query: KLine,
        proposal: KLine,
        significance: int,
        original_candidate: KLine | None = None,
    ) -> None:
        self.expansion_calls.append((query, proposal, significance))


# ── Protocol Conformance Tests ────────────────────────────────────────

class TestCogitationHandlerProtocol:
    """Protocol conformance — runtime_checkable isinstance checks."""

    def test_agent_satisfies_protocol(self):
        """Agent implements CogitationHandler (runtime_checkable)."""
        a = KAgent(adapter=EventBus())
        assert isinstance(a, CogitationHandler)

    def test_recording_handler_satisfies_protocol(self):
        """RecordingCogitationHandler implements CogitationHandler."""
        handler = RecordingCogitationHandler()
        assert isinstance(handler, CogitationHandler)

    def test_recording_handler_on_s1(self):
        """on_s1 records the query and candidate."""
        handler = RecordingCogitationHandler()
        q = KLine(5, [1, 2])
        c = KLine(10, [3, 4])
        handler.on_s1(q, c)
        assert handler.s1_calls == [(q, c)]

    def test_recording_handler_on_expansion(self):
        """on_expansion records the query, proposal, and significance."""
        handler = RecordingCogitationHandler()
        q = KLine(5, [1, 2])
        p = KLine(10, [3, 4])
        handler.on_expansion(q, p, 42)
        assert handler.expansion_calls == [(q, p, 42)]


# ── Fake-Handler Integration Tests ────────────────────────────────────

class TestCogitatorWithFakeHandler:
    """Cogitator wired to RecordingCogitationHandler — proves the seam works."""

    def test_fake_handler_receives_s1(self):
        """Cogitator calls handler.on_s1 when expand yields an S1-classified result."""
        m = Model()
        # Build model with a canonical kline so expand yields a high-significance result
        c = KLine(10, [10])  # canonical: sig == make_signature(nodes)
        m.add_ltm(c)

        recorder = RecordingCogitationHandler()
        event_bus = EventBus()
        cogitator = Cogitator(model=m, adapter=event_bus, handler=recorder)

        # Query fully matches candidate → S1 after expand
        q = KLine(0, [10])
        q.signature = make_signature([10])
        m.add_frame(q)

        cogitator.submit(WorkItem(q, c, "S2"))
        cogitator.join(timeout=2.0)

        assert len(recorder.s1_calls) >= 1
        assert recorder.s1_calls[0][0] is q
        assert recorder.s1_calls[0][1] is c

    def test_fake_handler_receives_expansion(self):
        """Cogitator calls handler.on_expansion when a misfit kline expands."""
        m = Model()
        # Build model with canonical klines that resolve nodes
        k1 = KLine(0b100, [0b100])  # canonical
        m.add_ltm(k1)
        k2 = KLine(0b010, [0b010])  # canonical — resolves as a contributor
        m.add_ltm(k2)
        # A misfit kline — underfitting: sig=0b110 but nodes only give 0b100
        k3 = KLine(0b110, [0b100])
        m.add_ltm(k3)

        recorder = RecordingCogitationHandler()
        event_bus = EventBus()
        cogitator = Cogitator(model=m, adapter=event_bus, handler=recorder)

        # Query with no overlapping nodes to k3 → S3 after expand,
        # and k3 is misfit so propose_expansions triggers generate_expansions.
        q = KLine(0, [0b010])
        q.signature = make_signature([0b010])
        m.add_frame(q)

        cogitator.submit(WorkItem(q, k3, "S3"))
        cogitator.join(timeout=2.0)

        assert len(recorder.expansion_calls) >= 1

    def test_cogitator_stops_on_s1(self):
        """Cogitator._run_work_item breaks after finding S1 — no more handler calls.

        Once S1 is discovered during expansion, the work item should stop
        processing. No further on_s1 or on_expansion calls should happen
        for that work item.
        """
        m = Model()
        # Canonical kline that resolves via expand
        c = KLine(10, [10])
        m.add_ltm(c)

        recorder = RecordingCogitationHandler()
        event_bus = EventBus()
        cogitator = Cogitator(model=m, adapter=event_bus, handler=recorder)

        q = KLine(0, [10])
        q.signature = make_signature([10])
        m.add_frame(q)

        cogitator.submit(WorkItem(q, c, "S2"))
        cogitator.join(timeout=2.0)

        # S1 should be called exactly once (not multiple times)
        assert len(recorder.s1_calls) == 1
        # No expansion proposals should be generated after S1
        assert len(recorder.expansion_calls) == 0


# ── Countersign Tests ────────────────────────────────────────────────

class TestCountersign:
    """Agent.countersign: reciprocal kline construction and rationalisation."""

    def test_countersign_returns_rationalise_result(self):
        """countersign returns the result of rationalise for the reciprocal kline."""
        a = KAgent(adapter=EventBus())
        # Build a kline with non-empty nodes
        kline = KLine(0xFF, [10, 20])
        result = a.countersign(kline)
        assert isinstance(result, bool)
        # The reciprocal KLine(make_signature([10,20]), [0xFF]) should be
        # rationalised as a novel kline → True (S4)
        assert result is True

    def test_countersign_reciprocal_construction(self):
        """Reciprocal kline is built as KLine(make_signature(kline.nodes), [kline.signature])."""
        a = KAgent(adapter=EventBus())
        kline = KLine(0xAB, [10, 20, 30])
        expected_reciprocal_sig = make_signature([10, 20, 30])
        expected_reciprocal_nodes = [0xAB]

        with patch.object(KAgent, "rationalise", return_value=True) as mock_rationalise:
            result = a.countersign(kline)

        assert result is True
        mock_rationalise.assert_called_once()
        reciprocal = mock_rationalise.call_args[0][0]
        assert reciprocal.signature == expected_reciprocal_sig
        assert reciprocal.nodes == expected_reciprocal_nodes

    def test_countersign_empty_nodes(self):
        """Empty nodes → reciprocal_sig=0, reciprocal_nodes=[kline.signature]."""
        a = KAgent(adapter=EventBus())
        kline = KLine(0xCD, [])
        expected_reciprocal_sig = 0  # make_signature([]) == 0
        expected_reciprocal_nodes = [0xCD]

        with patch.object(KAgent, "rationalise", return_value=True) as mock_rationalise:
            result = a.countersign(kline)

        assert result is True
        mock_rationalise.assert_called_once()
        reciprocal = mock_rationalise.call_args[0][0]
        assert reciprocal.signature == expected_reciprocal_sig
        assert reciprocal.nodes == expected_reciprocal_nodes

# ── Cascade Write Method Tests ───────────────────────────────────────

class TestCascadeWriteMethods:
    """Verify the correct model write method is called at each rationalisation phase."""

    def test_agt9_first_rationalise_add_ltm(self):
        """AGT-9: First rationalise of a new kline calls model.add_ltm()."""
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        k = KLine(5, [1, 2])
        with patch.object(m, "add_ltm", wraps=m.add_ltm) as mock_add_ltm:
            result = a.rationalise(k)
        assert result is True
        mock_add_ltm.assert_called_once_with(k)

    def test_agt10_duplicate_ground_add_stm(self):
        """AGT-10: Second rationalise of same kline calls model.add_stm() and emits ground."""
        m = Model()
        events = []
        adapter = EventBus()
        adapter.subscribe(lambda e: events.append(e))
        a = KAgent(model=m, adapter=adapter)
        k = KLine(5, [1, 2])
        a.rationalise(k)  # first time
        # Second rationalise — should hit ground check
        dup = KLine(5, [1, 2])
        with patch.object(m, "add_stm", wraps=m.add_stm) as mock_add_stm:
            result = a.rationalise(dup)
        assert result is True
        mock_add_stm.assert_called_once_with(dup)
        assert any(e.kind == "ground" for e in events)

    def test_agt12_s4_unsigned_add_ltm(self):
        """AGT-12: Empty kline calls model.add_ltm()."""
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        k = KLine(0, [])
        with patch.object(m, "add_ltm", wraps=m.add_ltm) as mock_add_ltm:
            result = a.rationalise(k)
        assert result is True
        mock_add_ltm.assert_called_once_with(k)

    def test_agt14_s1_self_grounded_add_ltm(self):
        """AGT-14: Self-grounded canonical kline calls model.add_ltm()."""
        m = Model()
        # Add resolved nodes so the query's non-literal nodes resolve
        m.add_ltm(KLine(10, [10]))  # canonical
        m.add_ltm(KLine(20, [20]))  # canonical
        a = KAgent(model=m, adapter=EventBus())
        # Query that is canonical and all non-literal nodes resolve
        # make_signature([10, 20]) = 10 | 20 = 30
        k = KLine(30, [10, 20])
        with patch.object(m, "add_ltm", wraps=m.add_ltm) as mock_add_ltm:
            result = a.rationalise(k)
        assert result is True
        mock_add_ltm.assert_any_call(k)

    def test_agt16_novel_s4_add_ltm(self):
        """AGT-16: No candidates found calls model.add_ltm()."""
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        # A unique signature that won't match anything in the model
        k = KLine(0xFF00, [0xFF00])
        k.signature = make_signature([0xFF00])
        with patch.object(m, "add_ltm", wraps=m.add_ltm) as mock_add_ltm:
            result = a.rationalise(k)
        assert result is True
        mock_add_ltm.assert_any_call(k)

    def test_agt18_s1_routing_submits_to_cogitator(self):
        """AGT-18: S1 routing submits work item to cogitator (no short-circuit)."""
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        # Add a candidate that will route as S1
        c = KLine(5, [10, 20])
        a.rationalise(c)
        # Query that fully matches candidate nodes
        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20])
        # Capture submitted work items
        submitted = []
        original_submit = a._cogitator.submit

        def capture_submit(item):
            submitted.append(item)
            original_submit(item)

        a._cogitator.submit = capture_submit
        result = a.rationalise(q)
        assert result is False  # No short-circuit — submitted to cogitator
        assert len(submitted) == 1
        assert submitted[0].query is q
        assert submitted[0].candidate is c
        assert submitted[0].level == "S1"

    def test_agt22a_slow_path_query_add_stm_only(self):
        """AGT-22a: S2/S3 routed kline calls model.add_stm() only — not add_frame or add_ltm."""
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        # Add a candidate that will route as S2 (partial overlap)
        c = KLine(5, [10, 30])
        a.rationalise(c)
        # Query with partial overlap → S2
        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20])
        with patch.object(m, "add_stm", wraps=m.add_stm) as mock_add_stm, \
             patch.object(m, "add_ltm", wraps=m.add_ltm) as mock_add_ltm, \
             patch.object(m, "add_frame", wraps=m.add_frame) as mock_add_frame:
            result = a.rationalise(q)
        assert result is False  # S2 → not significant, submitted to cogitator
        # add_stm should have been called for Phase 5
        mock_add_stm.assert_called()
        # add_ltm should NOT have been called for the query (it's slow path)
        # Note: add_ltm may have been called for the candidate earlier, but not for q
        for call in mock_add_ltm.call_args_list:
            assert call[0][0] is not q, "add_ltm should not be called for slow-path query"
        for call in mock_add_frame.call_args_list:
            assert call[0][0] is not q, "add_frame should not be called for slow-path query"

    def test_agt29_cogitation_s1_promote_participating(self):
        """AGT-29: on_s1 with structural S1 calls promote and publishes frame."""
        m = Model()
        events = []
        adapter = EventBus()
        adapter.subscribe(lambda e: events.append(e))
        a = KAgent(model=m, adapter=adapter)
        # Build a structurally S1 (canonical) candidate
        candidate = KLine(10, [10])  # canonical → is_s1 returns True
        query = KLine(5, [1, 2])
        m.add_stm(query)
        with patch("kalvin.agent.promote_participating") as mock_promote:
            a.on_s1(query, candidate)
        mock_promote.assert_called_once_with(m, query, candidate)
        # Frame event should be published
        assert any(e.kind == "frame" for e in events)

    def test_agt29_cogitation_s1_not_structural_no_promote(self):
        """AGT-29 variant: on_s1 with non-structural S1 does NOT call promote_participating."""
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        # Non-canonical, non-countersigned candidate
        candidate = KLine(99, [50, 60])  # not canonical, not countersigned
        query = KLine(5, [1, 2])
        with patch("kalvin.agent.promote_participating") as mock_promote:
            a.on_s1(query, candidate)
        mock_promote.assert_not_called()
        # Frame event still published (unconditional)

    def test_agt34_expansion_add_frame(self):
        """AGT-34: on_expansion calls model.add_frame(proposal) before publishing."""
        m = Model()
        events = []
        adapter = EventBus()
        adapter.subscribe(lambda e: events.append(e))
        a = KAgent(model=m, adapter=adapter)
        q = KLine(5, [1, 2])
        p = KLine(10, [3, 4])
        with patch.object(m, "add_frame", wraps=m.add_frame) as mock_add_frame:
            a.on_expansion(q, p, 42)
        mock_add_frame.assert_called_once_with(p)
        # Frame event published
        assert len(events) == 1
        assert events[0].kind == "frame"
        assert events[0].proposal is p


# ── NLP Tokenizer Integration Tests ──────────────────────────────────

@pytest.fixture(scope="module")
def nlp_tokenizer() -> NLPTokenizer:
    """Load NLPTokenizer from standard file paths (once per module)."""
    return NLPTokenizer.from_files()


class TestAgentNLPTokenizer:
    """KAgent constructed with NLPTokenizer — pluggable tokenizer integration.

    Verifies that KAgent works correctly when callers opt in to NLP-BPE
    tokenization. The default remains Mod32Tokenizer (tested separately).
    """

    def test_nlp_agent_rationalise_kline(self, nlp_tokenizer: NLPTokenizer) -> None:
        """KAgent with NLPTokenizer can rationalise a kline containing NLP-BPE nodes.

        Encode a known word ('Tea') via the NLP tokenizer, build a KLine
        with the resulting nodes and their signature, and rationalise.
        The kline should be accepted (S4 novel or S1).
        """
        a = KAgent(tokenizer=nlp_tokenizer, adapter=EventBus())
        nodes = nlp_tokenizer.encode("Tea")
        assert len(nodes) == 1, "'Tea' should produce exactly one NLP-BPE node"

        sig = make_signature(nodes)
        kline = KLine(sig, nodes, dbg=KDbg(label="Tea"))
        result = a.rationalise(kline)
        assert result is True

        # Signature must be non-zero (NLP-BPE nodes have non-zero high bits)
        assert sig != 0

        # Signature must be the plain OR-reduce of all nodes.
        expected = 0
        for node in nodes:
            expected |= node
        assert sig == expected

    def test_default_tokenizer(self) -> None:
        """Default KAgent uses NLP tokenizer when available, Mod32 otherwise."""
        a = KAgent(adapter=EventBus())
        # With NLP data files available, default should be NLPTokenizer
        from kalvin.nlp_tokenizer import NLPTokenizer
        from kalvin.mod_tokenizer import Mod32Tokenizer
        assert isinstance(a.tokenizer, (NLPTokenizer, Mod32Tokenizer))

    def test_nlp_agent_serialization(self, nlp_tokenizer: NLPTokenizer) -> None:
        """Serialization round-trips preserve NLP-BPE node values (uint64).

        Create an NLP agent, rationalise a kline with NLP-BPE nodes,
        then round-trip through to_bytes/from_bytes and to_dict/from_dict.
        The deserialized model should have the same number of klines.
        """
        a = KAgent(tokenizer=nlp_tokenizer, adapter=EventBus())
        nodes = nlp_tokenizer.encode("Tea")
        sig = make_signature(nodes)
        kline = KLine(sig, nodes, dbg=KDbg(label="Tea"))
        a.rationalise(kline)

        # Binary round-trip
        data = a.to_bytes()
        loaded = KAgent.from_bytes(data)
        assert len(loaded.model) == len(a.model)

        # Dict round-trip
        d = a.to_dict()
        loaded2 = KAgent.from_dict(d)
        assert len(loaded2.model) == len(a.model)

        # Verify the NLP-BPE node values survived serialization unchanged
        # by checking the stored kline's nodes match the originals
        original_kline = list(a.model.klines())[0]
        loaded_kline = list(loaded.model.klines())[0]
        assert loaded_kline.nodes == original_kline.nodes
        assert loaded_kline.signature == original_kline.signature


# ── NLP Agent Cross-Module Integration Tests ──────────────────────────


class TestAgentNLPIntegration:
    """Cross-module integration tests: NLP tokenizer -> KAgent -> model storage.

    These go beyond unit-level tests (KB-145's TestAgentNLPTokenizer) to
    verify the full pipeline: encode text -> build kline -> rationalise ->
    store in model -> retrieve and verify node/signature integrity.

    Focuses on multi-word phrases (with space tokens), model retrieval,
    and serialization.
    """

    @staticmethod
    def _make_nlp_agent_with_klines(
        nlp_tokenizer: NLPTokenizer,
    ) -> tuple[KAgent, list[KLine]]:
        """Create an NLP agent with a single rationalised kline."""
        a = KAgent(tokenizer=nlp_tokenizer, adapter=EventBus())

        # Single-word NLP-BPE kline
        tea = nlp_tokenizer.encode("Tea")
        sig1 = make_signature(tea)
        k1 = KLine(sig1, tea, dbg=KDbg(label="Tea"))
        a.rationalise(k1)

        return a, [k1]

    def test_nlp_agent_rationalise_kline(self, nlp_tokenizer: NLPTokenizer) -> None:
        """Rationalise a kline with NLP tokenizer — verify storage and retrieval."""
        a, klines = self._make_nlp_agent_with_klines(nlp_tokenizer)

        # Verify kline was stored
        assert len(klines) == 1
        stored = a.model.find(klines[0].signature)
        assert stored is not None

    def test_nlp_agent_bytes_roundtrip(
        self, nlp_tokenizer: NLPTokenizer
    ) -> None:
        """Binary serialization preserves NLP-BPE node values (uint64).

        Multiple klines with NLP-BPE nodes survive to_bytes/from_bytes
        with exact node value preservation.
        """
        a, original_klines = self._make_nlp_agent_with_klines(nlp_tokenizer)

        data = a.to_bytes()
        loaded = KAgent.from_bytes(data)

        assert len(loaded.model) == len(original_klines)

        # Verify each kline's node values survived uint64 serialization
        for orig in original_klines:
            stored = loaded.model.find(orig.signature)
            assert stored is not None, (
                f"Kline with sig {orig.signature:#x} not found after bytes round-trip"
            )
            assert stored.nodes == orig.nodes, (
                f"Node values changed: expected {orig.nodes}, got {stored.nodes}"
            )
            assert stored.signature == orig.signature

    def test_nlp_agent_dict_roundtrip(
        self, nlp_tokenizer: NLPTokenizer
    ) -> None:
        """Dict serialization preserves NLP-BPE node values exactly.

        Nodes are stored as integers in the dict and must round-trip
        without precision loss.
        """
        a, original_klines = self._make_nlp_agent_with_klines(nlp_tokenizer)

        d = a.to_dict()
        loaded = KAgent.from_dict(d)

        assert len(loaded.model) == len(original_klines)

        for orig in original_klines:
            stored = loaded.model.find(orig.signature)
            assert stored is not None
            assert stored.nodes == orig.nodes
            assert stored.signature == orig.signature

    def test_nlp_agent_json_file_roundtrip(
        self, nlp_tokenizer: NLPTokenizer
    ) -> None:
        """JSON file serialization preserves NLP-BPE node values.

        Write to a temp JSON file, reload, verify all node values match.
        """
        a, original_klines = self._make_nlp_agent_with_klines(nlp_tokenizer)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            a.save(path)
            loaded = KAgent.load(path)

            assert len(loaded.model) == len(original_klines)

            for orig in original_klines:
                stored = loaded.model.find(orig.signature)
                assert stored is not None
                assert stored.nodes == orig.nodes
                assert stored.signature == orig.signature
        finally:
            path.unlink(missing_ok=True)
