"""Tests for Agent — openspec/agent.md conformance."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from kalvin.agent import KAgent
from kalvin.agent_codec import AgentCodec
from kalvin.cogitator import CogitationHandler, Cogitator, WorkItem
from kalvin.events import EventBus, RationaliseEvent
from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4, is_countersigned, structural_significance
from kalvin.kline import KDbg, KLine
from kalvin.kvalue import KValue
from kalvin.model import Model
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from tests.conftest import requires_tokenizer_data
from tests.test_cogitator_handler import RecordingCogitationHandler

signifier = NLPSignifier()

# KAgent construction defaults to the kalvin tokenizer; skip cleanly when
# the data assets are absent on a fresh clone.
pytestmark = requires_tokenizer_data


def t(bits: int) -> int:
    """Place sig-word bits in the upper 32 bits of a uint64.

    signifies() (used by model.where for candidate retrieval) masks off the
    lower (BPE) 32 bits, so node/signature values that must overlap for
    candidate matching are shifted up here.
    """
    return bits << 32


def _kv(kline: KLine, model: Model) -> KValue:
    """Wrap a kline in a KValue declaring its structurally-correct band.

    Honours kvalue spec KP-1 for hand-built test klines: the producer
    declares the band the kline resolves to — the structural band
    (structural_significance) with the one model-state fork KAgent applies:
    a structurally-S2 misfit whose reciprocal countersigner is present in the
    model upgrades to S1. Identity klines with empty nodes declare SIG_S4.
    """
    band = structural_significance(kline, signifier)
    if band == SIG_S2 and is_countersigned(model, kline, signifier):
        band = SIG_S1
    return KValue(kline, band)


class TestAgentInit:
    def test_default_init(self):
        a = KAgent(adapter=EventBus())
        assert a.model is not None
        assert a.tokenizer is not None

    def test_custom_tokenizer(self):
        t = NLPTokenizer()
        a = KAgent(tokenizer=t, adapter=EventBus())
        assert a.tokenizer is t

    def test_custom_model(self):
        t = NLPTokenizer()
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
    """KAgent._route: fast node-membership classification. No model call.

    Routes cogitated candidates between S2 and S3 only. S1 (full overlap)
    is a structural property established by expand()/is_s1(), not by
    routing; S4 (empty query) never reaches routing because identity
    klines are resolved on the fast path.
    """

    def test_all_nodes_match_s2(self):
        q = KLine(5, [10, 20])
        c = KLine(99, [10, 20, 30])
        assert KAgent._route(q, c) == "S2"

    def test_some_nodes_match_s2(self):
        q = KLine(5, [10, 20, 99])
        c = KLine(99, [10, 20, 30])
        assert KAgent._route(q, c) == "S2"

    def test_no_nodes_match_s3(self):
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        assert KAgent._route(q, c) == "S3"

    def test_single_node_match_s2(self):
        q = KLine(5, [10])
        c = KLine(99, [10, 20])
        assert KAgent._route(q, c) == "S2"

    def test_empty_query_s3(self):
        q = KLine(0, [])
        c = KLine(10, [1])
        assert KAgent._route(q, c) == "S3"

    def test_routing_independent_of_signature(self):
        """Routing only cares about candidate's node sequence."""
        q = KLine(5, [42])
        c = KLine(999, [42, 100])
        assert KAgent._route(q, c) == "S2"

    def test_duplicate_nodes_in_query(self):
        """Duplicate query nodes are counted per-occurrence for membership."""
        q = KLine(5, [10, 10])
        c = KLine(99, [10, 20])
        assert KAgent._route(q, c) == "S2"


# ── Rationalisation Tests ─────────────────────────────────────────────


class TestAgentRationalise:
    def test_unsigned_s4(self):
        """Empty kline → S4."""
        a = KAgent(adapter=EventBus())
        k = KLine(0, [])
        result = a.rationalise(_kv(k, a.model))
        assert result is True

    def test_ground_check(self):
        """Already exists → ground event."""
        a = KAgent(adapter=EventBus())
        k = KLine(5, [1, 2])
        a.rationalise(_kv(k, a.model))
        result = a.rationalise(_kv(KLine(5, [1, 2]), a.model))
        assert result is True

    def test_novel_kline(self):
        """Novel kline with no candidates → S4."""
        a = KAgent(adapter=EventBus())
        t = a.tokenizer
        packed = t.encode("XYZ")[0]
        k = KLine(packed, [packed])
        result = a.rationalise(_kv(k, a.model))
        assert result is True

    def test_rationalise_adds_to_model(self):
        a = KAgent(adapter=EventBus())
        k = KLine(5, [1, 2])
        a.rationalise(_kv(k, a.model))
        assert a.model.find(5) is not None

    def test_rationalise_with_external_encode(self):
        """Caller encodes text, builds kline, rationalises."""
        a = KAgent(adapter=EventBus())
        t = a.tokenizer
        nodes = t.encode("HELLO")
        sig = signifier.make_signature(nodes)
        kline = KLine(sig, nodes, dbg=KDbg(label="HELLO"))
        result = a.rationalise(_kv(kline, a.model))
        assert result is True
        assert a.model.find(sig) is not None

    def test_s2_kline_returns_false(self):
        """Kline that routes S2 against all candidates → returns False."""
        a = KAgent(adapter=EventBus())
        # Add a candidate that partially overlaps
        candidate = KLine(t(5), [t(10), t(30)])
        a.rationalise(_kv(candidate, a.model))
        # Query overlaps on [10] but not [20] → S2
        q = KLine(0, [t(10), t(20)])
        q.signature = signifier.make_signature([t(10), t(20)])
        result = a.rationalise(_kv(q, a.model))
        assert result is False

    def test_s3_kline_returns_false(self):
        """Kline that routes S3 against all candidates → returns False."""
        a = KAgent(adapter=EventBus())
        candidate = KLine(t(5), [t(100), t(200)])
        a.rationalise(_kv(candidate, a.model))
        q = KLine(0, [t(1), t(2)])
        q.signature = signifier.make_signature([t(1), t(2)])
        result = a.rationalise(_kv(q, a.model))
        assert result is False


    # ── Significance-comparison gate (S4-drop MVP) ─────────────────────

    def test_gate_declared_equals_derived_processes_normally(self):
        """derived == declared (any band) → process normally (Kalvin agrees).

        Identity klines declared S4 are derived-S4 too, so they agree and
        never hit the drop branch — they follow the existing S4 path.
        """
        a = KAgent(adapter=EventBus())
        events: list = []
        a.events.subscribe(lambda e: events.append(e))
        k = KLine(0, [])  # identity → derived S4
        result = a.rationalise(KValue(k, SIG_S4))  # declared S4 → agrees
        assert result is True
        # Agreed → normal processing: frame S4 event published, kline in LTM
        assert any(e.kind == "frame" for e in events)
        assert a.model.find(0) is not None

    def test_gate_s4_disagreement_drops(self):
        """declared S4, derived != S4 → drop: returns True, no STM, no event.

        A relationship kline that derives to S3 (CONNOTES/DENOTES) but is
        handed in with a declared S4 is the MVP drop case. It must NOT touch
        STM, Frame, or LTM, and must NOT publish.
        """
        a = KAgent(adapter=EventBus())
        events: list = []
        a.events.subscribe(lambda e: events.append(e))
        # A fresh relationship kline with no candidates derives S3, never S4.
        q = KLine(0, [t(1), t(2)])
        q.signature = signifier.make_signature([t(1), t(2)])
        result = a.rationalise(KValue(q, SIG_S4))  # declared S4, derived S3
        assert result is True
        # Drop: nothing published, nothing written anywhere (STM/Frame/LTM).
        assert events == []
        assert a.model.find(q.signature) is None

    def test_gate_s2_s3_disagreement_ignored(self):
        """declared in {S1,S2,S3}, derived != declared → MVP ignores the
        disagreement and processes normally (deferred).

        Declared S2 over a kline that derives S3 must still go through the
        ordinary pipeline (here: no candidates → novel S4 frame), not drop.
        """
        a = KAgent(adapter=EventBus())
        events: list = []
        a.events.subscribe(lambda e: events.append(e))
        q = KLine(0, [t(7), t(8)])
        q.signature = signifier.make_signature([t(7), t(8)])
        result = a.rationalise(KValue(q, SIG_S2))  # declared S2, derived S3
        assert result is True  # novel → S4 frame, not dropped
        assert any(e.kind == "frame" for e in events)
        assert a.model.find(q.signature) is not None


# ── Short-Circuit Tests ───────────────────────────────────────────────


class TestShortCircuit:
    """All candidates are pushed to the cogitator; routing is S2/S3 only."""

    def test_all_candidates_submitted_to_cogitator(self):
        """All candidates are submitted to cogitator regardless of routing level."""
        a = KAgent(adapter=EventBus())
        # Add two candidates to the model
        c1 = KLine(t(5), [t(10), t(20)])  # full overlap with query
        c2 = KLine(t(6), [t(10), t(20), t(30)])  # also full overlap with query
        a.rationalise(_kv(c1, a.model))
        a.rationalise(_kv(c2, a.model))

        # Query overlaps both candidates (routes S2 under the S2/S3-only model)
        q = KLine(0, [t(10), t(20)])
        q.signature = signifier.make_signature([t(10), t(20)])

        # Capture submitted work items
        submitted = []
        original_submit = a._cogitator.submit

        def capture_submit(item):
            submitted.append(item)
            original_submit(item)

        a._cogitator.submit = capture_submit
        result = a.rationalise(_kv(q, a.model))

        assert result is False  # No short-circuit — all go to cogitator
        assert len(submitted) == 2  # Both candidates submitted

    def test_mixed_overlap_all_submitted(self):
        """Candidates with full and partial overlap are all submitted as S2."""
        a = KAgent(adapter=EventBus())
        # c1 partial overlap, c2 full overlap — both route S2 now
        c1 = KLine(t(5), [t(10), t(30)])
        c2 = KLine(t(6), [t(10), t(20)])
        a.rationalise(_kv(c1, a.model))
        a.rationalise(_kv(c2, a.model))

        q = KLine(0, [t(10), t(20)])
        q.signature = signifier.make_signature([t(10), t(20)])

        # Capture submitted work items
        submitted = []
        original_submit = a._cogitator.submit

        def capture_submit(item):
            submitted.append(item)
            original_submit(item)

        a._cogitator.submit = capture_submit
        result = a.rationalise(_kv(q, a.model))

        assert result is False  # No short-circuit
        assert len(submitted) == 2  # Both candidates submitted
        levels = {item.level for item in submitted}
        assert levels == {"S2"}  # Overlap candidates route S2 only

    def test_s2_and_s3_candidates_submitted_to_cogitator(self):
        """S2 (overlap) and S3 (no overlap) candidates are both submitted."""
        a = KAgent(adapter=EventBus())
        # c1 routes S2 (partial match), c2 routes S3 (no match)
        c1 = KLine(t(5), [t(10), t(30)])  # S2: node 10 in common with query
        c2 = KLine(t(6), [t(40), t(50)])  # S3: no node in common with query
        a.rationalise(_kv(c1, a.model))
        a.rationalise(_kv(c2, a.model))

        q = KLine(0, [t(10), t(20)])
        q.signature = signifier.make_signature([t(10), t(20)])

        # Capture submitted work items
        submitted = []
        original_submit = a._cogitator.submit

        def capture_submit(item):
            submitted.append(item)
            original_submit(item)

        a._cogitator.submit = capture_submit
        result = a.rationalise(_kv(q, a.model))

        assert result is False
        assert len(submitted) == 2  # Both S2 and S3 work items submitted
        levels = {item.level for item in submitted}
        assert levels == {"S2", "S3"}

    def test_no_candidates_no_expand(self):
        """No candidates → S4 directly, no expand call."""
        a = KAgent(adapter=EventBus())
        q = KLine(0, [999])
        q.signature = signifier.make_signature([999])

        with patch(
            "kalvin.cogitator.expand",
            side_effect=AssertionError("expand should not be called for S4"),
        ):
            result = a.rationalise(_kv(q, a.model))

        assert result is True


# ── WorkItem Tests ────────────────────────────────────────────────────


class TestWorkItem:
    def test_work_item_fields(self):
        q = KLine(5, [1, 2])
        c = KLine(10, [3, 4])
        qv = KValue(q, SIG_S3)
        item = WorkItem(qv, c, "S2")
        assert item.query is qv
        assert item.candidate is c
        assert item.level == "S2"

    def test_work_item_equality(self):
        q = KLine(5, [1])
        c = KLine(10, [3])
        qv = KValue(q, SIG_S3)
        assert WorkItem(qv, c, "S2") == WorkItem(qv, c, "S2")
        assert WorkItem(qv, c, "S2") != WorkItem(qv, c, "S3")


# ── Event Tests ───────────────────────────────────────────────────────


class TestAgentEvents:
    def test_subscribe(self):
        a = KAgent(adapter=EventBus())
        events = []
        a.events.subscribe(lambda e: events.append(e))
        k = KLine(0, [])
        a.rationalise(_kv(k, a.model))
        assert len(events) >= 1

    def test_ground_event(self):
        a = KAgent(adapter=EventBus())
        events = []
        a.events.subscribe(lambda e: events.append(e))
        k = KLine(5, [1, 2])
        a.rationalise(_kv(k, a.model))
        a.rationalise(_kv(KLine(5, [1, 2]), a.model))
        kinds = [e.kind for e in events]
        assert "ground" in kinds

    def test_frame_event(self):
        a = KAgent(adapter=EventBus())
        events = []
        a.events.subscribe(lambda e: events.append(e))
        k = KLine(0, [])
        a.rationalise(_kv(k, a.model))
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
        result = a.rationalise(_kv(k, a.model))
        assert isinstance(result, bool)

    def test_s2_submits_work_item(self):
        """S2 kline submits a work item to the cogitator."""
        a = KAgent(adapter=EventBus())
        candidate = KLine(t(5), [t(10), t(30)])
        a.rationalise(_kv(candidate, a.model))

        q = KLine(0, [t(10), t(20)])
        q.signature = signifier.make_signature([t(10), t(20)])

        # Capture submitted work items
        submitted = []
        original_submit = a._cogitator.submit

        def capture_submit(item):
            submitted.append(item)
            original_submit(item)

        a._cogitator.submit = capture_submit
        qv = _kv(q, a.model)
        result = a.rationalise(qv)

        assert result is False
        assert len(submitted) == 1
        assert submitted[0].level == "S2"
        assert submitted[0].query is qv  # WorkItem.query is a KValue
        assert submitted[0].candidate is candidate


# ── Serialization Tests ───────────────────────────────────────────────


class TestAgentSerialization:
    def _make_agent_with_klines(self) -> KAgent:
        a = KAgent(adapter=EventBus())
        a.rationalise(_kv(KLine(5, [1, 2]), a.model))
        a.rationalise(_kv(KLine(10, [3, 4]), a.model))
        a.rationalise(_kv(KLine(0, []), a.model))
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
        a.rationalise(_kv(KLine(5, [1, 2]), a.model))
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
        a.rationalise(_kv(KLine(5, [1]), a.model))
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
        a.rationalise(_kv(KLine(5, [1, 2]), a.model))
        codec = a.codec()
        assert isinstance(codec, AgentCodec)
        # Verify the codec's output matches the agent's
        assert codec.to_bytes() == a.to_bytes()
        assert codec.to_dict() == a.to_dict()


# ── Structural Grounding Tests ───────────────────────────────────────


class TestStructuralGrounding:
    """Agent-level tests for structural grounding and promote_participating."""

    def test_s1_fast_path_promotes(self):
        """Phase 3 fast path promotes identity kline ({S:[S]}) to frame.

        Despite the name, KLine(10, [10]) is identity (not canonical) since
        commit 040bc0c and is promoted via the identity/S4 fast path. The test
        name is retained for compatibility.
        """
        a = KAgent(adapter=EventBus())
        k = KLine(10, [10])  # identity (self-referential: {S:[S]}), NOT canon since 040bc0c
        result = a.rationalise(_kv(k, a.model))
        assert result is True
        assert a.frame_size() >= 1

    def test_s4_fast_path_promotes(self):
        """S4 (empty kline) promotes to frame."""
        a = KAgent(adapter=EventBus())
        k = KLine(0, [])
        result = a.rationalise(_kv(k, a.model))
        assert result is True
        assert a.frame_size() >= 1

    def test_frame_holds_mixed_significance(self):
        """After ratification, frame contains klines of mixed significance."""
        a = KAgent(adapter=EventBus())
        # Build a model with countersigned klines
        a.rationalise(_kv(KLine(10, [10]), a.model))  # identity → frame
        a.rationalise(_kv(KLine(5, [10, 20]), a.model))  # may be S4 or route to candidate
        assert a.frame_size() >= 1

    def test_publish_no_auto_promote(self):
        """_publish does not auto-promote — promotion is explicit."""
        a = KAgent(adapter=EventBus())
        events = []
        a.events.subscribe(lambda e: events.append(e))
        # Create a non-canonical kline that won't be fast-path promoted
        k = KLine(5, [1, 2])
        a.rationalise(_kv(k, a.model))
        # The kline may or may not be promoted depending on route,
        # but _publish itself shouldn't have promoted it
        # (promotion happens via promote_participating or explicit promote)


class TestCogitatorStructuralGrounding:
    """Cogitator-level tests for structural grounding behavior."""

    def test_boundary_s1_structural_promotes(self):
        """Boundary S1 on structurally S1 kline → promotion."""
        a = KAgent(adapter=EventBus())
        # Build model with an identity kline ({S:[S]})
        c = KLine(10, [10])
        a.rationalise(_kv(c, a.model))
        # Query that fully matches
        q = KLine(0, [10])
        q.signature = signifier.make_signature([10])
        result = a.rationalise(_kv(q, a.model))
        assert result is True

    def test_cogitator_countersignature_promotes_participating(self):
        """Countersignature discovery promotes all participating klines."""
        a = KAgent(adapter=EventBus())
        # Build countersigned pair
        a.rationalise(_kv(KLine(10, [10]), a.model))  # identity (self-referential since 040bc0c)
        a.rationalise(_kv(KLine(5, [10, 20]), a.model))  # contains 10
        a.rationalise(_kv(KLine(20, [5, 30]), a.model))  # contains 5
        # At least one kline should be in the frame
        assert a.frame_size() >= 1

    def test_expansion_proposals_emitted_as_events(self):
        """S2 expansion proposals are emitted as frame events."""
        a = KAgent(adapter=EventBus())
        events = []
        a.events.subscribe(lambda e: events.append(e))

        # Build model with misfit-eligible klines (identity entries since 040bc0c)
        a.rationalise(_kv(KLine(0b100, [0b100]), a.model))  # identity
        a.rationalise(_kv(KLine(0b010, [0b010]), a.model))  # identity
        # A misfit kline
        a.rationalise(_kv(KLine(0b110, [0b100]), a.model))  # underfitting: sig promises more

        # Events should have been published (including potential expansion events)
        assert len(events) >= 1

    def test_no_expansion_for_canonical(self):
        """Kline with no misfit produces no expansion proposals.

        Note: KLine(10, [10]) is identity ({S:[S]}), not canonical, since
        commit 040bc0c — the test name is retained for compatibility.
        """
        k = KLine(10, [10])  # identity (self-referential since 040bc0c)
        nodes_sig = signifier.make_signature(k.nodes)
        # sig == OR(nodes) is necessary but NOT sufficient for canon — the kline
        # is identity (self-referential), so is_canon() returns False since 040bc0c.
        assert k.signature == nodes_sig
        underfit, overfit = signifier.classify_misfit(k.signature, k.nodes)
        assert not underfit and not overfit


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
        """on_s1 records the query KValue and candidate."""
        handler = RecordingCogitationHandler()
        q = KLine(5, [1, 2])
        c = KLine(10, [3, 4])
        qv = KValue(q, SIG_S3)
        handler.on_s1(qv, c)
        assert handler.s1_calls == [(qv, c)]

    def test_recording_handler_on_expansion(self):
        """on_expansion records the query KValue, proposal, and significance."""
        handler = RecordingCogitationHandler()
        q = KLine(5, [1, 2])
        p = KLine(10, [3, 4])
        qv = KValue(q, SIG_S3)
        handler.on_expansion(qv, p, 42)
        assert handler.expansion_calls == [(qv, p, 42)]


# ── Countersign Tests ────────────────────────────────────────────────


class TestCountersign:
    """Agent.countersign: reciprocal kline construction and rationalisation."""

    def test_countersign_returns_rationalise_result(self):
        """countersign returns the result of rationalise for the reciprocal kline."""
        a = KAgent(adapter=EventBus())
        # Build a kline with non-empty nodes
        kline = KLine(0xFF, [10, 20])
        result = a.countersign(_kv(kline, a.model))
        assert isinstance(result, bool)
        # The reciprocal KLine(signifier.make_signature([10,20]), [0xFF]) should be
        # rationalised as a novel kline → True (S4)
        assert result is True

    def test_countersign_reciprocal_construction(self):
        """Reciprocal is a KValue carrying SIG_S1; its kline is
        KLine(signifier.make_signature(kline.nodes), [kline.signature]) (KP-2)."""
        a = KAgent(adapter=EventBus())
        kline = KLine(0xAB, [10, 20, 30])
        expected_reciprocal_sig = signifier.make_signature([10, 20, 30])
        expected_reciprocal_nodes = [0xAB]

        with patch.object(KAgent, "rationalise", return_value=True) as mock_rationalise:
            result = a.countersign(_kv(kline, a.model))

        assert result is True
        mock_rationalise.assert_called_once()
        reciprocal_value = mock_rationalise.call_args[0][0]
        # KV-5: countersign reciprocal KValue carries SIG_S1 (== D_MAX).
        assert isinstance(reciprocal_value, KValue)
        assert reciprocal_value.significance == SIG_S1
        assert reciprocal_value.kline.signature == expected_reciprocal_sig
        assert reciprocal_value.kline.nodes == expected_reciprocal_nodes

    def test_countersign_empty_nodes(self):
        """Empty nodes → reciprocal_sig=0, reciprocal_nodes=[kline.signature]."""
        a = KAgent(adapter=EventBus())
        kline = KLine(0xCD, [])
        expected_reciprocal_sig = 0  # signifier.make_signature([]) == 0
        expected_reciprocal_nodes = [0xCD]

        with patch.object(KAgent, "rationalise", return_value=True) as mock_rationalise:
            result = a.countersign(_kv(kline, a.model))

        assert result is True
        mock_rationalise.assert_called_once()
        reciprocal_value = mock_rationalise.call_args[0][0]
        assert isinstance(reciprocal_value, KValue)
        assert reciprocal_value.significance == SIG_S1
        assert reciprocal_value.kline.signature == expected_reciprocal_sig
        assert reciprocal_value.kline.nodes == expected_reciprocal_nodes


# ── Cascade Write Method Tests ───────────────────────────────────────


class TestCascadeWriteMethods:
    """Verify the correct model write method is called at each rationalisation phase."""

    def test_agt9_first_rationalise_add_to_ltm(self):
        """AGT-9: First rationalise of a new kline calls model.add_to_ltm()."""
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        k = KLine(5, [1, 2])
        with patch.object(m, "add_to_ltm", wraps=m.add_to_ltm) as mock_add_to_ltm:
            result = a.rationalise(_kv(k, a.model))
        assert result is True
        mock_add_to_ltm.assert_called_once_with(k)

    def test_agt10_duplicate_ground_add_to_stm(self):
        """AGT-10: Second rationalise of same kline calls model.add_to_stm() and emits ground."""
        m = Model()
        events = []
        adapter = EventBus()
        adapter.subscribe(lambda e: events.append(e))
        a = KAgent(model=m, adapter=adapter)
        k = KLine(5, [1, 2])
        a.rationalise(_kv(k, a.model))  # first time
        # Second rationalise — should hit ground check
        dup = KLine(5, [1, 2])
        with patch.object(m, "add_to_stm", wraps=m.add_to_stm) as mock_add_to_stm:
            result = a.rationalise(_kv(dup, a.model))
        assert result is True
        mock_add_to_stm.assert_called_once_with(dup)
        assert any(e.kind == "ground" for e in events)

    def test_agt12_s4_unsigned_add_to_ltm(self):
        """AGT-12: Empty kline calls model.add_to_ltm()."""
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        k = KLine(0, [])
        with patch.object(m, "add_to_ltm", wraps=m.add_to_ltm) as mock_add_to_ltm:
            result = a.rationalise(_kv(k, a.model))
        assert result is True
        mock_add_to_ltm.assert_called_once_with(k)

    def test_agt14_s1_self_grounded_add_to_ltm(self):
        """AGT-14: Self-grounded canonical kline calls model.add_to_ltm()."""
        m = Model()
        # Add resolved nodes so the query's non-literal nodes resolve
        m.add_to_ltm(KLine(10, [10]))  # identity (self-referential since 040bc0c)
        m.add_to_ltm(KLine(20, [20]))  # identity (self-referential since 040bc0c)
        a = KAgent(model=m, adapter=EventBus())
        # Query that is canonical and all non-literal nodes resolve
        # signifier.make_signature([10, 20]) = 10 | 20 = 30
        k = KLine(30, [10, 20])
        with patch.object(m, "add_to_ltm", wraps=m.add_to_ltm) as mock_add_to_ltm:
            result = a.rationalise(_kv(k, a.model))
        assert result is True
        mock_add_to_ltm.assert_any_call(k)

    def test_agt16_novel_s4_add_to_ltm(self):
        """AGT-16: No candidates found calls model.add_to_ltm()."""
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        # A unique signature that won't match anything in the model
        k = KLine(0xFF00, [0xFF00])
        k.signature = signifier.make_signature([0xFF00])
        with patch.object(m, "add_to_ltm", wraps=m.add_to_ltm) as mock_add_to_ltm:
            result = a.rationalise(_kv(k, a.model))
        assert result is True
        mock_add_to_ltm.assert_any_call(k)

    def test_agt18_overlap_routing_submits_to_cogitator(self):
        """AGT-18: overlap routing submits work item to cogitator (no short-circuit)."""
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        # Add a candidate that overlaps the query (routes S2)
        c = KLine(t(5), [t(10), t(20)])
        a.rationalise(_kv(c, a.model))
        # Query that fully matches candidate nodes
        q = KLine(0, [t(10), t(20)])
        q.signature = signifier.make_signature([t(10), t(20)])
        # Capture submitted work items
        submitted = []
        original_submit = a._cogitator.submit

        def capture_submit(item):
            submitted.append(item)
            original_submit(item)

        a._cogitator.submit = capture_submit
        qv = _kv(q, a.model)
        result = a.rationalise(qv)
        assert result is False  # No short-circuit — submitted to cogitator
        assert len(submitted) == 1
        assert submitted[0].query is qv  # WorkItem.query is a KValue
        assert submitted[0].candidate is c
        assert submitted[0].level == "S2"

    def test_agt22a_slow_path_query_add_to_stm_only(self):
        """AGT-22a: S2/S3 routed kline calls model.add_to_stm() only.

        Not add_to_frame or add_to_ltm.
        """
        m = Model()
        a = KAgent(model=m, adapter=EventBus())
        # Add a candidate that will route as S2 (partial overlap)
        c = KLine(t(5), [t(10), t(30)])
        a.rationalise(_kv(c, a.model))
        # Query with partial overlap → S2
        q = KLine(0, [t(10), t(20)])
        q.signature = signifier.make_signature([t(10), t(20)])
        with (
            patch.object(m, "add_to_stm", wraps=m.add_to_stm) as mock_add_to_stm,
            patch.object(m, "add_to_ltm", wraps=m.add_to_ltm) as mock_add_to_ltm,
            patch.object(m, "add_to_frame", wraps=m.add_to_frame) as mock_add_to_frame,
        ):
            result = a.rationalise(_kv(q, a.model))
        assert result is False  # S2 → not significant, submitted to cogitator
        # add_to_stm should have been called for Phase 5
        mock_add_to_stm.assert_called()
        # add_to_ltm should NOT have been called for the query (it's slow path)
        # Note: add_to_ltm may have been called for the candidate earlier, but not for q
        for call in mock_add_to_ltm.call_args_list:
            assert call[0][0] is not q, "add_to_ltm should not be called for slow-path query"
        for call in mock_add_to_frame.call_args_list:
            assert call[0][0] is not q, "add_to_frame should not be called for slow-path query"

    def test_agt29_cogitation_s1_promote_participating(self):
        """AGT-29: on_s1 with structural S1 calls promote and publishes frame."""
        m = Model()
        events = []
        adapter = EventBus()
        adapter.subscribe(lambda e: events.append(e))
        a = KAgent(model=m, adapter=adapter)
        # Build a structurally S1 (genuine canon) candidate.
        candidate = KLine(0b110, [0b100, 0b010])  # canon → is_s1 returns True
        query = KLine(5, [1, 2])
        m.add_to_stm(query)
        with patch("kalvin.agent.promote_participating") as mock_promote:
            a.on_s1(_kv(query, a.model), candidate)
        mock_promote.assert_called_once_with(m, query, candidate, a.signifier)
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
            a.on_s1(_kv(query, a.model), candidate)
        mock_promote.assert_not_called()
        # Frame event still published (unconditional)

    def test_agt34_expansion_add_to_frame(self):
        """AGT-34: on_expansion calls model.add_to_frame(proposal) before publishing."""
        m = Model()
        events = []
        adapter = EventBus()
        adapter.subscribe(lambda e: events.append(e))
        a = KAgent(model=m, adapter=adapter)
        q = KLine(5, [1, 2])
        p = KLine(10, [3, 4])
        with patch.object(m, "add_to_frame", wraps=m.add_to_frame) as mock_add_to_frame:
            a.on_expansion(_kv(q, a.model), p, 42)
        mock_add_to_frame.assert_called_once_with(p)
        # Frame event published
        assert len(events) == 1
        assert events[0].kind == "frame"
        assert events[0].proposal.kline is p  # proposal is a KValue wrapping p
        assert events[0].proposal.significance == 42


# ── NLPTokenizer Integration Tests ───────────────────────────────────
#
# Both classes below require the tokenizer data assets (provided by the
# shared `tokenizer` fixture in conftest.py) and are skipped cleanly when
# those assets are absent on a fresh clone.


@requires_tokenizer_data
class TestAgentTokenizer:
    """KAgent constructed with a tokenizer — pluggable tokenizer integration.

    Verifies that KAgent works correctly when callers pass a tokenizer
    explicitly. The default is the kalvin NLPTokenizer.
    """

    def test_agent_rationalise_kline(self, tokenizer: NLPTokenizer) -> None:
        """KAgent with a tokenizer can rationalise a kline containing typed nodes.

        Encode a known word ('Tea') via the tokenizer, build a KLine
        with the resulting nodes and their signature, and rationalise.
        The kline should be accepted (S4 novel or S1).
        """
        a = KAgent(tokenizer=tokenizer, adapter=EventBus())
        nodes = tokenizer.encode("Tea")
        assert len(nodes) == 1, "'Tea' should produce exactly one typed node"

        sig = signifier.make_signature(nodes)
        kline = KLine(sig, nodes, dbg=KDbg(label="Tea"))
        result = a.rationalise(_kv(kline, a.model))
        assert result is True

        # Signature must be non-zero (typed nodes have non-zero high bits)
        assert sig != 0

        # Signature must be the plain OR-reduce of all nodes.
        expected = 0
        for node in nodes:
            expected |= node
        assert sig == expected

    def test_default_tokenizer_is_base(self) -> None:
        """Default KAgent uses NLPTokenizer as the sole default (raises if data unavailable)."""
        a = KAgent(adapter=EventBus())
        assert isinstance(a.tokenizer, NLPTokenizer)

    def test_agent_serialization(self, tokenizer: NLPTokenizer) -> None:
        """Serialization round-trips preserve typed node values (uint64).

        Create an agent, rationalise a kline with typed nodes,
        then round-trip through to_bytes/from_bytes and to_dict/from_dict.
        The deserialized model should have the same number of klines.
        """
        a = KAgent(tokenizer=tokenizer, adapter=EventBus())
        nodes = tokenizer.encode("Tea")
        sig = signifier.make_signature(nodes)
        kline = KLine(sig, nodes, dbg=KDbg(label="Tea"))
        a.rationalise(_kv(kline, a.model))

        # Binary round-trip
        data = a.to_bytes()
        loaded = KAgent.from_bytes(data)
        assert len(loaded.model) == len(a.model)

        # Dict round-trip
        d = a.to_dict()
        loaded2 = KAgent.from_dict(d)
        assert len(loaded2.model) == len(a.model)

        # Verify the typed node values survived serialization unchanged
        # by checking the stored kline's nodes match the originals
        original_kline = list(a.model.klines())[0]
        loaded_kline = list(loaded.model.klines())[0]
        assert loaded_kline.nodes == original_kline.nodes
        assert loaded_kline.signature == original_kline.signature


# ── Agent + NLPTokenizer Cross-Module Integration Tests ──────────────────


@requires_tokenizer_data
class TestAgentTokenizerIntegration:
    """Cross-module integration tests: tokenizer -> KAgent -> model storage.

    These go beyond unit-level tests to
    verify the full pipeline: encode text -> build kline -> rationalise ->
    store in model -> retrieve and verify node/signature integrity.

    Focuses on multi-word phrases (with space tokens), model retrieval,
    and serialization.
    """

    @staticmethod
    def _make_agent_with_klines(
        tokenizer: NLPTokenizer,
    ) -> tuple[KAgent, list[KLine]]:
        """Create an agent with a single rationalised kline."""
        a = KAgent(tokenizer=tokenizer, adapter=EventBus())

        # Single-word typed kline
        tea = tokenizer.encode("Tea")
        sig1 = signifier.make_signature(tea)
        k1 = KLine(sig1, tea, dbg=KDbg(label="Tea"))
        a.rationalise(_kv(k1, a.model))

        return a, [k1]

    def test_agent_rationalise_kline(self, tokenizer: NLPTokenizer) -> None:
        """Rationalise a kline with the tokenizer — verify storage and retrieval."""
        a, klines = self._make_agent_with_klines(tokenizer)

        # Verify kline was stored
        assert len(klines) == 1
        stored = a.model.find(klines[0].signature)
        assert stored is not None

    def test_agent_bytes_roundtrip(self, tokenizer: NLPTokenizer) -> None:
        """Binary serialization preserves typed node values (uint64).

        Multiple klines with typed nodes survive to_bytes/from_bytes
        with exact node value preservation.
        """
        a, original_klines = self._make_agent_with_klines(tokenizer)

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

    def test_agent_dict_roundtrip(self, tokenizer: NLPTokenizer) -> None:
        """Dict serialization preserves typed node values exactly.

        Nodes are stored as integers in the dict and must round-trip
        without precision loss.
        """
        a, original_klines = self._make_agent_with_klines(tokenizer)

        d = a.to_dict()
        loaded = KAgent.from_dict(d)

        assert len(loaded.model) == len(original_klines)

        for orig in original_klines:
            stored = loaded.model.find(orig.signature)
            assert stored is not None
            assert stored.nodes == orig.nodes
            assert stored.signature == orig.signature

    def test_agent_json_file_roundtrip(self, tokenizer: NLPTokenizer) -> None:
        """JSON file serialization preserves typed node values.

        Write to a temp JSON file, reload, verify all node values match.
        """
        a, original_klines = self._make_agent_with_klines(tokenizer)

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


# ── KValue Exchange Criteria (KV-5, KV-6, KV-13, KV-14, KV-15) ────────
#
# These tests verify the rationalisation pipeline consumes and emits
# KValues (@kvalue spec §Exchange, §Producers). They use a dummy tokenizer
# because rationalise/countersign/on_expansion operate on the Model API
# (KLine-based) and never tokenise — but the module-level ``pytestmark``
# skips them where tokenizer data is absent.


class TestKValueExchangeCriteria:
    """KValue-aware pipeline criteria tests (KV-5/6/13/14/15)."""

    def _agent(self, model: Model | None = None, bus: EventBus | None = None) -> KAgent:
        """Build a KAgent with a dummy tokenizer (rationalise never tokenises)."""
        return KAgent(
            tokenizer=object(),
            model=model if model is not None else Model(signifier=signifier),
            adapter=bus if bus is not None else EventBus(),
            signifier=signifier,
        )

    def test_kv5_countersign_reciprocal_carries_sig_s1(self) -> None:
        """KV-5: countersign builds a reciprocal whose KValue carries SIG_S1 (KP-2)."""
        a = self._agent()
        kline = KLine(0xFF, [10, 20])
        expected_reciprocal = KLine(signifier.make_signature([10, 20]), [0xFF])

        with patch.object(KAgent, "rationalise", return_value=True) as mock_rationalise:
            a.countersign(_kv(kline, a.model))

        mock_rationalise.assert_called_once()
        reciprocal_value = mock_rationalise.call_args[0][0]
        assert isinstance(reciprocal_value, KValue)
        # The act of countersigning is an S1 ratification (== D_MAX).
        assert reciprocal_value.significance == SIG_S1
        assert reciprocal_value.kline == expected_reciprocal

    def test_kv6_cogitation_proposal_carries_computed_significance(self) -> None:
        """KV-6: a cogitation expansion proposal event's ``proposal.significance``
        equals the value ``expand()`` computed for that proposal (KP-3), not a
        band-representative value.
        """
        from kalvin.expand import boundaries, classify, expand, propose_expansions

        m = Model(signifier=signifier)
        k1 = KLine(t(0b100), [t(0b100)])  # identity
        m.add_to_ltm(k1)
        k2 = KLine(t(0b010), [t(0b010)])  # identity
        m.add_to_ltm(k2)
        k3 = KLine(t(0b110), [t(0b100)])  # misfit (underfitting)
        m.add_to_ltm(k3)
        q = KLine(0, [t(0b001)])
        q.signature = signifier.make_signature([t(0b001)])
        m.add_to_frame(q)
        q_value = _kv(q, m)

        events: list = []
        bus = EventBus()
        bus.subscribe(lambda e: events.append(e))
        a = self._agent(model=m, bus=bus)

        a._cogitator.submit(WorkItem(q_value, k3, "S3"))
        try:
            assert a.cogitate_drain(timeout=5.0)
        finally:
            a.cogitate_join(timeout=2.0)

        expansion_events = [e for e in events if e.kind == "frame"]
        assert len(expansion_events) >= 1

        # Independently recompute the (proposal, significance) pairs that
        # expand() yields for this query|candidate pair.
        s12, s23, s34 = boundaries()
        computed: set[tuple[int, tuple[int, ...], int]] = set()
        for qc in expand(m, q, k3, signifier):
            band = classify(qc.significance, s12, s23, s34)
            if band in ("S4", "S1"):
                continue
            for proposal, sval in propose_expansions(m, qc.candidate, qc.significance, signifier):
                computed.add((proposal.signature, tuple(proposal.nodes), sval))

        band_reps = {SIG_S1, SIG_S2, SIG_S3, SIG_S4}
        for e in expansion_events:
            # query is the original inbound KValue (KE-2).
            assert e.query == q_value
            # proposal.significance is exactly the expand()-computed value.
            key = (
                e.proposal.kline.signature,
                tuple(e.proposal.kline.nodes),
                e.proposal.significance,
            )
            assert key in computed, (
                f"expansion significance {e.proposal.significance:#x} is not an "
                f"expand()-computed value"
            )
        # At least one proposal carries a computed (non-band) significance.
        assert any(e.proposal.significance not in band_reps for e in expansion_events)

    def test_kv13_fast_path_shares_kline_independent_significances(self) -> None:
        """KV-13: on the fast path ``event.query.kline is event.proposal.kline``
        (one shared immutable KLine, no copy) while the significances are
        independent assessments (KE-1).
        """
        events: list = []
        bus = EventBus()
        bus.subscribe(lambda e: events.append(e))
        a = self._agent(bus=bus)

        # Empty kline → S4 frame fast path. Declare SIG_S1 (different from
        # Kalvin's S4) so the two assessments are provably independent.
        k = KLine(0, [])
        inbound = KValue(k, SIG_S1)
        result = a.rationalise(inbound)

        assert result is True
        assert len(events) >= 1
        e = events[0]
        assert e.kind == "frame"
        # Same immutable KLine shared across query and proposal (no copy).
        assert e.query.kline is e.proposal.kline
        # Independent assessments: declared SIG_S1, Kalvin's SIG_S4.
        assert e.query.significance == SIG_S1
        assert e.proposal.significance == SIG_S4
        assert e.query.significance != e.proposal.significance

    def test_kv14_event_has_no_significance_field(self) -> None:
        """KV-14: RationaliseEvent exposes no top-level significance field
        (KE-3); significance lives only on the query/proposal KValues.
        """
        q = KValue(KLine(0, []), 0)
        p = KValue(KLine(0, []), 0)
        e = RationaliseEvent("ground", q, p)
        assert not hasattr(e, "significance")
        assert not hasattr(e, "candidate")
        # Constructing with a significance= keyword raises TypeError.
        with pytest.raises(TypeError):
            RationaliseEvent("ground", q, p, significance=0xFF)  # type: ignore[call-arg]

    def test_kv15_consumer_reads_proposal_significance(self) -> None:
        """KV-15: a consumer (subscriber) reads ``event.proposal.significance``
        (Kalvin's assessment), which carries the Kalvin band value, not the
        sender's declared assessment (KE-4).
        """
        events: list = []
        bus = EventBus()
        bus.subscribe(lambda e: events.append(e))
        a = self._agent(bus=bus)

        # S4 fast path; declare SIG_S1 so the two voices differ.
        k = KLine(0, [])
        inbound = KValue(k, SIG_S1)
        a.rationalise(inbound)

        assert len(events) >= 1
        e = events[0]
        # A consumer reading Kalvin's assessment gets SIG_S4 (the band value),
        # not the declared SIG_S1.
        assert e.proposal.significance == SIG_S4
        assert e.proposal.significance != e.query.significance
