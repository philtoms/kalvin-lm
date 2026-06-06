"""Tests for countersign resolution spec (specs/countersign-resolution.md).

Covers: pre-registration, ground-check STM exclusion, self-filter,
sig_level propagation, undersign=connotate identity, and countersign
pair resolution.
"""

import pytest
from kalvin.agent import KAgent
from kalvin.events import EventBus, RationaliseEvent
from kalvin.kline import KLine
from kalvin.model import Model
from kalvin.signature import make_signature
from kscript import compile_source


class TestCountersignPairResolution:
    """CR-1: Countersign pair both resolve S1 via is_countersigned()."""

    def test_countersign_pair_both_resolve_s1(self):
        bus = EventBus()
        a = KAgent(adapter=bus)

        # Add identities
        a.rationalise(KLine(0x2000, []))  # M
        a.rationalise(KLine(0x100, []))   # H

        entries = compile_source("M == H", dev=True)
        assert len(entries) == 2

        # Pre-register (simulates adapter behaviour)
        for e in entries:
            a.model.add_stm(e)

        events = []
        orig = bus.on_event
        def cap(ev):
            events.append(ev)
            orig(ev)
        bus.on_event = cap

        for e in entries:
            result = a.rationalise(e)
            assert result is True

        # Both should produce frame events with S1 significance
        frame_events = [e for e in events if e.kind == "frame"]
        assert len(frame_events) == 2
        for ev in frame_events:
            assert ev.significance > 0  # S1 range


class TestGroundCheckExcludesSTM:
    """CR-3, CR-4: Model.grounded() checks Frame/LTM only."""

    def test_grounded_returns_false_for_stm_only(self):
        m = Model()
        kl = KLine(0xFF, [1, 2])
        m.add_stm(kl)
        assert m.exists(kl) is True
        assert m.grounded(kl) is False

    def test_grounded_returns_true_for_frame_entry(self):
        m = Model()
        kl = KLine(0xFF, [1, 2])
        m.add_frame(kl)
        assert m.grounded(kl) is True

    def test_grounded_returns_true_for_ltm_entry(self):
        m = Model()
        kl = KLine(0xFF, [1, 2])
        m.add_ltm(kl)
        assert m.grounded(kl) is True


class TestSelfFilterInCandidates:
    """CR-5: rationalise() excludes self from candidates."""

    def test_rationalise_excludes_self_from_candidates(self):
        bus = EventBus()
        a = KAgent(adapter=bus)

        # Add a candidate that partially overlaps with query signature
        candidate = KLine(5, [10, 30])
        a.rationalise(candidate)

        # Query overlaps on [10] but not [20] -> should be S2, not S1
        q = KLine(0, [10, 20])
        q.signature = make_signature([10, 20])
        result = a.rationalise(q)
        # S2 should return False (slow path) even though q is in STM
        assert result is False


class TestSigLevelPropagation:
    """CR-6: sig_level is set on CompiledEntry matching operator type."""

    def test_sig_level_set_on_compiled_entry(self):
        entries = compile_source("M == H", dev=True)
        for e in entries:
            assert e.sig_level == "S1"

        entries = compile_source("M > H", dev=True)
        for e in entries:
            if e.nodes:  # skip unsigned identities
                assert e.sig_level == "S3"

        entries = compile_source("M => H", dev=True)
        for e in entries:
            if e.nodes and not isinstance(e.nodes, list):
                assert e.sig_level == "S2"

    def test_sig_level_none_by_default(self):
        kl = KLine(0xFF, [1])
        assert kl.sig_level is None


class TestUndersignIsConnotateReversed:
    """CR-7: Undersign gets no special fast path. CR-8: connotate through slow path."""

    def test_undersign_no_special_fast_path(self):
        """Undersign {M: S} should NOT shortcut to S1 in rationalise."""
        bus = EventBus()
        a = KAgent(adapter=bus)

        # Add identities
        a.rationalise(KLine(0x2000, []))  # M
        a.rationalise(KLine(0x80000, []))  # S

        # Compile undersign: S = M -> {M: S}
        entries = compile_source("S = M", dev=True)
        # Filter to just the undersign entry (not the unsigned identity)
        undersign = [e for e in entries if e.nodes]
        assert len(undersign) == 1
        e = undersign[0]
        assert e.sig_level == "S1"

        # Pre-register
        a.model.add_stm(e)

        result = a.rationalise(e)
        # Without countersigner or canonical structure, undersign should
        # NOT resolve as True (S1 fast path) — it goes to slow path
        assert result is False

    def test_connotate_goes_through_slow_path(self):
        """Connotate {A: D} should go through candidate retrieval -> slow path."""
        bus = EventBus()
        a = KAgent(adapter=bus)

        # Add identity A only (D is unknown)
        a.rationalise(KLine(0x2, []))  # A

        entries = compile_source("A > D", dev=True)
        connotate = [e for e in entries if e.nodes]
        assert len(connotate) == 1
        e = connotate[0]
        assert e.sig_level == "S3"

        a.model.add_stm(e)
        result = a.rationalise(e)
        # No candidates for A signature, so it should be novel (S4 -> True)
        # OR if candidates exist, S3 -> False (slow path)
        # Since only {A: None} exists, where(A) returns it
        # _route({A: [D]}, {A: []}) -> S3 -> False
        assert result is False
