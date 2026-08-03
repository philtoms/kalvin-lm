"""Tests for expand module — direct function calls, not Model forwarding."""

import pytest

from kalvin.expand import (
    D_MAX,
    DEFAULT_AGGREGATOR,
    DEFAULT_S2_S3_BOUNDARY,
    MASK64,
    SIG8_MAX,
    SIG8_MIN,
    SIG_MASK,
    SIG_S1,
    SIG_S2,
    SIG_S3,
    SIG_S4,
    BandLayout,
    band_significance,
    edge_hops,
    expand,
    is_canon,
    is_countersigned,
    is_s1,
    promote_participating,
    propose_expansions,
    structural_significance,
)
from kalvin.kline import KLine
from kalvin.model import Model
from kalvin.nlp_tokenizer import COMPOUND_TOKEN
from kalvin.signifier import NLPSignifier

signifier = NLPSignifier()


def make_model(stm_bound: int = 256) -> Model:
    return Model(stm_bound=stm_bound)


def t(bits: int) -> int:
    """Place sig-word bits in the upper 32 bits of a uint64.

    signifies() masks off the lower (BPE) 32 bits, so node/signature values
    that must participate in significance matching are shifted up here.
    """
    return bits << 32


class TestBandRepresentativeConstants:
    """Verify the four band-representative constants match @model spec."""

    def test_constants_are_spec_values(self):
        """SIG_S1..SIG_S4 are D_MAX, D_MAX-1, D_MAX-101, 0 (@model spec)."""
        assert SIG_S1 == D_MAX
        assert SIG_S2 == D_MAX - 1
        assert SIG_S3 == D_MAX - 101
        assert SIG_S4 == 0

    def test_constants_are_inverted_distances(self):
        """Each representative equals (~distance) & MASK64 for its distance.

        Confirms significance inversion (@model spec §Significance Inversion):
        distance 0 → SIG_S1, distance 1 → SIG_S2, distance 101 → SIG_S3.
        """
        assert (~0) & MASK64 == SIG_S1
        assert (~1) & MASK64 == SIG_S2
        assert (~101) & MASK64 == SIG_S3

    def test_strict_ordering(self):
        """Unsigned ordering holds: SIG_S1 > SIG_S2 > SIG_S3 > SIG_S4."""
        assert SIG_S1 > SIG_S2 > SIG_S3 > SIG_S4

    def test_all_valid_uint64(self):
        """All band-representative values are non-negative uint64."""
        for val in (SIG_S1, SIG_S2, SIG_S3, SIG_S4):
            assert 0 <= val <= MASK64


class TestBandSignificance:
    """Verify band_significance() maps structural relationships to band constants (KP-1)."""

    def test_countersigned_is_s1(self):
        """COUNTERSIGNS → SIG_S1."""
        assert band_significance("COUNTERSIGNS") == SIG_S1

    def test_canonized_is_s2(self):
        """CANONIZES → SIG_S2."""
        assert band_significance("CANONIZES") == SIG_S2

    def test_connoted_is_s3(self):
        """CONNOTES → SIG_S3."""
        assert band_significance("CONNOTES") == SIG_S3

    def test_denoted_is_s3(self):
        """DENOTES → SIG_S3 (same band as CONNOTES)."""
        assert band_significance("DENOTES") == SIG_S3

    def test_identity_is_s4(self):
        """UNKNOWN → SIG_S4."""
        assert band_significance("UNKNOWN") == SIG_S4

    def test_unknown_op_defaults_to_s4(self):
        """Unknown op → SIG_S4 (the safe floor)."""
        assert band_significance("NOPE") == SIG_S4
        assert band_significance("") == SIG_S4


class TestIsCanon:
    def test_canon_match(self):
        """sig == signature_of(nodes), non-self-referential → canonical."""
        # sig = 0b110 = OR(0b100, 0b010); neither node equals the signature.
        k = KLine(0b110, [0b100, 0b010])
        assert is_canon(k, signifier) is True

    def test_canon_mismatch(self):
        """sig != signature_of(nodes) → non-canonical."""
        k = KLine(5, [10])  # signature_of([10]) = 10 ≠ 5
        assert is_canon(k, signifier) is False

    def test_self_referential_is_not_canon(self):
        """{S: [S]} is identity, not canon — overrules canon classification."""
        k = KLine(10, [10])  # signature_of([10]) = 10, but self-referential
        assert is_canon(k, signifier) is False


class TestEdgeHops:
    def test_edge_hops_unresolvable(self):
        """Node that doesn't resolve → empty generator."""
        m = make_model()
        assert list(edge_hops(m, 99, signifier)) == []

    def test_edge_hops_canonical(self):
        """Node that resolves to canonical → empty generator."""
        m = make_model()
        # Genuine canon: sig 0b110 = OR(0b100, 0b010).
        m.add_to_frame(KLine(0b110, [0b100, 0b010]))
        assert list(edge_hops(m, 0b110, signifier)) == []

    def test_edge_hops_identity(self):
        """Node that resolves to identity (incl. self-referential) → empty."""
        m = make_model()
        m.add_to_frame(KLine(10, []))  # empty-nodes identity
        assert list(edge_hops(m, 10, signifier)) == []
        m.add_to_frame(KLine(30, [30]))  # self-referential identity
        assert list(edge_hops(m, 30, signifier)) == []

    def test_edge_hops_chain(self):
        """Non-canonical chain terminates at a genuine canon."""
        m = make_model()
        m.add_to_frame(KLine(0b110, [0b100, 0b010]))  # canonical terminator
        m.add_to_frame(KLine(20, [0b110]))  # non-canon: sig=20, make_sig=[0b110]=0b110
        m.add_to_frame(KLine(10, [20]))  # non-canon: sig=10, make_sig([20])=20
        m.add_to_frame(KLine(5, [10]))  # non-canon: sig=5,  make_sig([10])=10
        assert list(edge_hops(m, 5, signifier)) == [(1, 10), (2, 20), (3, 0b110)]
        assert list(edge_hops(m, 10, signifier)) == [(1, 20), (2, 0b110)]
        assert list(edge_hops(m, 20, signifier)) == [(1, 0b110)]
        assert list(edge_hops(m, 0b110, signifier)) == []  # canonical
        assert list(edge_hops(m, 99, signifier)) == []  # unresolvable

    def test_edge_hops_cycle_detection_er1(self):
        """ER-1: Countersigned pair produces bounded hops, not MAX_HOP."""
        m = make_model()
        # {A: [B]} ↔ {B: [A]} — mutual non-canonical resolution
        m.add_to_frame(KLine(5, [10]))  # sig=5, make_sig([10])=10
        m.add_to_frame(KLine(10, [5]))  # sig=10, make_sig([5])=5
        hops = list(edge_hops(m, 5, signifier))
        # Without cycle detection: 100 hops alternating 10,5,10,5...
        # With cycle detection: at most 2 hops before revisiting sig
        assert len(hops) <= 3
        assert hops == [(1, 10), (2, 5)]

    def test_edge_hops_identity_kline_er2(self):
        """ER-2: Identity kline {A: []} yields zero hops."""
        m = make_model()
        # Identity kline: sig > 0, nodes = []
        # signature_of([]) = 0, so it's not canonical (sig ≠ 0)
        m.add_to_frame(KLine(42, []))  # identity, not canonical
        hops = list(edge_hops(m, 42, signifier))
        # Without guard: yields (1, 0) which is a dead end
        # With guard: yields nothing (breaks on sig == 0)
        assert hops == []


class TestExpand:
    def test_expand_self_no_model(self):
        """Self-comparison: 3 matched-but-ungrounded nodes -> decay(1) each."""
        m = make_model()
        k = KLine(10, [10, 20, 30])
        results = list(expand(m, k, k, signifier))
        # All 3 nodes match; none grounded -> 3 x decay(1) (Q17a).
        assert len(results) == 1
        assert results[-1].significance == DEFAULT_AGGREGATOR.compose_terminal(
            [DEFAULT_AGGREGATOR.decay(1)] * 3
        )

    def test_expand_no_resolution(self):
        """1 matched-ungrounded + 4 unresolvable -> mostly unaccounted."""
        m = make_model()
        q = KLine(5, [1, 2, 3])
        c = KLine(6, [1, 4, 5])
        # matched: {1} (ungrounded -> decay(1)); mismatched {2,3,4,5} unresolvable.
        results = list(expand(m, q, c, signifier))
        assert results[-1].significance == DEFAULT_AGGREGATOR.compose_terminal(
            [DEFAULT_AGGREGATOR.decay(1), 0.0, 0.0, 0.0, 0.0]
        )

    def test_expand_with_grounding(self):
        """1 grounded match + 2 unresolvable -> grounded contributes 1.0."""
        m = make_model()
        m.add_to_frame(KLine(0b110, [0b100, 0b010]))  # genuine canon (S1)
        q = KLine(5, [0b110, 2])
        c = KLine(6, [0b110, 3])
        # Slots: [1.0 (grounded), 0.0, 0.0].
        results = list(expand(m, q, c, signifier))
        assert results[-1].significance == DEFAULT_AGGREGATOR.compose_terminal(
            [1.0, 0.0, 0.0]
        )

    def test_expand_hop_reaches_opposing_mismatch(self):
        """Mismatched node whose chain reaches the opposing mismatch set."""
        m = make_model()
        m.add_to_frame(KLine(t(0b110), [t(0b100), t(0b010)]))  # genuine canon — chain terminator
        m.add_to_frame(KLine(t(20), [t(0b110)]))  # non-canon
        m.add_to_frame(KLine(t(10), [t(20)]))  # non-canon
        m.add_to_frame(KLine(t(5), [t(10)]))  # non-canon

        q = KLine(100, [t(5), t(2)])  # mismatched_q: {5, 2}
        c = KLine(200, [t(10), t(3)])  # mismatched_c: {10, 3}
        results = list(expand(m, q, c, signifier))
        # Q18: cardinality unchanged from the old scheme.
        assert len(results) == 6
        # Terminal slots: q-5 resolves to c-10 at 1 hop (decay(1));
        # q-2, c-3 unresolvable (0.0); c-10 signifies at 2 hops (decay(2)).
        assert results[-1].significance == DEFAULT_AGGREGATOR.compose_terminal(
            [DEFAULT_AGGREGATOR.decay(1), 0.0, DEFAULT_AGGREGATOR.decay(2), 0.0]
        )

        # S2 signifies side-candidates reaching sig 0b110 at 2 hops carry the
        # byte for [decay(2)] (Q18 D: side-candidate, decay-derived).
        sig_two_hops = DEFAULT_AGGREGATOR.compose_terminal([DEFAULT_AGGREGATOR.decay(2)])
        signifies_two_hops = [r for r in results[:-1] if r.significance == sig_two_hops]
        assert len(signifies_two_hops) >= 1

    def test_expand_bidirectional_hop_match(self):
        """Both query and candidate mismatched nodes reach opposing sets."""
        m = make_model()
        m.add_to_frame(KLine(10, [10]))  # identity
        m.add_to_frame(KLine(5, [10]))  # non-canon
        m.add_to_frame(KLine(30, [30]))  # identity
        m.add_to_frame(KLine(20, [30]))  # non-canon

        q = KLine(100, [5, 20])  # mismatched_q: {5, 20}
        c = KLine(200, [10, 30])  # mismatched_c: {10, 30}
        results = list(expand(m, q, c, signifier))
        # Q18: 2 recursive connotations + terminal.
        assert len(results) == 3
        # Resolution is directional: q-nodes 5,20 resolve to c-nodes 10,30 at
        # 1 hop (decay(1) each); c-nodes 10,30 are identity terminals, so
        # edge_hops from their side yields nothing -> 0.0 each.
        assert results[-1].significance == DEFAULT_AGGREGATOR.compose_terminal(
            [DEFAULT_AGGREGATOR.decay(1), DEFAULT_AGGREGATOR.decay(1), 0.0, 0.0]
        )

    def test_expand_all_matched_grounded(self):
        """All nodes match and all resolve to S1 → no penalty → max significance."""
        m = make_model()
        m.add_to_frame(KLine(0b110, [0b100, 0b010]))  # genuine canon, node 0b110 is S1
        m.add_to_frame(KLine(0b1100, [0b1000, 0b0100]))  # genuine canon, node 0b1100 is S1
        q = KLine(5, [0b110, 0b1100])
        c = KLine(6, [0b110, 0b1100])
        results = list(expand(m, q, c, signifier))
        # Fully accounted (2 x 1.0) -> saturates to SIG8_MAX (Q9).
        assert results[-1].significance == SIG8_MAX

    def test_expand_in_valid_byte_range(self):
        """Significance is always a valid byte in [0x00, 0xFF]."""
        m = make_model()
        q = KLine(5, [1])
        c = KLine(6, list(range(1000)))
        results = list(expand(m, q, c, signifier))
        assert 0 <= (results[-1].significance & SIG_MASK) <= SIG8_MAX

    def test_expand_range_s2(self):
        """A graded result is a valid byte."""
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(1, [1, 3, 4])
        results = list(expand(m, q, c, signifier))
        assert 0 <= (results[-1].significance & SIG_MASK) <= SIG8_MAX

    def test_expand_topology_driven(self):
        """Significance is topology-driven (replaces old level-independence test)."""
        m = make_model()
        q = KLine(5, [1, 2])
        c = KLine(100, [3, 4])
        sig = list(expand(m, q, c, signifier))[-1].significance
        # Both mismatched nodes unresolvable -> 0.0 slots -> SIG8_MIN.
        assert sig == SIG8_MIN

    def test_expand_significance_ordering(self):
        """Verify significance ordering: closer match → higher significance."""
        m = make_model()
        m.add_to_frame(KLine(10, [10]))  # identity — chain terminator
        m.add_to_frame(KLine(5, [10]))  # non-canon

        q = KLine(100, [5, 2])  # mismatched_q: {5, 2}
        c = KLine(200, [10, 3])  # mismatched_c: {10, 3}
        results = list(expand(m, q, c, signifier))
        # q-5 resolves to c-10 at 1 hop (decay(1)); q-2, c-10 (identity terminal),
        # c-3 all unresolvable (0.0). Four slots, mean dominated by zeros.
        assert results[-1].significance == DEFAULT_AGGREGATOR.compose_terminal(
            [DEFAULT_AGGREGATOR.decay(1), 0.0, 0.0, 0.0]
        )
        assert results[-1].significance > SIG8_MIN

    def test_expand_connotation_bridging(self):
        """Connotation bridging: indirect path through intermediate signature.

        S3 connotation hops use linear distance S2_S3_DISTANCE + hop_count,
        ensuring S3 distances exceed S2 distances with linear (not quadratic)
        growth.

        Uses powers-of-2 nodes (4, 2, 8) so that signifies() returns False,
        ensuring the S3 connotation path is exercised (signifies short-circuits
        before S3 when signatures share bits).
        """
        m = make_model()
        m.add_to_frame(KLine(8, [8]))  # identity (self-referential) — dead end
        m.add_to_frame(KLine(4, [8]))  # non-canon: edge_hops(4, signifier) = [(1, 8)]
        m.add_to_frame(KLine(2, [8]))  # non-canon: edge_hops(2, signifier) = [(1, 8)]

        q = KLine(100, [4])  # mismatched_q: {4}
        c = KLine(200, [2])  # mismatched_c: {2}

        # signifies(4,8)=False, signifies(2,8)=False -> S3 path exercised.
        # s3_connotations[8] = 1 (from q-4); c-2 bridges at s3_hop = 1+1 = 2.
        results = list(expand(m, q, c, signifier))
        # Q18 (E): connotation recurses, no side-candidate here. 1 nested
        # terminal + top-level terminal.
        assert len(results) == 2

        # Nested terminal (recursive expand(2, 8)): node 8 matched-ungrounded
        # (identity is not S1) -> [decay(1)].
        nested = results[0]
        assert nested.query.signature == 2
        assert nested.candidate.signature == 8
        assert nested.significance == DEFAULT_AGGREGATOR.compose_terminal(
            [DEFAULT_AGGREGATOR.decay(1)]
        )

        # Top-level terminal: q-4 does not resolve directly (0.0); c-2 bridges
        # via s3_hop=2 -> decay(2). Two slots: [0.0, decay(2)].
        terminal = results[1]
        assert terminal.query is q
        assert terminal.candidate is c
        assert terminal.significance == DEFAULT_AGGREGATOR.compose_terminal(
            [0.0, DEFAULT_AGGREGATOR.decay(2)]
        )

    def test_expand_signifies_cogitation(self):
        """S2 signifies loose match yields additional QueryCandidates.

        When a mismatched node's edge hop reaches a signature that shares bits
        (signifies) but isn't an exact match, a QueryCandidate is yielded for
        cogitation. The mismatched node still contributes MAX_HOP to the
        terminal distance (signifies doesn't resolve the mismatch).
        """
        m = make_model()
        m.add_to_frame(KLine(t(30), [t(30)]))  # identity — chain terminator
        m.add_to_frame(KLine(t(20), [t(30)]))  # non-canon
        m.add_to_frame(KLine(t(10), [t(20)]))  # non-canon
        m.add_to_frame(KLine(t(5), [t(10)]))  # non-canon

        q = KLine(100, [t(5)])  # mismatched_q: {5}
        c = KLine(200, [t(10)])  # mismatched_c: {10}

        # q-node 5: edge_hops(5, signifier) = [(1,10), (2,20), (3,30)]
        #   hop 1: match_sig=10 IS in mismatched_c → exact match, expand
        #   → expand(5, 10, 1, signifier) produces its own results
        #
        # c-node 10: edge_hops(10, signifier) = [(1,20), (2,30)]
        #   hop 1: match_sig=20 not in mismatched_q, signifies(10,20)=False
        #   hop 2: match_sig=30 not in mismatched_q, signifies(10,30)=True
        #   → yields QueryCandidate(find(10), find(30), (~2) & MASK64)
        #   hop_distance stays MAX_HOP
        #
        # Terminal: distance = 1 (exact match hop) + MAX_HOP (c-node unresolved)

        results = list(expand(m, q, c, signifier))

        # Find the signifies side-candidate from c-10 -> 30 at 2 hops.
        signifies_candidates = [
            r for r in results if r.query.signature == t(10) and r.candidate.signature == t(30)
        ]
        assert len(signifies_candidates) == 1
        sig_cand = signifies_candidates[0]
        # Q18 D: side-candidate carries the byte for [decay(hops)].
        assert sig_cand.significance == DEFAULT_AGGREGATOR.compose_terminal(
            [DEFAULT_AGGREGATOR.decay(2)]
        )

    def test_expand_signifies_before_s3(self):
        """Signifies (S2) takes precedence over s3_connotations (S3).

        When signifies matches, s3_connotations is not populated for that
        signature, preventing S3 connotation bridging for the same hop.
        """
        m = make_model()
        m.add_to_frame(KLine(t(0b11100), [t(0b11100)]))  # identity (self-referential) (28)
        m.add_to_frame(KLine(t(0b10100), [t(0b11100)]))  # non-canon: sig=20, make_sig=28
        m.add_to_frame(KLine(t(0b01100), [t(0b11100)]))  # non-canon: sig=12, make_sig=28

        q = KLine(100, [t(0b10100)])  # mismatched_q: {20}
        c = KLine(200, [t(0b01100)])  # mismatched_c: {12}

        # q-node 20: edge_hops(20, signifier) = [(1, 28)]
        #   28 not in mismatched_c, signifies(20, 28) = True (20 & 28 = 20)
        #   → S2 signifies candidate, break (s3_connotations NOT populated)
        #
        # c-node 12: edge_hops(12, signifier) = [(1, 28)]
        #   28 not in mismatched_q, signifies(12, 28) = True (12 & 28 = 12)
        #   → S2 signifies candidate, break

        results = list(expand(m, q, c, signifier))
        # Q18: 2 signifies side-candidates + terminal.
        assert len(results) == 3

        # Both signifies candidates reach sig 28 at 1 hop -> byte for [decay(1)].
        sig_one_hop = DEFAULT_AGGREGATOR.compose_terminal([DEFAULT_AGGREGATOR.decay(1)])
        assert results[0].candidate.signature == t(0b11100)
        assert results[0].significance == sig_one_hop
        assert results[1].candidate.signature == t(0b11100)
        assert results[1].significance == sig_one_hop

        # Terminal: both mismatched nodes signify at 1 hop -> slots [decay(1), decay(1)].
        assert results[-1].significance == DEFAULT_AGGREGATOR.compose_terminal(
            [DEFAULT_AGGREGATOR.decay(1), DEFAULT_AGGREGATOR.decay(1)]
        )

    def test_expand_significance_in_range(self):
        """Significance is always in valid uint64 range."""
        m = make_model()
        q = KLine(5, list(range(1000)))
        c = KLine(6, list(range(1000, 2000)))
        results = list(expand(m, q, c, signifier))
        sig = results[-1].significance
        assert 0 <= (sig & SIG_MASK) <= SIG8_MAX

    def test_expand_no_crash_on_unresolvable_match_sig_er6(self):
        """ER-6: expand() does not crash when edge_hops yields an unresolvable sig."""
        m = make_model()
        # Build a scenario where edge_hops produces match_sig=0 from identity kline
        # Identity kline {42: []} → make_sig([]) = 0, which doesn't resolve
        m.add_to_ltm(KLine(42, []))  # identity
        # Query and candidate that trigger the mismatched-node path
        q = KLine(42 | 10, [42, 10])  # sig includes 42, nodes include 42
        c = KLine(42 | 10, [10])  # partial overlap on 10, mismatched_c = {}
        m.add_to_ltm(c)
        # expand should complete without ValueError
        results = list(expand(m, q, c, signifier))
        assert len(results) >= 1  # at least terminal yield

    def test_expand_countersign_cycle_no_crash_er7(self):
        """ER-7: S2 scenario with countersigned klines completes without exception."""
        m = make_model()
        # Uppercase to mirror KLine protocol signatures; noqa to avoid clash with model `m`.
        M = 0x2000  # noqa: N806
        H = 0x100  # noqa: N806
        A = 0x2  # noqa: N806
        MH = M | H  # noqa: N806
        # Identities
        m.add_to_ltm(KLine(M, []))
        m.add_to_ltm(KLine(H, []))
        m.add_to_ltm(KLine(A, []))
        # Countersign pair
        m.add_to_ltm(KLine(M, [H]))
        m.add_to_ltm(KLine(H, [M]))
        # MTS canonical
        m.add_to_ltm(KLine(MH, [M, H]))
        # The S2 query
        query = KLine(MH, [H, A])
        candidate = KLine(M, [H])  # misfit, routes S2
        # Must not raise ValueError
        results = list(expand(m, query, candidate, signifier))
        assert len(results) >= 1


# ── Structural Grounding Tests ───────────────────────────────────────


class TestIsS1:
    def test_canonical_kline(self):
        """Genuine canon (sig == signature_of(nodes), non-self-referential) → S1."""
        m = Model()
        # sig 0b110 = OR(0b100, 0b010); a genuine canon.
        k = KLine(0b110, [0b100, 0b010])
        assert is_s1(m, k, signifier) is True

    def test_self_referential_is_not_s1(self):
        """{S: [S]} is identity, not canon → not S1 by canon."""
        m = Model()
        k = KLine(10, [10])
        assert is_s1(m, k, signifier) is False

    def test_countersigned_in_model(self):
        """Two klines with mutual node references → S1."""
        m = Model()
        a = KLine(5, [10])
        b = KLine(10, [5])
        m.add_to_frame(a)
        m.add_to_frame(b)
        # a is countersigned: a.nodes has 10, model.find(10)=b, b.nodes has 5=a.signature
        assert is_s1(m, a, signifier) is True

    def test_neither_canonical_nor_countersigned(self):
        """Non-canonical, non-countersigned kline → not S1."""
        m = Model()
        k = KLine(5, [10])  # not canonical (make_sig([10])=10≠5)
        assert is_s1(m, k, signifier) is False

    def test_countersigned_skips_unresolved_nodes(self):
        """Unresolved nodes in kline.nodes are skipped in countersigned search."""
        m = Model()
        a = KLine(5, [99])  # node 99 not in model
        m.add_to_frame(a)
        assert is_s1(m, a, signifier) is False  # not canonical, no resolved nodes to check


class TestIsCountersigned:
    def test_countersigned_in_model(self):
        """Query = {5: [10, 20]}, Countersigner = {make_sig([10,20]): [5]}"""
        m = Model()
        query = KLine(5, [10, 20])
        # make_sig([10, 20]) = 30 (XOR)
        countersigner = KLine(30, [5])
        m.add_to_frame(query)
        m.add_to_frame(countersigner)
        assert is_countersigned(m, query, signifier) is True

    def test_one_way_only(self):
        m = Model()
        a = KLine(5, [10])
        b = KLine(10, [20, 30])  # sig doesn't match make_sig(a.nodes)
        m.add_to_frame(a)
        m.add_to_frame(b)
        assert is_countersigned(m, a, signifier) is False

    def test_no_model_match(self):
        m = Model()
        a = KLine(5, [10])
        assert is_countersigned(m, a, signifier) is False

    def test_countersigner_wrong_node(self):
        """Countersigner has matching sig but wrong node → not countersigned."""
        m = Model()
        query = KLine(5, [10, 20])
        countersigner = KLine(30, [99])  # make_sig([10,20])=30, but node != query.sig
        m.add_to_frame(query)
        m.add_to_frame(countersigner)
        assert is_countersigned(m, query, signifier) is False

    def test_countersigner_multiple_nodes(self):
        """Countersigner has matching sig but more than one node → not countersigned."""
        m = Model()
        query = KLine(5, [10, 20])
        countersigner = KLine(30, [5, 99])  # make_sig([10,20])=30, node has 5 but len>1
        m.add_to_frame(query)
        m.add_to_frame(countersigner)
        assert is_countersigned(m, query, signifier) is False


class TestStructuralSignificance:
    """structural_significance — the pure-structural band, no model state.

    Composed from the kline predicates (is_terminal, is_unknown, is_identity,
    is_canon, is_misfit) and node count; the compound-word identity form is a
    terminal and never named here. The sole model-state adjustment (the S2→S1
    countersigned fork) is applied at the call site, not here.
    """

    def test_empty_identity_ask_is_s4(self):
        assert structural_significance(KLine(42, []), signifier) == SIG_S4

    def test_self_referential_identity_is_s1(self):
        # {S: [S]} is identity, with nodes → self-grounded S1.
        assert structural_significance(KLine(42, [42]), signifier) == SIG_S1

    def test_compound_word_is_s1(self):
        # A §11.3 compound-word: COMPOUND_TOKEN is among the nodes.
        # It is an Identity terminal with nodes → self-grounded S1; its
        # subwords are opaque and never re-enter the band logic here.
        nodes = [0b100, 0b010, COMPOUND_TOKEN]
        kl = KLine(0b110 | COMPOUND_TOKEN, nodes)
        assert structural_significance(kl, signifier) == SIG_S1

    def test_canon_is_s1(self):
        # A canon is a grounded aggregation by structure → S1. (A canon whose
        # reciprocal countersigner is in the model is also S1, trivially.)
        kl = KLine(0b110, [0b100, 0b010])  # sig == signature_of(nodes)
        assert structural_significance(kl, signifier) == SIG_S1

    def test_single_node_relationship_is_s3(self):
        # {A: [B]} (A ≠ B) — connotation / denotation.
        assert structural_significance(KLine(0xFF, [0x01]), signifier) == SIG_S3

    def test_multi_node_misfit_is_s2(self):
        # {AB: [A, C]} — multi-node, not identity, not canon → misfit.
        assert structural_significance(KLine(0b110, [0b100, 0b001]), signifier) == SIG_S2


class TestPromoteParticipating:
    def test_promotes_query_and_candidate(self):
        """Both query and candidate are promoted to LTM."""
        m = Model(stm_bound=256)
        q = KLine(5, [10, 20])
        c = KLine(10, [5, 30])
        m.add_to_frame(q)
        m.add_to_frame(c)
        promote_participating(m, q, c, signifier)
        assert m.find(q.signature) is not None
        assert m.find(c.signature) is not None

    def test_promotes_stm_klines_with_matching_signatures(self):
        """STM klines whose signatures appear in the node set are also promoted."""
        m = Model(stm_bound=256)
        # Identity kline (S4) with sig that appears in query nodes
        identity = KLine(10, [100])  # sig=10 appears in query.nodes
        m.add_to_frame(identity)
        q = KLine(5, [10, 20])
        c = KLine(20, [5, 30])
        m.add_to_frame(q)
        m.add_to_frame(c)
        promote_participating(m, q, c, signifier)
        # identity (sig=10) is in q.nodes, should also be promoted via LTM cascade
        assert m.find(10) is not None

    def test_no_double_promote(self):
        """Calling promote_participating on already-LTM klines is safe (idempotent)."""
        m = Model(stm_bound=256)
        q = KLine(5, [10, 20])
        c = KLine(10, [5, 30])
        m.add_to_frame(q)
        m.add_to_frame(c)
        m.add_to_ltm(q)  # promote to LTM first
        m.add_to_ltm(c)
        promote_participating(m, q, c, signifier)
        # Klines still exist in the model after double promotion
        assert m.find(q.signature) is not None
        assert m.find(c.signature) is not None

    def test_promote_participating_returns_none(self):
        """promote_participating returns None (void)."""
        m = Model(stm_bound=256)
        q = KLine(5, [10, 20])
        c = KLine(10, [5, 30])
        m.add_to_frame(q)
        m.add_to_frame(c)
        result = promote_participating(m, q, c, signifier)
        assert result is None


# ── Significance Boundary Tests ───────────────────────────────────────


class TestBandLayout:
    """Verify BandLayout.classify maps bytes to S1/S2/S3/S4 bands (Q4/Q5).

    Replaces the old TestBoundaries/TestClassify (removed: the 64-bit
    boundaries()/classify() functions are gone; BandLayout is the new path).
    """

    def test_default_boundary(self):
        layout = BandLayout()
        assert layout.s2_s3_boundary == DEFAULT_S2_S3_BOUNDARY == 0x80

    def test_fixed_sentinels(self):
        layout = BandLayout()
        assert layout.sig_s1 == 0xFF
        assert layout.sig_s4 == 0x00

    def test_representatives(self):
        layout = BandLayout()
        assert layout.sig_s2 == 0xFE
        assert layout.sig_s3 == 0x7F  # boundary - 1

    def test_strict_ordering(self):
        layout = BandLayout()
        assert layout.sig_s1 > layout.sig_s2 > layout.sig_s3 > layout.sig_s4

    def test_boundary_is_only_knob(self):
        # Moving the boundary reshuffles S2/S3 but leaves S1/S4 fixed.
        lo = BandLayout(s2_s3_boundary=0x20)
        hi = BandLayout(s2_s3_boundary=0xC0)
        assert lo.sig_s1 == hi.sig_s1 == 0xFF
        assert lo.sig_s4 == hi.sig_s4 == 0x00
        assert lo.sig_s3 < hi.sig_s3

    def test_boundary_must_leave_nonempty_bands(self):
        with pytest.raises(ValueError):
            BandLayout(s2_s3_boundary=0x01)
        with pytest.raises(ValueError):
            BandLayout(s2_s3_boundary=0xFF)

    @pytest.mark.parametrize("boundary", [0x02, 0x40, 0x80, 0xC0, 0xFE])
    def test_classify_covers_all_bands(self, boundary):
        layout = BandLayout(s2_s3_boundary=boundary)
        assert layout.classify(0xFF) == "S1"
        assert layout.classify(0xFE) == "S2"
        assert layout.classify(boundary) == "S2"
        assert layout.classify(boundary - 1) == "S3"
        assert layout.classify(0x01) == "S3"
        assert layout.classify(0x00) == "S4"

    def test_classify_uses_low_byte_only(self):
        # Q7: classification is the routing use; it sees only the low byte.
        layout = BandLayout()
        assert layout.classify(0xDEAD_BEEF) == layout.classify(0xEF)
        assert layout.classify(0x0000_0000) == "S4"


class TestProposeExpansions:
    """Verify propose_expansions() yields (KLine, int) tuples for misfits."""

    def test_canonical_yields_nothing(self):
        """Genuine canon (non-self-referential) → no proposals."""
        m = Model(signifier=signifier)
        # sig t(0b110) = OR(t(0b100), t(0b010)); a genuine canon.
        k = KLine(t(0b110), [t(0b100), t(0b010)])
        result = list(propose_expansions(m, k, 42, signifier))
        assert result == []

    def test_identity_yields_nothing(self):
        """Identity (empty or self-referential) → no proposals.

        An expansion proposal must carry decomposition information; identity
        klines do not.
        """
        m = Model(signifier=signifier)
        assert list(propose_expansions(m, KLine(10, []), 42, signifier)) == []
        assert list(propose_expansions(m, KLine(10, [10]), 42, signifier)) == []

    def test_underfit_yields_proposals(self):
        """Underfit candidate (sig promises more than nodes deliver) → proposals.

        KLine(sig=t(0b110), nodes=[t(0b100)]) has gap t(0b010). With a genuine
        canon contributor headed by t(0b010) in the model, propose_expansions
        yields proposal klines with the passed significance.
        """
        m = Model(signifier=signifier)
        # Genuine canon contributor whose signature overlaps the gap t(0b010).
        contributor = KLine(t(0b110), [t(0b100), t(0b010)])
        m.add_to_frame(contributor)
        m.add_to_ltm(contributor)

        # Underfit kline: sig=t(0b110) promises bits nodes=[t(0b100)] don't deliver
        candidate = KLine(t(0b110), [t(0b100)])
        significance = 0xDEAD

        results = list(propose_expansions(m, candidate, significance, signifier))
        assert len(results) >= 1
        for proposal, sig in results:
            assert isinstance(proposal, KLine)
            assert sig == significance

    def test_overfit_yields_trimmed_and_companion(self):
        """Overfit candidate → trimmed proposal; identity companion is dropped.

        KLine(sig=t(0b110), nodes=[t(0b100), t(0b010), t(0b001)]) has excess
        t(0b001). Trimming yields the genuine canon
        `{t(0b110): [t(0b100), t(0b010)]}`, which is emitted. The companion
        from the single excess node would be `{t(0b001): [t(0b001)]}` —
        identity — and is dropped.
        """
        m = Model(signifier=signifier)
        candidate = KLine(t(0b110), [t(0b100), t(0b010), t(0b001)])
        significance = 0xBEEF

        results = list(propose_expansions(m, candidate, significance, signifier))
        assert len(results) == 1
        proposal, sig = results[0]
        assert proposal.signature == t(0b110)
        assert proposal.nodes == [t(0b100), t(0b010)]
        assert sig == significance

    def test_yields_are_kline_int_tuples(self):
        """Every yield is a (KLine, int) tuple."""
        m = Model(signifier=signifier)
        # Genuine canon contributor whose signature overlaps the gap.
        contributor = KLine(t(0b011), [t(0b001), t(0b010)])
        m.add_to_frame(contributor)
        candidate = KLine(t(0b110), [t(0b100)])

        for item in propose_expansions(m, candidate, 42, signifier):
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], KLine)
            assert isinstance(item[1], int)

    def test_no_self_referential_proposals(self):
        """Self-referential proposals (`{S: [S]}`) are never yielded.

        Regression: a single excess node produced a companion `{n: [n]}`
        and a self-contributor produced a proposal whose nodes included its
        own signature. `{S: [S]}` is identity (not a valid decomposition),
        so it is never emitted as an expansion proposal; under Recency
        Precedence such klines would otherwise displace genuine canons and
        collapse `Model.unpack()` to identity.
        """
        m = Model(signifier=signifier)
        # Genuine canon contributor headed by t(0b100) (overlaps the candidate sig).
        m.add_to_frame(KLine(t(0b110), [t(0b100), t(0b010)]))
        # Overfit candidate whose single excess node (t(0b010)) would yield a
        # self-loop companion {t(0b010): [t(0b010)]}.
        candidate = KLine(t(0b100), [t(0b110)])
        for proposal, _ in propose_expansions(m, candidate, 0, signifier):
            assert proposal.signature not in proposal.nodes
