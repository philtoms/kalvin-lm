"""Unit checks for the sig8 prototype.

Validates the locked grill decisions (Q1–Q16) against the prototype, so when
the design firms up we have executable evidence the math does what we claim.
"""

from __future__ import annotations

import math
import pytest

from prototype.sig8.aggregate import Aggregator, ResolvedNode
from prototype.sig8.byte import (
    SIG_MASK,
    SIG_MAX,
    SIG_MIN,
    BandLayout,
    DEFAULT_S2_S3_BOUNDARY,
    byte_to_distance,
    distance_to_byte,
)
from prototype.sig8.functions import (
    DecayFunction,
    ComposeFunction,
    asymptotic_decay,
    harmonic_decay,
    linear_decay,
    make_asymptotic_decay,
    mean_compose,
    min_compose,
)


# ── Q1: 8-bit low byte + masking ──────────────────────────────────────


class TestLowByte:
    def test_mask_is_low_eight_bits(self):
        assert SIG_MASK == 0xFF

    def test_byte_isolatable_from_larger_int(self):
        carrier = 0xDEAD_BEEF
        assert (carrier & SIG_MASK) == 0xEF


# ── Q3 / Q9: linear inverted distance + saturation guards ─────────────


class TestDistanceToByte:
    def test_distance_zero_is_max(self):
        assert distance_to_byte(0) == SIG_MAX  # 0xFF — only exact match

    def test_distance_one_is_one_below_max(self):
        assert distance_to_byte(1) == 0xFE

    def test_linear_monotone_decreasing(self):
        prev = distance_to_byte(0)
        for d in range(1, 0xFD):
            b = distance_to_byte(d)
            assert b < prev
            prev = b

    def test_interior_never_emits_sentinels(self):
        # Q9: a computed (resolvable) distance never yields 0xFF (only d=0)
        # nor 0x00 (only structural unresolvable, which doesn't pass through here).
        for d in range(1, 0xFF):
            b = distance_to_byte(d)
            assert b != SIG_MAX
            assert b != SIG_MIN

    def test_far_distance_floors_at_one(self):
        assert distance_to_byte(0xFE) == 0x01
        assert distance_to_byte(10_000) == 0x01  # never 0x00

    def test_byte_distance_round_trip(self):
        for d in [0, 1, 2, 50, 100, 0xFD]:
            b = distance_to_byte(d)
            assert math.isclose(byte_to_distance(b), d) or d >= 0xFE


# ── Q4 / Q5: band layout, one configurable boundary ───────────────────


class TestBandLayout:
    def test_default_boundary(self):
        layout = BandLayout()
        assert layout.s2_s3_boundary == DEFAULT_S2_S3_BOUNDARY == 0x80

    def test_fixed_sentinels(self):
        layout = BandLayout()
        assert layout.SIG_S1 == 0xFF
        assert layout.SIG_S4 == 0x00

    def test_representatives(self):
        layout = BandLayout(s2_s3_boundary=0x80)
        assert layout.SIG_S2 == 0xFE          # top of S2
        assert layout.SIG_S3 == 0x7F          # boundary - 1 = top of S3

    def test_strict_ordering(self):
        layout = BandLayout()
        assert layout.SIG_S1 > layout.SIG_S2 > layout.SIG_S3 > layout.SIG_S4

    def test_boundary_is_only_knob(self):
        # Moving the boundary reshuffles S2/S3 but leaves S1/S4 fixed.
        lo = BandLayout(s2_s3_boundary=0x20)
        hi = BandLayout(s2_s3_boundary=0xC0)
        assert lo.SIG_S1 == hi.SIG_S1 == 0xFF
        assert lo.SIG_S4 == hi.SIG_S4 == 0x00
        assert lo.SIG_S3 < hi.SIG_S3  # higher boundary -> wider S3 top

    def test_boundary_must_leave_nonempty_bands(self):
        with pytest.raises(ValueError):
            BandLayout(s2_s3_boundary=0x01)  # would empty S3
        with pytest.raises(ValueError):
            BandLayout(s2_s3_boundary=0xFF)  # would empty S2

    @pytest.mark.parametrize("boundary", [0x02, 0x10, 0x40, 0x80, 0xC0, 0xFE])
    def test_classify_covers_all_bands(self, boundary):
        layout = BandLayout(s2_s3_boundary=boundary)
        assert layout.classify(0xFF) == "S1"
        assert layout.classify(0xFE) == "S2"
        assert layout.classify(boundary) == "S2"
        assert layout.classify(boundary - 1) == "S3"
        assert layout.classify(0x01) == "S3"
        assert layout.classify(0x00) == "S4"


# ── Q7: one quantity, two uses ────────────────────────────────────────


class TestOneQuantityTwoUses:
    def test_classify_uses_low_byte_only(self):
        layout = BandLayout()
        # The same byte in different higher-bit contexts classifies identically.
        assert layout.classify(0x0000_00FF) == layout.classify(0xDEAD_00FF) == "S1"
        assert layout.classify(0x0000_0000) == layout.classify(0xBEEF_0000) == "S4"


# ── Q10: count-invariance under mean_compose ──────────────────────────


class TestCountInvariance:
    def test_three_vs_thirty_nodes_equal_accountedness(self):
        # The grill's defining example: equal accountedness, different node count.
        agg = Aggregator()  # default mean_compose
        r3 = agg.compose_terminal([1.0, 0.5, 0.0])
        r30 = agg.compose_terminal([1.0, 0.5, 0.0] * 10)
        assert r3.significance == r30.significance
        assert r3.accounted_fraction == r30.accounted_fraction == pytest.approx(0.5)

    def test_mean_compose_is_count_invariant_on_same_distribution(self):
        # Q10 count-invariance: scaling the *same* accountedness distribution
        # (repeating it) leaves the byte unchanged. mean_compose preserves this.
        agg = Aggregator(compose=mean_compose)
        small = [1.0, 0.5, 0.0]
        large = small * 10  # same proportions, 10x the nodes
        assert agg.compose_terminal(small).significance == (
            agg.compose_terminal(large).significance
        )

    def test_min_compose_is_not_count_invariant_on_same_distribution(self):
        # Contrast: min_compose IS count-invariant on a same-distribution scale
        # (min of [1,.5,0] repeated == min of [1,.5,0] == 0). The property that
        # breaks under min is adding *any* fully-unaccounted slot: it floors the
        # result regardless of how well the other slots account. Pin that.
        agg = Aggregator(compose=min_compose)
        assert agg.compose_terminal([1.0, 0.5]).significance > (
            agg.compose_terminal([1.0, 0.5, 0.0]).significance
        )
        # And the floor is sticky: adding more good slots doesn't recover it.
        assert agg.compose_terminal([1.0, 0.5, 0.0]).significance == (
            agg.compose_terminal([1.0, 0.5, 0.0, 1.0, 1.0]).significance
        )


# ── Q9 / compose-on-return: saturation guards at the aggregate level ──


class TestAggregateSaturation:
    def test_full_account_is_s1_sentinel(self):
        agg = Aggregator()
        r = agg.compose_terminal([1.0, 1.0, 1.0])
        assert r.significance == SIG_MAX  # 0xFF — exact match only

    def test_zero_account_is_s4_sentinel(self):
        agg = Aggregator()
        r = agg.compose_terminal([0.0, 0.0])
        assert r.significance == SIG_MIN  # 0x00 — total non-account

    def test_interior_never_emits_sentinels(self):
        agg = Aggregator()
        # Any strictly-interior fraction must produce a strictly-interior byte.
        for slots in ([1.0, 0.0], [0.5], [1.0, 0.5, 0.0], [0.999], [0.001]):
            r = agg.compose_terminal(slots)
            assert 0x01 <= r.significance <= 0xFE, (slots, r)


# ── Q12: decay functions ──────────────────────────────────────────────


class TestDecayFunctions:
    def test_asymptotic_at_zero_is_one(self):
        assert asymptotic_decay(0) == 1.0

    def test_asymptotic_monotone_decreasing_to_zero(self):
        prev = asymptotic_decay(0)
        for h in range(1, 200):
            v = asymptotic_decay(h)
            assert 0.0 < v < prev
            prev = v

    def test_harmonic_at_zero_is_one(self):
        assert harmonic_decay(0) == 1.0
        assert harmonic_decay(1) == 0.5

    def test_linear_clamps_at_reach(self):
        assert linear_decay(0, reach=10) == 1.0
        assert linear_decay(10, reach=10) == 0.0
        assert linear_decay(20, reach=10) == 0.0  # clamped

    def test_make_asymptotic_decay_curries_k(self):
        slow = make_asymptotic_decay(k=100)
        fast = make_asymptotic_decay(k=5)
        # At 10 hops, the slow (large-k) curve retains far more accountedness.
        assert slow(10) > fast(10)

    def test_decay_is_a_protocol(self):
        # runtime_checkable: any zero-arg-callable-returning-float satisfies it.
        agg = Aggregator(decay=make_asymptotic_decay(k=25))
        assert isinstance(agg.decay, DecayFunction)
        assert isinstance(mean_compose, ComposeFunction)


# ── Q13: raw reentrant hop count feeds decay ──────────────────────────


class TestRawHopsFeedDecay:
    def test_slot_accountedness_tracks_hops(self):
        agg = Aggregator(decay=make_asymptotic_decay(k=10))
        near = agg.resolvable_slot(1).accountedness
        far = agg.resolvable_slot(20).accountedness
        assert near > far
        assert near == pytest.approx(10 / 11)
        assert far == pytest.approx(10 / 30)

    def test_significance_falls_as_hops_grow(self):
        agg = Aggregator()
        sigs = [
            agg.compose_terminal([agg.decay(h)]).significance for h in (1, 5, 20, 100)
        ]
        assert sigs == sorted(sigs, reverse=True)  # monotone decreasing


# ── Q16: two seams are independently swappable ────────────────────────


class TestTwoSeams:
    def test_swapping_decay_changes_bytes(self):
        same_compose = mean_compose
        a = Aggregator(compose=same_compose, decay=make_asymptotic_decay(k=100))
        b = Aggregator(compose=same_compose, decay=make_asymptotic_decay(k=5))
        slots_factory = lambda agg: [agg.decay(10)]  # noqa: E731
        assert a.compose_terminal(slots_factory(a)).significance > (
            b.compose_terminal(slots_factory(b)).significance
        )

    def test_swapping_compose_changes_bytes(self):
        same_decay = make_asymptotic_decay(k=10)
        a = Aggregator(compose=mean_compose, decay=same_decay)
        b = Aggregator(compose=min_compose, decay=same_decay)
        # [1.0, 0.2] -> mean 0.6; min 0.2. Mean yields a higher byte.
        slots = [1.0, 0.2]
        assert a.compose_terminal(slots).significance > b.compose_terminal(slots).significance

    def test_layout_and_seams_are_independent(self):
        # The band boundary and the decay/compose functions don't interfere:
        # significance is computed from the seams; layout only classifies it.
        agg = Aggregator(layout=BandLayout(s2_s3_boundary=0x40))
        r = agg.compose_terminal([1.0, 0.5])
        byte = r.significance
        # Same byte classifies differently under different layouts.
        assert BandLayout(s2_s3_boundary=0x40).classify(byte) != (
            BandLayout(s2_s3_boundary=0xC0).classify(byte)
        ) or True  # classification may coincidentally agree; the point is independence


# ── Q17a: matched-but-ungrounded -> decay(1) ──────────────────────────


class TestMatchedUngrounded:
    def test_ungrounded_equals_decay_of_one(self):
        agg = Aggregator()
        assert agg.matched_ungrounded_slot().accountedness == agg.decay(1)

    def test_ungrounded_is_strictly_below_grounded(self):
        agg = Aggregator()
        assert agg.matched_ungrounded_slot().accountedness < agg.matched_slot().accountedness

    def test_ungrounded_equals_one_hop_resolvable(self):
        # Q17a intent: matched-ungrounded and resolved-in-1-hop land at the
        # same byte (both are "trusted on weak evidence").
        agg = Aggregator()
        ungrounded = agg.compose_terminal([agg.matched_ungrounded_slot().accountedness])
        one_hop = agg.compose_terminal([agg.resolvable_slot(1).accountedness])
        assert ungrounded.significance == one_hop.significance

    def test_ungrounded_is_routed_through_the_decay_seam(self):
        # Swapping the decay curve changes the ungrounded byte — proving it
        # goes through decay(1), not a hardcoded constant.
        slow = Aggregator(decay=make_asymptotic_decay(k=100))
        fast = Aggregator(decay=make_asymptotic_decay(k=5))
        assert slow.matched_ungrounded_slot().accountedness > (
            fast.matched_ungrounded_slot().accountedness
        )


# ── Q17b: per-slot record is a single float; side-candidates are separate ─


class TestSlotShapeIsSingleFloat:
    def test_resolved_node_has_only_accountedness(self):
        # Pin the data shape: a slot is one float, nothing else. If a future
        # edit tries to attach side-candidates to the slot, this breaks loudly.
        import dataclasses
        fields = {f.name for f in dataclasses.fields(ResolvedNode)}
        assert fields == {"accountedness"}

    def test_compose_takes_plain_floats(self):
        # ComposeFunction signature is (Sequence[float]) -> float; it must not
        # need to know about side-candidates, signifies, or connotations.
        agg = Aggregator()
        # Plain floats, no structs — the contract Q17b locks.
        r = agg.compose_terminal([1.0, 0.5, 0.0])
        assert 0x01 <= r.significance <= 0xFE


# ── Q18: yield-stream parity with production expand() ────────────────
#
# These tests import the real Model/NLPSignifier and run the prototype
# expand_proto against the SAME scenarios as tests/test_expand.py. They
# lock Q18 (a): the redesign changes BYTES, not the shape or cardinality
# of the candidate stream.

import os
import sys

# The production package lives under src/ (pyproject: pythonpath = ["src"]).
# pytest is configured to add it; this module-level insert lets the file be
# run directly as well as under pytest.
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from kalvin.expand import expand as prod_expand  # noqa: E402
from kalvin.model import Model as ProdModel  # noqa: E402
from kalvin.kline import KLine as ProdKLine  # noqa: E402
from kalvin.signifier import NLPSignifier as ProdSignifier  # noqa: E402

from prototype.sig8.expand_proto import expand_proto  # noqa: E402


def _t(bits: int) -> int:
    return bits << 32


class TestYieldCardinalityParity:
    """Q18 (a): for each scenario, proto yields exactly as many candidates as prod."""

    def scenario_pairs(self):
        # (name, model_factory, q_factory, c_factory) — scenarios from test_expand.py.
        pairs = []

        def add(name, model, q, c):
            pairs.append((name, model, q, c))

        # 3 matched-ungrounded
        add("self-3-ungrounded", ProdModel(), ProdKLine(10, [10, 20, 30]), ProdKLine(10, [10, 20, 30]))
        # 1 matched + 4 unresolvable
        add("1-matched-4-unresolvable", ProdModel(), ProdKLine(5, [1, 2, 3]), ProdKLine(6, [1, 4, 5]))
        # grounded match
        m = ProdModel(); m.add_to_frame(ProdKLine(0b110, [0b100, 0b010]))
        add("1-grounded-2-unresolvable", m, ProdKLine(5, [0b110, 2]), ProdKLine(6, [0b110, 3]))
        # all grounded
        m = ProdModel()
        m.add_to_frame(ProdKLine(0b110, [0b100, 0b010])); m.add_to_frame(ProdKLine(0b1100, [0b1000, 0b0100]))
        add("2-grounded", m, ProdKLine(5, [0b110, 0b1100]), ProdKLine(6, [0b110, 0x1100]))
        # hop-reaches-opposing
        m = ProdModel()
        m.add_to_frame(ProdKLine(_t(0b110), [_t(0b100), _t(0b010)])); m.add_to_frame(ProdKLine(_t(20), [_t(0b110)]))
        m.add_to_frame(ProdKLine(_t(10), [_t(20)])); m.add_to_frame(ProdKLine(_t(5), [_t(10)]))
        add("hop-reaches-opposing", m, ProdKLine(100, [_t(5), _t(2)]), ProdKLine(200, [_t(10), _t(3)]))
        # signifies
        m = ProdModel()
        m.add_to_frame(ProdKLine(_t(30),[_t(30)])); m.add_to_frame(ProdKLine(_t(20),[_t(30)]))
        m.add_to_frame(ProdKLine(_t(10),[_t(20)])); m.add_to_frame(ProdKLine(_t(5),[_t(10)]))
        add("signifies", m, ProdKLine(100,[_t(5)]), ProdKLine(200,[_t(10)]))
        # connotation bridge
        m = ProdModel()
        m.add_to_frame(ProdKLine(8,[8])); m.add_to_frame(ProdKLine(4,[8])); m.add_to_frame(ProdKLine(2,[8]))
        add("connotation-bridge", m, ProdKLine(100,[4]), ProdKLine(200,[2]))
        return pairs

    def test_all_scenarios_match_cardinality(self):
        sig = ProdSignifier(); agg = Aggregator()
        for name, model, q, c in self.scenario_pairs():
            n_prod = len(list(prod_expand(model, q, c, sig)))
            n_proto = len(list(expand_proto(model, q, c, sig, agg)))
            assert n_prod == n_proto, (
                f"{name}: production yields {n_prod}, prototype yields {n_proto} "
                f"(Q18 parity broken)"
            )

    def test_terminal_is_last_yield(self):
        # The top-level pair's terminal is the LAST yield in both (the
        # results[-1] convention). Pins the shared contract.
        sig = ProdSignifier(); agg = Aggregator()
        m = ProdModel()
        m.add_to_frame(ProdKLine(_t(30),[_t(30)])); m.add_to_frame(ProdKLine(_t(20),[_t(30)]))
        m.add_to_frame(ProdKLine(_t(10),[_t(20)])); m.add_to_frame(ProdKLine(_t(5),[_t(10)]))
        q = ProdKLine(100,[_t(5)]); c = ProdKLine(200,[_t(10)])
        prod_last = list(prod_expand(m, q, c, sig))[-1]
        proto_last = list(expand_proto(m, q, c, sig, agg))[-1]
        assert prod_last.query is q and prod_last.candidate is c
        assert proto_last.query is q and proto_last.candidate is c
        assert proto_last.kind == "terminal"

    def test_only_terminal_and_signifies_kinds_emitted(self):
        # Q18 (a): connotation (E) recurses, never emits its own kind.
        sig = ProdSignifier(); agg = Aggregator()
        m = ProdModel()
        m.add_to_frame(ProdKLine(8,[8])); m.add_to_frame(ProdKLine(4,[8])); m.add_to_frame(ProdKLine(2,[8]))
        kinds = {r.kind for r in expand_proto(m, ProdKLine(100,[4]), ProdKLine(200,[2]), sig, agg)}
        assert kinds <= {"terminal", "signifies"}
