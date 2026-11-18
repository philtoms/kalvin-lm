"""The model — tests against docs/rationalisation.md."""

from __future__ import annotations

import pytest
from fractions import Fraction

from rationalisation import (
    BETA,
    Band,
    Cogitator,
    Domain,
    KLine,
    band,
    can,
    effort,
    entropy,
    rationalise,
    significance,
)

A, B, C, D = 1, 2, 4, 8
AB, BC, CD, ABC, ABD, ABCD = 3, 6, 12, 7, 11, 15


def k(sig: int, *adj: int) -> KLine:
    return KLine(sig, list(adj))


def base_domain(*extra: KLine) -> Domain:
    """Single atoms held canonically — a simple substrate."""
    return Domain([k(A, A), k(B, B), k(C, C), k(D, D), *extra])


# --- §2 Definitions 4–6: can, canonical, witness ---


def test_can_folds_composition():
    assert can([]) == 0
    assert can([AB]) == AB
    assert can([AB, C]) == ABC


# --- §4 Definition 11: the band ---


def test_bands():
    assert band(k(AB, A, B)) is Band.S1  # canonical
    assert band(k(A, A)) is Band.S1  # single-node canonical
    assert band(k(ABC, AB, C)) is Band.S1  # canonical — composition equals head
    assert band(k(ABCD, AB, C)) is Band.S2  # partial overlap, not canonical
    assert band(k(AB, CD)) is Band.S3  # disjoint
    assert band(k(AB)) is Band.S4  # no adjacency


def test_s2_requires_overlap_short_of_canonical():
    # σ ⪯ can(N) with excess: pure overfit — still S2
    assert band(k(AB, A, B, C)) is Band.S2
    # can(N) ⪯ σ with excess: pure underfit — still S2
    assert band(k(ABCD, AB)) is Band.S2


# --- §4 Definition 13: entropy and graded significance ---


def test_entropy_endpoints():
    assert entropy(k(AB, A, B)) == 0  # canon: no surprise at S1
    assert entropy(k(AB, AB)) == 0  # identity: no surprise at S1 either
    assert significance(k(AB, A, B)) == 1
    assert entropy(k(C, A, B)) == 1  # S3: no correspondence, full surprise
    assert entropy(KLine(AB, [])) == 1  # S4: 100% surprise


def test_significance_interior_for_s2():
    assert significance(k(AB, A)) == Fraction(1, 2)  # F = A, σ ∨ â = AB
    assert 0 < significance(k(ABC, AB)) < 1


def test_band_grade_correspondence():
    for kk in (k(AB, A, B), k(AB, AB), k(AB, A), k(C, A, B), KLine(AB, [])):
        b = band(kk)
        g = significance(kk)
        assert (b is Band.S1) == (g == 1)
        assert (b is Band.S2) == (0 < g < 1)
        assert (b in (Band.S3, Band.S4)) == (g == 0)  # entropy does not split S3/S4


def test_multiple_canonical_witnesses_coexist():
    """Multiple canonical klines coexist under one head — a store never
    displaces an earlier one."""
    d = base_domain(k(AB, AB), k(AB, A, B))
    assert k(AB, AB) in d and k(AB, A, B) in d
    assert d.is_canonical(k(AB, AB)) and d.is_canonical(k(AB, A, B))
    assert len(list(d)) == sum(1 for _ in base_domain()) + 2  # both kept


# --- §4 Definition 14: effort — entropy biased by repairability ---


def test_effort_anchors():
    assert effort(k(AB, A, B)) == 0  # S1 — canon
    assert effort(k(AB, AB)) == 0  # S1 — identity
    assert effort(KLine(AB, [])) == 1  # S4 — exactly full, nothing to work with


def test_effort_s3_strictly_below_s4():
    assert effort(k(C, A, B)) < 1  # S3: adjacency present — repairable
    assert effort(k(C, A, B)) == Fraction(1 + BETA * 2, 3)  # (μ(σ) + β·μ(â))/μ(σ ∨ â) = (1+1)/3
    assert effort(KLine(C, [])) == 1  # S4 with the same claim: strictly more


def test_effort_s2_interior_both_kinds():
    assert 0 < effort(k(AB, A)) < 1  # pure underfit: U carries it
    assert 0 < effort(k(A, AB)) < 1  # pure overfit: β·O carries it (β > 0)


def test_effort_beta_interpolation():
    for kk in (k(AB, A), k(C, A, B), k(ABC, AB), k(AB, A, B)):
        assert effort(kk, beta=Fraction(1)) == entropy(kk)  # β → 1 recovers H
        assert effort(kk) <= entropy(kk)  # E ≤ H for β ≤ 1


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_ends_carry_the_halt_measure():
    d = base_domain(k(AB))  # terminal neighbour
    r = rationalise(d, k(A, A))
    halted = [e for e in r.ends if e.kline is not None]
    assert halted and all(e.effort == effort(e.kline) for e in halted)


# --- §5 Definition 15: proposals at halt ---


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_proposal_connect_from_empty_overlap():
    # nothing is checked at presentation — the no-neighbours halt is the ask
    r = rationalise(Domain(), k(AB, A, B))
    ps = proposals(r)
    assert len(ps) == 1
    assert ps[0].kind == "connect" and ps[0].kline == k(AB, A, B)


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_crossing_candidate_stalls_at_the_underfit():
    # the first candidate ⟨ABC,{AB,16}⟩ fits AB, overfits 16, underfits C:
    # the 16 child grounds under its own claim, the AB child merges the
    # grounded 16 (both clear), and the base stalls proposing the
    # grounded-so-far kline with C still asking
    d = Domain([KLine(0b1011, [AB, D]), KLine(0b10011, [AB, 16])])
    r = rationalise(d, k(ABC, A, B, C))
    assert r.stores == [KLine(16, [16]), KLine(AB, [AB, 16])]
    assert [(e.reason, e.kline) for e in r.ends] == [
        ("no candidates — the entry terminates", k(C, C)),
        ("stalled at S2: underfit ungrounded", KLine(ABC, [AB, 16])),
    ]


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_disjoint_candidate_asks_s4():
    # ⟨C, {AB}⟩ shares no bit with the claim — F = ∅: the ask is the
    # empty kline ⟨C, ∅⟩, the full-surprise acquire
    d = base_domain(k(AB), k(ABC, AB))
    r = rationalise(d, k(C, C))
    assert r.stores == []
    ends = [e for e in r.ends if e.reason == "S4: candidate with no fit"]
    assert ends and ends[0].kline == k(C) and ends[0].effort == 1


def test_unaligned_stalls_broadcast_bare():
    # atoms only — the C ask meets nobody but identities: both unaligned
    # sides stall and broadcast bare at the run's end
    d = base_domain()
    stalls: list[KLine] = []
    cog = Cogitator(d, on_stall=stalls.append)
    cog.rationalise(k(C, C), goal=k(AB, A, B))
    cog.cogitate()
    assert sorted(int(s.signature) for s in stalls) == [AB, C]  # both sides starved
    assert all(not s.nodes for s in stalls)  # a broadcast stall is bare — pure S4
    assert all(effort(s) == 1 for s in stalls)  # exactly full — nothing to work with


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_proposals_ordered_by_significance():
    # ends at E = 1/3 (the stalled frame's grounded-so-far ⟨ABC,{A,B}⟩)
    # and E = 1 (the S4 ask ⟨C, ∅⟩): least entropy sorts first
    d = Domain([k(AB, A, B), k(C)])  # k(C) is terminal
    r = rationalise(d, k(ABC, A, B, C))
    ps = proposals(r)
    assert len(ps) == 2
    assert ps == sorted(ps, key=lambda p: p.effort)  # ascending effort
    assert ps[0].kind == "connect" and ps[0].kline == KLine(ABC, [A, B]) and ps[0].effort == Fraction(1, 3)
    assert ps[1].kind == "acquire" and ps[1].kline == k(C) and ps[1].effort == 1


# --- §5 R1: stores; admission iff S1 ---


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_r1_stores_the_s1_candidate():
    d = base_domain(k(ABD, A, B))  # held but non-canonical (can = AB)
    r = rationalise(d, k(A, A, B))
    # the candidate ⟨AB, {A, B}⟩ is itself S1 — stored as presented (Rule R1)
    assert r.stores == [k(AB, A, B)]
    assert k(A, A, B) not in d  # the entry itself is never stored
    assert d.is_canonical(k(AB, A, B))  # a full canon (Definition 5)


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_terminal_candidate_asks_s4():
    d = base_domain(k(AB))  # terminal neighbour
    r = rationalise(d, k(A, A))
    assert any(e.reason.startswith("S4") for e in r.ends)


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_empty_lookup_terminates_entry():
    d = Domain()  # no neighbours overlap the entry signature
    r = rationalise(d, k(AB, A, B))
    assert any("no candidates" in e.reason for e in r.ends)
    assert r.stores == []


# --- §5 rationalise: the ask run ---


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_canon_memo_is_per_run():
    """The canon memo bounds one run's recursion (§7) — a later run of the
    same entry starts fresh: both runs process fully and agree."""

    def fresh():
        return rationalise(base_domain(k(AB, A, B)), k(A, A))

    first, second = fresh(), fresh()
    # both runs ground ⟨A, {A,B}⟩: the B atom grounds the overfit side,
    # the merge folds it in, both sides clear
    assert first.stores == second.stores == [KLine(A, [A, B])]
    assert first.ends == second.ends == []  # the memo leaves no residue across runs


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_s3_grounds_nothing_asks_instead_of_claiming():
    # the ⟨ABC,{AB}⟩ witness shares no bit with the claim C — F = ∅: an
    # acquire ask on the empty kline, not a contentless licensed claim
    d = base_domain(k(AB, A, B), k(ABC, AB))
    r = rationalise(d, k(C, C))
    assert r.stores == []
    ends = [e for e in r.ends if e.reason == "S4: candidate with no fit"]
    assert ends and ends[0].kline == k(C)


# --- §5 candidate consolidation ---


def test_the_claim_grounds_when_both_sides_resolve():
    # the ask's underfit region reenters and grounds bottom-up over the
    # held canons; the claim broadcasts at every S2 alignment and the
    # domain is untouched
    d = Domain([k(AB, A, B), k(BC, B, C)])
    before = set(d)
    claims: list[KLine] = []
    stalls: list[KLine] = []
    cog = Cogitator(d, on_claim=claims.append, on_stall=stalls.append)
    cog.rationalise(k(ABC, A, B, C), goal=k(AB, A, B))
    cog.cogitate()
    assert claims  # at least one claim left — and every one is the claim
    assert all(c == KLine(ABC, [AB]) for c in claims)  # σ_ask over σ_goal
    assert stalls == []  # nothing stalled — every side aligned
    assert set(d) == before  # nothing stored — the domain is the harness's


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_identities_witness_nothing():
    # over a pure atom substrate every neighbour is an identity — the
    # entry cannot ground through its own reflexive decomposition; it asks
    d = base_domain()
    r = rationalise(d, k(ABC, A, B, C))
    assert r.stores == []
    assert [e.reason for e in r.ends] == ["no candidates — the entry terminates"]
    ps = proposals(r)
    assert ps[0].kind == "connect" and ps[0].kline == k(ABC, A, B, C)


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_partial_fit_never_stores_alone_the_frame_stalls():
    # only the AB canon witnessed: the frame's f = {A,B}, u = {C}; the C
    # reentry meets nobody and asks, then the frame stalls proposing the
    # grounded-so-far kline — the fit itself never stores
    d = Domain([k(AB, A, B)])
    r = rationalise(d, k(ABC, A, B, C))
    assert r.stores == []
    ps = proposals(r)
    assert [(p.kind, p.kline) for p in ps] == [
        ("connect", k(C, C)),
        ("connect", KLine(ABC, [A, B])),
    ]


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_terminal_neighbours_consolidate_into_one_ask():
    # two terminal witnesses overlap the canon — one popped candidate, one
    # S4 ask on ⟨A, ∅⟩, not one per terminal neighbour
    d = base_domain(k(AB), k(A | C))
    r = rationalise(d, k(A, A))
    halts = [e for e in r.ends if e.reason == "S4: candidate with no fit"]
    assert len(halts) == 1 and halts[0].kline == k(A)


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_disjoint_overfit_asks_s4():
    # the popped candidate ⟨AB, {D}⟩ shares no bit with the claim — F = ∅:
    # the empty kline asks; no substitution runs at reentry anymore
    d = base_domain(k(C, D), k(D, A, B))
    r = rationalise(d, k(C, A, B))
    assert r.stores == []
    ends = [e for e in r.ends if e.reason == "S4: candidate with no fit"]
    assert ends and ends[0].kline == k(AB)


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_witness_node_never_a_head_asks_s4():
    # the witness node C is never a held head — the candidate ⟨AB,{C}⟩
    # fits nothing: the empty kline asks, no proof substitution
    d = Domain([KLine(A | C, [C])])
    r = rationalise(d, k(AB, A, B))
    assert r.stores == []
    ends = [e for e in r.ends if e.reason == "S4: candidate with no fit"]
    assert ends and ends[0].kline == k(AB)


@pytest.mark.xfail(reason="mid-migration debt: old module-level rationalise semantics (e93c192); superseded by the Cogitator reentry - port pending", strict=False)
def test_crossing_node_witnesses_nothing():
    # the crossing node BC lies in neither the fit nor the overfit region
    # — n_f and n_o stay empty; the grounded A side merges, both clear,
    # and the claim grounds carrying only A
    d = base_domain(k(ABC, BC))
    r = rationalise(d, k(AB, A, B))
    assert r.stores == [KLine(AB, [A])]
    assert r.ends == []

