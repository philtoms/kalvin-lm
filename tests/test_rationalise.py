"""Unit tests for the Rationalising Trainee (plan §Phase 2.1).

Spec seam: ``@specs/dialogue-driven-training.md`` §Actor — the Rationaliser
satisfies the existing Actor contract; this plan introduces no new spec IDs.
These tests exercise each mechanism branch in isolation (entry rule, S4
identity, S3 pairing, cleanup, grouping, MTS discrimination, termination), not only through
the MHALL golden master (which the runner integration test in
``test_dialogue_runner.py`` covers end-to-end).

Synthetic scenarios use small OR-reduced signatures built from the signifier so
canon/MTS status is controlled (a single-token signature is non-MTS; an
OR-reduction of ≥2 tokens is an MTS once its canon is grounded).

Deferred coverage gaps (G1: S2 multi-node proposal branch; G2: group-size
escalation on trainer-S4 refusal) are marked ``pytest.skip`` pointing at the gap
rather than asserting unimplemented behaviour.
"""

from __future__ import annotations

import pytest

from kalvin.expand import SIG_S1, SIG_S2, SIG_S4
from kalvin.kline import KLine, is_canon
from kalvin.kvalue import KValue
from kalvin.signifier import NLPSignifier
from tests._fixtures import mhall_table
from training.dialogue.decoder import decode, load_table
from training.dialogue.rationalise import Rationaliser

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def signifier() -> NLPSignifier:
    return NLPSignifier()


@pytest.fixture
def rationaliser(signifier: NLPSignifier) -> Rationaliser:
    return Rationaliser(signifier)


def _value(kline: KLine, sig: int) -> KValue:
    """Build an incoming KValue carrying ``kline`` at ``sig``."""
    return KValue(kline, sig)


def _work_list(r: Rationaliser) -> list[tuple[int, list[int]]]:
    return [(e.signature, list(e.nodes)) for e in r._state.work_list]


# ── Construction & engine interface ────────────────────────────────────────


def test_rationalise_returns_batch_or_empty(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """rationalise returns a list of KValues while there is work, empty when idle."""
    a = signifier.make_signature([0b1])
    ab = signifier.make_signature([0b1, 0b10])
    # No work, no incoming -> empty list (D12: termination is the runner's job).
    assert rationaliser.rationalise(None) == []
    # With an S2 incoming, cogitation emits an identity blast (every workable
    # identity at S4) — here the query's signature and its unrecognised node.
    emitted = rationaliser.rationalise(_value(KLine(ab, [a]), SIG_S2))
    assert isinstance(emitted, list)
    assert len(emitted) == 2
    assert all(e.significance == SIG_S4 for e in emitted)
    assert {e.kline.signature for e in emitted} == {ab, a}
    assert all(list(e.kline.nodes) == [] for e in emitted)


# ── Entry rule ────────────────────────────────────────────────────────────


def test_entry_s2_pushes_query_and_identities(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """An S2 kline is unpacked: the query kline + node identities + (if new) the
    signature identity are pushed onto the work-list."""
    a = signifier.make_signature([0b1])
    ab = signifier.make_signature([0b1, 0b10])  # MTS
    rationaliser._process_query(_value(KLine(ab, [a]), SIG_S2))
    # query kline, node identity {a:[]}, signature identity {ab:[]} all present.
    assert (ab, [a]) in _work_list(rationaliser)
    assert (a, []) in _work_list(rationaliser)
    assert (ab, []) in _work_list(rationaliser)


def test_entry_s2_recognised_node_not_repushed(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """A node already grounded or in flight is not pushed as a fresh identity."""
    a = signifier.make_signature([0b1])
    b = signifier.make_signature([0b10])
    ab = signifier.make_signature([0b1, 0b10])
    # Ground A first, then feed {AB:[A,B]} — A is grounded, B is new.
    rationaliser._process_query(_value(KLine(a, []), SIG_S1))
    rationaliser._process_query(_value(KLine(ab, [a, b]), SIG_S2))
    wl = _work_list(rationaliser)
    assert wl.count((a, [])) == 0  # A grounded -> no identity pushed
    assert wl.count((b, [])) == 1  # B new -> one identity


def test_entry_s4_pops_matching_identity(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """An S4 retires the matching in-flight identity work-item (stalemate)."""
    a = signifier.make_signature([0b1])
    ab = signifier.make_signature([0b1, 0b10])
    rationaliser._process_query(_value(KLine(ab, [a]), SIG_S2))
    assert (ab, []) in _work_list(rationaliser)
    rationaliser._process_query(_value(KLine(ab, []), SIG_S4))
    assert (ab, []) not in _work_list(rationaliser)


def test_entry_s1_grounds_and_cleanup_retires_identities(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """An S1 grounds the kline; cleanup retires identities whose signature now
    grounds (an identity {sig:[]} ≡ {sig:[sig]} grounds when sig grounds)."""
    a = signifier.make_signature([0b1])
    ab = signifier.make_signature([0b1, 0b10])
    rationaliser._process_query(_value(KLine(ab, [a]), SIG_S2))  # pushes {ab:[]}
    assert (ab, []) in _work_list(rationaliser)
    rationaliser._process_query(_value(KLine(a, []), SIG_S1))  # ground A
    assert a in rationaliser._state.grounded
    # {a:[]} identity (if present) retires; {ab:[]} stays (AB not yet grounded).
    # AB grounding requires its canon {AB:[A,B]} nodes all grounded — not yet.


def test_entry_s1_canon_grounds_when_all_nodes_ground(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """A canon {S:[nodes]} grounds (recursively) once all its nodes ground."""
    a = signifier.make_signature([0b1])
    b = signifier.make_signature([0b10])
    ab = signifier.make_signature([0b1, 0b10])
    # Ground the canon {AB:[A,B]} directly via S1; A and B should not need
    # separate grounding — the canon IS the S1 trigger.
    rationaliser._process_query(_value(KLine(ab, [a, b]), SIG_S1))
    assert ab in rationaliser._state.grounded


# ── S4 (Identity) ────────────────────────────────────────────────────────


def test_level0_emits_identity_at_s4(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """S4: an unrecognised signature's identity is emitted at S4.

    Cogitation batches every workable identity: the query's signature and its
    unrecognised node are both emitted at S4 in one blast."""
    ab = signifier.make_signature([0b1, 0b10])
    a = signifier.make_signature([0b1])
    emitted = rationaliser.rationalise(_value(KLine(ab, [a]), SIG_S2))
    assert len(emitted) == 2
    assert all(e.significance == SIG_S4 for e in emitted)
    assert {e.kline.signature for e in emitted} == {ab, a}
    assert all(list(e.kline.nodes) == [] for e in emitted)


def test_level0_no_opening_special_case(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """D8: the opening turn (first S2, no prior incoming) flows through the same
    path as any other S2 — there is no opening special-case."""
    ab = signifier.make_signature([0b1, 0b10])
    a = signifier.make_signature([0b1])
    opening = rationaliser.rationalise(_value(KLine(ab, [a]), SIG_S2))
    # Same emission as test_level0_emits_identity_at_s4 — turn-0 == any S2 turn.
    assert len(opening) == 2
    assert {e.kline.signature for e in opening} == {ab, a}
    assert opening[0].significance == SIG_S4


# ── Cleanup & MTS discrimination ──────────────────────────────────────────


def test_relationship_grounds_by_elevation_on_rereceipt(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """A relationship grounds by ELEVATION on re-receipt, not by node-resolution
    in cleanup. {a:[Det]} is received at S2 (Det ungrounded) and stays pending;
    once Det grounds, re-receiving {a:[Det]} at S2 elevates it — K re-derives
    its own significance (node now grounded) and grounds it at S1. This is the
    async-grounding mechanism."""
    a = signifier.make_signature([0b1])
    det = signifier.make_signature([0b10, 0b100])
    # First receipt: {a:[Det]} at S2, Det ungrounded -> not elevatable -> unpack.
    rationaliser._process_query(_value(KLine(a, [det]), SIG_S2))
    assert a not in rationaliser._state.grounded
    # Ground Det.
    rationaliser._process_query(_value(KLine(det, []), SIG_S1))
    assert a not in rationaliser._state.grounded  # still not grounded — no re-receipt yet
    # Re-receive {a:[Det]} at S2: now Det grounded -> elevation -> a grounds.
    assert rationaliser._elevatable(KLine(a, [det]))
    rationaliser._process_query(_value(KLine(a, [det]), SIG_S2))
    assert a in rationaliser._state.grounded


def test_relationship_does_not_ground_via_cleanup(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """Relationships never ground by node-resolution in cleanup (only canons and
    identities do). The opening {MHALL:[SVO]} (a relationship) is NOT
    cleanup-groundable even once SVO grounds — it grounds by K's own closing S1
    broadcast, not by cleanup. Asserts the _groundable predicate directly."""
    a = signifier.make_signature([0b1])
    ab = signifier.make_signature([0b1, 0b10])  # compound
    rel = KLine(ab, [a])  # a relationship (non-canon, non-identity)
    assert not rationaliser._groundable(rel)  # relationships are never cleanup-groundable
    # Even after grounding A, the relationship {AB:[A]} is not cleanup-groundable.
    rationaliser._process_query(_value(KLine(a, []), SIG_S1))
    assert not rationaliser._groundable(rel)


# ── S3 (Relationships) — grouping (D10) ───────────────────────────────────


@pytest.fixture(scope="module")
def _decoded_mhall():
    table = load_table(mhall_table())
    return decode(table, signifier=NLPSignifier())


def _drive_to_s3(rationaliser: Rationaliser, decoded) -> None:
    """Drive the rationaliser through the identity phase until the work-list has
    settled to the opening relationship (S3 boundary).

    Mirrors the working end-to-end simulation: T-rows become incoming, K-rows
    trigger respond(incoming). The first S3 K-row is NOT consumed, but the
    T-row preceding it (which grounds the last operand and triggers the
    cleanup reducing the work-list) is processed via its entry rule so the
    work-list settles to the opening relationship ready for S3 cogitation.
    """
    incoming = None
    ki = 0
    k_rows = [t for t in decoded if t.role == "K"]
    for turn in decoded:
        if turn.role == "T":
            incoming = _value(turn.value.kline, turn.value.significance)
        else:
            if len(k_rows[ki].value.kline.nodes) != 0:
                # First S3 row. Its preceding T-row's entry rule settles
                # the work-list — process it now without cogitating.
                if incoming is not None:
                    rationaliser._process_query(incoming)
                break
            rationaliser.rationalise(incoming)
            incoming = None
            ki += 1


def _s3_entry(rationaliser: Rationaliser) -> KLine:
    """Find the S3-pairable opening in the work-list (the entry cogitation
    would dispatch to S3 pairing) — not necessarily the LIFO top, which may be
    an async-pending relationship skipped by cogitation."""
    for entry in reversed(rationaliser._state.work_list):
        if rationaliser._s3_pairable(entry):
            return entry
    raise AssertionError("no S3-pairable entry in work-list")


def test_level1_grouping_emits_canonical_request_for_residual(
    rationaliser: Rationaliser, signifier: NLPSignifier, _decoded_mhall
) -> None:
    """D10 (async model): the 3-vs-1 residual does NOT assert a binding to a
    synthesised signature. The relationship plan pairs Mary<->Subject,
    had<->Verb, then carries a residual [a,little,lamb] for the grouped pair —
    which S3 pairing emits as a canonical request {make_signature(residual):
    residual} at S2 (a hypothesis), not a relationship assertion."""
    _drive_to_s3(rationaliser, _decoded_mhall)
    entry = _s3_entry(rationaliser)
    left_nodes = rationaliser._find_canon_nodes(entry.signature)  # MHALL canon
    right_nodes = rationaliser._find_canon_nodes(entry.nodes[0])  # SVO canon
    plan = rationaliser._relationship_plan(left_nodes, right_nodes)
    # Three pairs; the last carries a non-empty residual (the grouped operands).
    assert len(plan) == 3
    last_lhs, last_rhs, last_residual = plan[-1]
    assert last_residual  # grouped pair -> residual present
    assert last_lhs == signifier.make_signature(last_residual)  # synthesised sig


def test_level1_one_to_one_pairs_carry_no_residual(
    rationaliser: Rationaliser, signifier: NLPSignifier, _decoded_mhall
) -> None:
    """A 1:1 pair (one left operand, one right operand) carries no residual —
    S3 pairing emits it as a CONNOTED relationship at S3. MHALL's first two pairs
    (Mary<->Subject, had<->Verb) are 1:1."""
    _drive_to_s3(rationaliser, _decoded_mhall)
    entry = _s3_entry(rationaliser)
    left_nodes = rationaliser._find_canon_nodes(entry.signature)
    right_nodes = rationaliser._find_canon_nodes(entry.nodes[0])
    plan = rationaliser._relationship_plan(left_nodes, right_nodes)
    for lhs_sig, rhs_node, residual in plan[:2]:  # the 1:1 pairs
        assert residual == []  # no residual -> 1:1 -> CONNOTED at S3


@pytest.mark.skip(
    reason="Coverage gap G1: S2 (non-1:1) proposal branch is built but "
    "unexercised by MHALL"
)
def test_s3_non_one_to_one_is_s2() -> None:
    """A multi-node proposal {S:[n1,n2]} (one signature, multiple nodes) would
    be S2. MHALL produces none; supply a synthetic golden master to cover."""


# ── S2 (Misfit) routing — scripts/dialogue-rationalisation-behaviours.md §3a ─


def test_s2_misfit_routes_to_s2_path_and_idles(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """B1/§3a + §4 rule 1: a multi-node misfit (signature != make_signature(nodes))
    routes to the S2 path. With no grounded kline under its nodes, rule 1
    (node-expansion) fires nothing and the proposal equals the entry shape,
    emitted at S2. The entry persists in the work-list (B1 — no self-close).
    S3 structures continue to take the pairing path (covered by MHALL)."""
    x = 0b1 << 32
    y = 0b10 << 32
    z = 0b100 << 32
    # A multi-node misfit: signature != make_signature(nodes).
    sig = signifier.make_signature([x])           # carries the x type bit only
    nodes = [y, z]                                # make_signature([y,z]) = y|z != sig
    misfit = KLine(sig, nodes)
    assert not is_canon(misfit, signifier)
    assert not rationaliser._s3_pairable(misfit)  # not single-node
    assert rationaliser._s2_eligible(misfit)      # multi-node misfit -> S2

    rationaliser._state.work_list.append(misfit)
    emitted = rationaliser.rationalise(None)

    # Rule 1 fired no expansions (nothing grounded under y/z): proposal is the
    # entry shape, emitted at S2. Entry persists (B1).
    assert len(emitted) == 1
    assert emitted[0].significance == SIG_S2
    assert emitted[0].kline.signature == sig
    assert list(emitted[0].kline.nodes) == [y, z]
    assert (sig, [y, z]) in _work_list(rationaliser)


# ── S2 rule 1 (node-expansion) — behaviours doc §4 ───────────────────────


def test_s2_node_expansion_replaces_signature_node(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§4 rule 1: a node in the entry that is a grounded kline's signature is
    replaced by that kline's nodes. Mirrors the WDMH `#47` shape: entry
    {WDMH:[Mary,had,what]} with grounded {had:[did,have]} expands `had` ->
    [did,have], yielding target [Mary,did,have,what]. Emitted at S2; entry
    persists (B1)."""
    mary = 0b1 << 32
    had = 0b10 << 32
    what = 0b100 << 32
    did = 0b1000 << 32
    have = 0b10000 << 32
    # A genuine misfit: signature carries a type bit none of the nodes have.
    wdmh = signifier.make_signature([mary, had, what, 0b100000 << 32])
    entry = KLine(wdmh, [mary, had, what])
    assert not is_canon(entry, signifier)           # precondition: real misfit
    _ground(rationaliser, KLine(had, [did, have]))   # `had` canon grounded

    rationaliser._state.work_list.append(entry)
    emitted = rationaliser.rationalise(None)

    assert len(emitted) == 1
    assert emitted[0].significance == SIG_S2
    assert emitted[0].kline.signature == wdmh
    # `had` replaced by [did, have]; Mary and what persist.
    assert list(emitted[0].kline.nodes) == [mary, did, have, what]
    # B1: the (unexpanded) entry persists in the work-list.
    assert (wdmh, [mary, had, what]) in _work_list(rationaliser)


def test_s2_node_expansion_skips_identities(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§4 rule 1: a node that is a grounded *identity* (empty nodes) is NOT
    expanded — identities carry no decomposition to substitute with. The node
    persists unchanged in the target."""
    mary = 0b1 << 32
    had = 0b10 << 32
    what = 0b100 << 32
    wdmh = signifier.make_signature([mary, had, what, 0b100000 << 32])
    entry = KLine(wdmh, [mary, had, what])
    assert not is_canon(entry, signifier)
    _ground(rationaliser, KLine(mary, []))           # identity under Mary

    rationaliser._state.work_list.append(entry)
    emitted = rationaliser.rationalise(None)

    # Mary is an identity (no nodes) -> not expanded; target unchanged.
    assert list(emitted[0].kline.nodes) == [mary, had, what]


def test_s2_node_expansion_applies_all_in_one_cogitation(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§4: rule 1 runs every expansion in one cogitation, then emits the
    accumulated target. Two expandable nodes both expand in a single emission."""
    mary = 0b1 << 32
    had = 0b10 << 32
    what = 0b100 << 32
    m1, m2 = 0b1000 << 32, 0b10000 << 32    # Mary decomposes into m1, m2
    w1, w2 = 0b100000 << 32, 0b1000000 << 32  # what decomposes into w1, w2
    wdmh = signifier.make_signature([mary, had, what, 0b10000000 << 32])
    entry = KLine(wdmh, [mary, had, what])
    assert not is_canon(entry, signifier)
    _ground(rationaliser, KLine(mary, [m1, m2]))
    _ground(rationaliser, KLine(what, [w1, w2]))

    rationaliser._state.work_list.append(entry)
    emitted = rationaliser.rationalise(None)

    # Both Mary and what expanded in the one emission; had persists.
    assert list(emitted[0].kline.nodes) == [m1, m2, had, w1, w2]


# ── S2 rule 2 precondition: must_match resolution (§5) ───────────────────


def test_must_match_direct_match_needs_no_resolution(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§5: when every must_match node is directly in the candidate's nodes,
    resolution is trivial — fully matched, no canon lookup."""
    mary = 0b1 << 32
    had = 0b10 << 32
    a = 0b100 << 32
    mhall_nodes = [mary, had, a]
    resolved, fully = rationaliser._resolve_must_match([mary, had], mhall_nodes)
    assert fully is True
    assert set(resolved) == {mary, had}


def test_must_match_resolves_failed_nodes_via_grounded_kline(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§5 worked example: must_match=[Mary,did,have] against MHALL.nodes.
    Mary matches directly; {did,have} fail but resolve via the grounded kline
    {had:[did,have]} -> had, which MHALL has. Result: fully matched, shallower
    must_match [Mary,had].
    """
    mary = 0b1 << 32
    did = 0b10 << 32
    have = 0b100 << 32
    had = 0b1000 << 32                            # distinct from did|have: a misfit
    assert had != signifier.make_signature([did, have])
    a = 0b10000 << 32
    little = 0b100000 << 32
    lamb = 0b1000000 << 32
    _ground(rationaliser, KLine(had, [did, have]))          # grounded (ratified) misfit
    mhall_nodes = [mary, had, a, little, lamb]

    resolved, fully = rationaliser._resolve_must_match([mary, did, have], mhall_nodes)
    assert fully is True
    # Mary stayed; [did,have] resolved to had. Order is not guaranteed by spec;
    # assert as a set.
    assert set(resolved) == {mary, had}


def test_must_match_rejects_when_no_kline_covers_failed(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§5: if the failed set cannot be covered by any grounded kline, the
    candidate is rejected (not fully matched). must_match is returned in its
    best-effort resolved form, but fully=False."""
    mary = 0b1 << 32
    did = 0b10 << 32
    have = 0b100 << 32
    unknown = 0b1000 << 32                         # no kline covers this
    mhall_nodes = [mary]                            # candidate shares only Mary

    resolved, fully = rationaliser._resolve_must_match(
        [mary, did, have, unknown], mhall_nodes
    )
    assert fully is False
    # Mary matched; the rest uncovered (no klines grounded).
    assert unknown in resolved


def test_must_match_resolves_recursively_to_fixed_point(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§5: resolution recurses — a resolved signature may, with another failed
    node, form a new coverable subset. Chain: {b,c}->B grounded, {B,d}->D
    grounded. must_match=[a, b, c, d] against candidate.nodes=[a, D]. First
    pass: a matches; {b,c}->B (failed becomes [B,d]); {B,d}->D. D is in the
    candidate -> fully matched."""
    a = 0b1 << 32
    b = 0b10 << 32
    c = 0b100 << 32
    d = 0b1000 << 32
    bc_sig = signifier.make_signature([b, c])
    d_sig = signifier.make_signature([bc_sig, d])
    _ground(rationaliser, KLine(bc_sig, [b, c]))       # {b,c}->bc_sig
    _ground(rationaliser, KLine(d_sig, [bc_sig, d]))   # {bc_sig,d}->d_sig

    resolved, fully = rationaliser._resolve_must_match([a, b, c, d], [a, d_sig])
    assert fully is True
    assert set(resolved) == {a, d_sig}


def test_must_match_partition_picks_maximal_disjoint_cover(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§5: the partition search is maximal — greedy on size is insufficient.
    Two canons cover overlapping failed sets; the search must pick the disjoint
    combination covering the most nodes. {x,y}->P and {y,z}->Q both subset of
    failed {x,y,z}; they overlap on y so can't both fire. Either alone covers 2;
    the search returns one of them (covered==2), leaving the third node."""
    x = 0b1 << 32
    y = 0b10 << 32
    z = 0b100 << 32
    p_sig = signifier.make_signature([x, y])
    q_sig = signifier.make_signature([y, z])
    _ground(rationaliser, KLine(p_sig, [x, y]))
    _ground(rationaliser, KLine(q_sig, [y, z]))

    resolved, fully = rationaliser._resolve_must_match([x, y, z], [])  # nothing direct
    assert fully is False                            # z or x always uncovered
    # Exactly one canon fired (2 nodes covered); the third node remains.
    covered_sigs = {p_sig, q_sig}
    fired = [n for n in resolved if n in covered_sigs]
    assert len(fired) == 1
    # One of x/z remains uncovered (the one not under the fired canon).
    remaining = [n for n in resolved if n not in covered_sigs]
    assert remaining in ([x], [z])


# ── S2 rule 2 (node-graft) — behaviours doc §4–§5 ────────────────────────


def test_s2_graft_produces_wdmh_48_shape(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§7 worked example, end-to-end: entry {WDMH:[Mary,had,what]} with grounded
    {had:[did,have]} (resolving misfit) and {MHALL:[Mary,had,a,little,lamb]}
    (B3 candidate). Rule 1 expands had->[did,have]; the MHALL graft resolves
    [did,have]->had, coarsens the core to [Mary,had], and grafts [a,little,lamb]
    into the open `what` slot. Result: {WDMH:[Mary,had,a,little,lamb]} at S2
    (the `#48` shape)."""
    mary = 0b1 << 32
    had = 0b10 << 32
    what = 0b100 << 32
    did = 0b1000 << 32
    have = 0b10000 << 32
    a = 0b100000 << 32
    little = 0b1000000 << 32
    lamb = 0b10000000 << 32
    wdmh = signifier.make_signature([mary, had, what, 0b100000000 << 32])
    entry = KLine(wdmh, [mary, had, what])
    assert not is_canon(entry, signifier)
    _ground(rationaliser, KLine(had, [did, have]))               # resolving misfit
    mhall = signifier.make_signature([mary, had, a, little, lamb])
    _ground(rationaliser, KLine(mhall, [mary, had, a, little, lamb]))

    rationaliser._state.work_list.append(entry)
    emitted = rationaliser.rationalise(None)

    assert len(emitted) == 1
    assert emitted[0].kline.signature == wdmh
    assert list(emitted[0].kline.nodes) == [mary, had, a, little, lamb]


def test_s2_graft_replaces_open_with_candidate_difference(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§4: E_open non-empty, C_open non-empty -> REPLACE. Entry [A,B,open1],
    candidate [A,B,X,Y] (shares A,B). Core=[A,B], E_open=[open1], C_open=[X,Y].
    Result: [A,B,X,Y] (open1 replaced by X,Y)."""
    a = 0b1 << 32
    b = 0b10 << 32
    open1 = 0b100 << 32
    x = 0b1000 << 32
    y = 0b10000 << 32
    sig = signifier.make_signature([a, b, open1, 0b100000 << 32])  # misfit
    entry = KLine(sig, [a, b, open1])
    cand_sig = signifier.make_signature([a, b, x, y])
    _ground(rationaliser, KLine(cand_sig, [a, b, x, y]))

    rationaliser._state.work_list.append(entry)
    emitted = rationaliser.rationalise(None)

    assert list(emitted[0].kline.nodes) == [a, b, x, y]


def test_s2_graft_extends_when_target_open_empty(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§4: E_open empty, C_open non-empty -> EXTEND. Entry [A,B] fully resolves
    into candidate [A,B,X]; core=[A,B], E_open=[], C_open=[X]. Result [A,B,X]."""
    a = 0b1 << 32
    b = 0b10 << 32
    x = 0b100 << 32
    sig = signifier.make_signature([a, b, 0b1000 << 32])          # misfit
    entry = KLine(sig, [a, b])
    cand_sig = signifier.make_signature([a, b, x])
    _ground(rationaliser, KLine(cand_sig, [a, b, x]))

    rationaliser._state.work_list.append(entry)
    emitted = rationaliser.rationalise(None)

    assert list(emitted[0].kline.nodes) == [a, b, x]


def test_s2_graft_does_not_fire_without_foothold(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§4/B2: a candidate whose resolution yields an empty core (shares/resolves
    to nothing in the target) does not fire — no foothold means invention.
    Tested directly on _apply_node_graft: a target and a candidate sharing no
    node produce an empty core, so the target is returned unchanged."""
    a = 0b1 << 32
    b = 0b10 << 32
    c = 0b100 << 32
    extra = 0b1000 << 32
    target = [a, b]
    cand = KLine(signifier.make_signature([c, extra]), [c, extra])  # shares nothing
    result = rationaliser._apply_node_graft(target, cand)
    assert result == [a, b]       # unchanged: no foothold, did not fire


def test_s2_drops_already_grounded_proposal(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """B4/§6: if the shaped proposal is isomorphic to a grounded kline, K drops
    it (no emission) and advances. Mirrors the post-ratification advance: after
    T ratifies the #48 shape, K re-shapes it, finds it grounded, emits nothing."""
    mary = 0b1 << 32
    had = 0b10 << 32
    what = 0b100 << 32
    did = 0b1000 << 32
    have = 0b10000 << 32
    a = 0b100000 << 32
    little = 0b1000000 << 32
    lamb = 0b10000000 << 32
    wdmh = signifier.make_signature([mary, had, what, 0b100000000 << 32])
    entry = KLine(wdmh, [mary, had, what])
    _ground(rationaliser, KLine(had, [did, have]))
    mhall = signifier.make_signature([mary, had, a, little, lamb])
    _ground(rationaliser, KLine(mhall, [mary, had, a, little, lamb]))
    # Ground the proposal K would shape (#48 shape) -> B4 will drop it.
    _ground(rationaliser, KLine(wdmh, [mary, had, a, little, lamb]))

    rationaliser._state.work_list.append(entry)
    emitted = rationaliser.rationalise(None)

    assert emitted == []           # shaped proposal is grounded -> dropped (B4)
    assert (wdmh, [mary, had, what]) in _work_list(rationaliser)  # entry persists


# ── S2 no-invention (B2) — behaviours doc §3 ─────────────────────────────


def test_s2_substituted_nodes_all_come_from_grounded_klines(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """B2: every SUBSTITUTED node in the proposal is a node of a grounded kline.
    The entry's own substrate nodes may pass through untouched (received, not
    substituted); only what rule 1 (expansion) and rule 2 (graft) introduce is
    bound by B2, and both source from grounded klines. Verified on the #48
    shape: the substituted nodes ([a, little, lamb] from the MHALL graft) are
    all nodes of the grounded MHALL kline."""
    mary = 0b1 << 32
    had = 0b10 << 32
    what = 0b100 << 32
    did = 0b1000 << 32
    have = 0b10000 << 32
    a = 0b100000 << 32
    little = 0b1000000 << 32
    lamb = 0b10000000 << 32
    wdmh = signifier.make_signature([mary, had, what, 0b100000000 << 32])
    entry = KLine(wdmh, [mary, had, what])
    _ground(rationaliser, KLine(had, [did, have]))
    mhall = signifier.make_signature([mary, had, a, little, lamb])
    mhall_kline = KLine(mhall, [mary, had, a, little, lamb])
    _ground(rationaliser, mhall_kline)

    rationaliser._state.work_list.append(entry)
    emitted = rationaliser.rationalise(None)

    proposal_nodes = set(emitted[0].kline.nodes)
    entry_nodes = set(entry.nodes)
    substituted = proposal_nodes - entry_nodes           # what rules introduced
    grounded_nodes = set(mhall_kline.nodes) | {had}       # nodes of grounded klines
    # Every substituted node is a node of a grounded kline (B2).
    assert substituted <= grounded_nodes


def test_single_node_unpairable_relationship_not_routed_to_s2(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """§3a regression: a single-node relationship whose operand canons are not
    yet seen is S3-structure but not workable. Cogitation must SKIP it (it
    awaits elevation/cleanup), NOT route it to the S2 path — else MHALL's
    pairings stall forever. S2 eligibility requires a multi-node misfit."""
    x = 0b1 << 32
    y = 0b10 << 32
    lhs = signifier.make_signature([x])
    rhs = signifier.make_signature([y])
    rel = KLine(lhs, [rhs])                       # single-node, operands not seen
    assert not rationaliser._s3_pairable(rel)      # not yet workable
    assert not rationaliser._s2_eligible(rel)      # single-node -> NOT S2

    rationaliser._state.work_list.append(rel)
    emitted = rationaliser.rationalise(None)

    # Skipped (not workable), not routed to S2: no emission, entry persists
    # awaiting the operand canons that make it S3-pairable.
    assert emitted == []
    assert (lhs, [rhs]) in _work_list(rationaliser)


# ── S2 candidate admission (B3) — behaviours doc §3 ──────────────────────


def _ground(rationaliser: Rationaliser, kline: KLine) -> None:
    """Ground a kline directly (test helper, mirrors Rationaliser._ground)."""
    rationaliser._ground(kline)


def test_s2_candidates_admit_shared_node_klines(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """B3: a grounded kline sharing >=1 node value with the entry's nodes is
    admitted. Mirrors the WDMH worked example's rule-2 (graft) candidate:
    entry {WDMH:[Mary,had,what]} admits {MHALL:[Mary,had,a,little,lamb]}
    (shares `Mary`,`had` as nodes).

    Note {had:[did,have]} is NOT a B3 candidate — it shares no *node* with the
    entry (`had` is its signature, not its node). It is sourced separately by
    rule 1 (node-expansion) via `signature in entry.nodes`, not via B3.
    """
    mary = 0b1 << 32
    had = 0b10 << 32
    what = 0b100 << 32
    did = 0b1000 << 32
    have = 0b10000 << 32
    a = 0b100000 << 32
    little = 0b1000000 << 32
    lamb = 0b10000000 << 32
    wdmh = signifier.make_signature([mary, had, what])   # entry signature
    mhall = signifier.make_signature([mary, had, a, little, lamb])

    entry = KLine(wdmh, [mary, had, what])
    had_canon = KLine(had, [did, have])                  # NOT a B3 candidate
    mhall_canon = KLine(mhall, [mary, had, a, little, lamb])  # shares Mary,had
    _ground(rationaliser, had_canon)
    _ground(rationaliser, mhall_canon)

    admitted = rationaliser._s2_candidates(entry)
    admitted_sigs = {k.signature for k in admitted}
    assert mhall in admitted_sigs         # {MHALL:[...]} admitted (shared nodes)
    assert had not in admitted_sigs       # {had:[did,have]} NOT admitted (no shared node)


def test_s2_candidates_exclude_identities_and_no_overlap(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """B3: identities (empty nodes) never admit — they have nothing to
    substitute with. A grounded kline sharing no node value with the entry is
    not admitted."""
    mary = 0b1 << 32
    had = 0b10 << 32
    what = 0b100 << 32
    unrelated = 0b1000 << 32
    wdmh = signifier.make_signature([mary, had, what])
    entry = KLine(wdmh, [mary, had, what])

    # An identity under `mary` (empty nodes) — grounded, but no nodes to share.
    _ground(rationaliser, KLine(mary, []))
    # A grounded kline sharing no node with the entry.
    _ground(rationaliser, KLine(unrelated, [unrelated]))

    admitted = rationaliser._s2_candidates(entry)
    assert admitted == []                # neither identity nor no-overlap admits


def test_s2_candidates_exclude_entry_itself(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """B3: the entry itself is not admitted as a candidate (a kline is not its
    own substitution source)."""
    mary = 0b1 << 32
    had = 0b10 << 32
    wdmh = signifier.make_signature([mary, had])
    entry = KLine(wdmh, [mary, had])
    # Even if a kline identical to the entry were grounded, the entry object
    # itself (passed in) is excluded by identity.
    rationaliser._state.work_list.append(entry)
    _ground(rationaliser, KLine(wdmh, [mary, had]))   # same shape, different obj

    admitted = rationaliser._s2_candidates(entry)
    # The grounded same-shape kline IS admitted (it's a distinct grounded kline
    # sharing nodes); only the entry object itself is excluded.
    assert any(k.signature == wdmh for k in admitted)
    assert all(k is not entry for k in admitted)


# ── Termination (D12) ──────────────────────────────────────────────────


def test_termination_returns_empty_when_idle(rationaliser: Rationaliser) -> None:
    """D12: an empty work-list after the entry rule -> empty batch; the runner
    signals termination (termination is the runner's job, not K's)."""
    assert rationaliser.rationalise(None) == []


# ── Escalation (D11) — deferred ───────────────────────────────────────────


@pytest.mark.skip(
    reason="Coverage gap G2: group-size escalation needs a trainer-S4 refusal, "
    "which the stateless SynthesizingTrainer cannot produce. Validated only "
    "against TableTrainer with an authored refusal golden master."
)
def test_escalation_on_trainer_s4_refusal() -> None:
    """D11: a trainer S4 refusal increments group size and restarts the pass."""
