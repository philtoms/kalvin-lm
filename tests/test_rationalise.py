"""Unit tests for the Rationalising Trainee (plan §Phase 2.1).

Spec seam: ``@specs/dialogue-driven-training.md`` §Actor — the Rationaliser
satisfies the existing Actor contract; this plan introduces no new spec IDs.
These tests exercise each mechanism branch in isolation (entry rule, Level 0,
Level 1, cleanup, grouping, MTS discrimination, termination), not only through
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
from kalvin.kline import KLine
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


# ── Level 0 (Identity) ────────────────────────────────────────────────────


def test_level0_emits_identity_at_s4(
    rationaliser: Rationaliser, signifier: NLPSignifier
) -> None:
    """Level 0: an unrecognised signature's identity is emitted at S4.

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


# ── Level 1 (Relationships) — grouping (D10) ──────────────────────────────


@pytest.fixture(scope="module")
def _decoded_mhall():
    table = load_table(mhall_table())
    return decode(table, signifier=NLPSignifier())


def _drive_to_level1(rationaliser: Rationaliser, decoded) -> None:
    """Drive the rationaliser through the identity phase until the work-list has
    settled to the opening relationship (Level-1 boundary).

    Mirrors the working end-to-end simulation: T-rows become incoming, K-rows
    trigger respond(incoming). The first Level-1 K-row is NOT consumed, but the
    T-row preceding it (which grounds the last operand and triggers the
    cleanup reducing the work-list) is processed via its entry rule so the
    work-list settles to the opening relationship ready for Level 1.
    """
    incoming = None
    ki = 0
    k_rows = [t for t in decoded if t.role == "K"]
    for turn in decoded:
        if turn.role == "T":
            incoming = _value(turn.value.kline, turn.value.significance)
        else:
            if len(k_rows[ki].value.kline.nodes) != 0:
                # First Level-1 row. Its preceding T-row's entry rule settles
                # the work-list — process it now without cogitating.
                if incoming is not None:
                    rationaliser._process_query(incoming)
                break
            rationaliser.rationalise(incoming)
            incoming = None
            ki += 1


def _level1_entry(rationaliser: Rationaliser) -> KLine:
    """Find the Level-1-eligible opening in the work-list (the entry cogitation
    would dispatch to Level 1) — not necessarily the LIFO top, which may be an
    async-pending relationship skipped by cogitation."""
    for entry in reversed(rationaliser._state.work_list):
        if rationaliser._level1_eligible(entry):
            return entry
    raise AssertionError("no Level-1-eligible entry in work-list")


def test_level1_grouping_emits_canonical_request_for_residual(
    rationaliser: Rationaliser, signifier: NLPSignifier, _decoded_mhall
) -> None:
    """D10 (async model): the 3-vs-1 residual does NOT assert a binding to a
    synthesised signature. The relationship plan pairs Mary<->Subject,
    had<->Verb, then carries a residual [a,little,lamb] for the grouped pair —
    which Level 1 emits as a canonical request {make_signature(residual):
    residual} at S2 (a hypothesis), not a relationship assertion."""
    _drive_to_level1(rationaliser, _decoded_mhall)
    entry = _level1_entry(rationaliser)
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
    Level 1 emits it as a CONNOTED relationship at S3. MHALL's first two pairs
    (Mary<->Subject, had<->Verb) are 1:1."""
    _drive_to_level1(rationaliser, _decoded_mhall)
    entry = _level1_entry(rationaliser)
    left_nodes = rationaliser._find_canon_nodes(entry.signature)
    right_nodes = rationaliser._find_canon_nodes(entry.nodes[0])
    plan = rationaliser._relationship_plan(left_nodes, right_nodes)
    for lhs_sig, rhs_node, residual in plan[:2]:  # the 1:1 pairs
        assert residual == []  # no residual -> 1:1 -> CONNOTED at S3


@pytest.mark.skip(
    reason="Coverage gap G1: S2 (non-1:1) proposal branch is built but "
    "unexercised by MHALL"
)
def test_level1_non_one_to_one_is_s2() -> None:
    """A multi-node proposal {S:[n1,n2]} (one signature, multiple nodes) would
    be S2. MHALL produces none; supply a synthetic golden master to cover."""


# ── Termination (D12) ─────────────────────────────────────────────────────


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
