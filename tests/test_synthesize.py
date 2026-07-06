"""Unit tests for the synthesis function (plan §Phase 4.1).

Spec seam: ``@specs/dialogue-driven-training.md`` §Actor — the synthesizer
satisfies the existing Actor contract; this plan introduces no new spec IDs.
Each rule (R1 opening, R2 reply-to-identity, R3 echo-compiled) is exercised in
isolation against small synthetic compiled scripts, plus a golden-master-driven
check that the MHALL-compiled script reproduces all 16 trainer turns.

Synthetic scripts build KLines directly with ``KDbg`` so op (CANONIZED /
UNDERSIGNED / CONNOTED) and canon status (``signature == make_signature(nodes)``)
are controlled. ``dbg`` is read by the synthesizer only as an ordering hint
for R2's relation precedence, never for the canon/identity decisions.
"""

from __future__ import annotations

import pytest

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.kline import KDbg, KLine
from kalvin.kvalue import KValue
from kalvin.signifier import NLPSignifier
from tests._fixtures import mhall_table
from training.dialogue.decoder import decode, load_table
from training.dialogue.synthesize import synthesize


@pytest.fixture
def signifier() -> NLPSignifier:
    return NLPSignifier()


def _kline(signifier: NLPSignifier, op: str, nodes: list[int]) -> KLine:
    """A KLine with ``signature == make_signature(nodes)`` (a canon) and op ``op``."""
    sig = signifier.make_signature(nodes)
    return KLine(sig, nodes, dbg=KDbg(op=op, label=op))


def _value(signifier: NLPSignifier, op: str, nodes: list[int]) -> KValue:
    return KValue(_kline(signifier, op, nodes), SIG_S2)


def _identity(signifier: NLPSignifier, token: int) -> KValue:
    """A single-token identity KValue (no decomposition)."""
    return KValue(KLine(token, [], dbg=KDbg(op="IDENTITY")), SIG_S4)


# ── R1 — Opening ─────────────────────────────────────────────────────────


def test_r1_opening_emits_first_compiled_entry_at_s2(
    signifier: NLPSignifier,
) -> None:
    a, b, c = 0x1, 0x2, 0x4
    primary = _kline(signifier, "COUNTERSIGNED", [a, b, c])
    compiled = [KValue(primary, SIG_S2), _value(signifier, "CANONIZED", [a])]

    emitted = synthesize(compiled, None, signifier)

    assert emitted.kline == primary
    assert emitted.significance == SIG_S2


def test_r1_empty_compiled_raises(signifier: NLPSignifier) -> None:
    with pytest.raises(IndexError):
        synthesize([], None, signifier)


# ── R2 — Reply to an identity ────────────────────────────────────────────


def test_r2_canon_outranks_relation(signifier: NLPSignifier) -> None:
    """A signature's canon is emitted over a relation sharing the signature."""
    a, b = 0x1, 0x2
    canon = _kline(signifier, "CANONIZED", [a, b])  # signature == make_signature
    # A relation sharing the canon's signature would need signature==make_signature([a]),
    # which is a *different* signature — so instead build a relation whose signature
    # matches the canon's by giving it the canon's signature with a single node.
    relation = KLine(canon.signature, [a], dbg=KDbg(op="UNDERSIGNED", label="rel"))
    compiled = [KValue(relation, SIG_S3), KValue(canon, SIG_S2)]

    emitted = synthesize(compiled, _identity(signifier, canon.signature), signifier)

    assert emitted.kline == canon


def test_r2_all_leaf_nodes_emit_at_s1(signifier: NLPSignifier) -> None:
    """When every node is a leaf (no decomposition), emit the canon at S1."""
    a, b = 0x1, 0x2
    canon = _kline(signifier, "CANONIZED", [a, b])
    compiled = [KValue(canon, SIG_S2)]

    emitted = synthesize(compiled, _identity(signifier, canon.signature), signifier)

    assert emitted.kline == canon
    assert emitted.significance == SIG_S1


def test_r2_compound_node_emits_at_s2(signifier: NLPSignifier) -> None:
    """When a node is itself decomposable, emit at S2 (the trainee will re-ask)."""
    a, b = 0x1, 0x2
    inner = _kline(signifier, "CANONIZED", [a, b])  # decomposable node
    outer_sig = signifier.make_signature([inner.signature, 0x4])
    outer = KLine(outer_sig, [inner.signature, 0x4], dbg=KDbg(op="CANONIZED", label="outer"))
    compiled = [KValue(inner, SIG_S2), KValue(outer, SIG_S2)]

    emitted = synthesize(compiled, _identity(signifier, outer.signature), signifier)

    assert emitted.kline == outer
    assert emitted.significance == SIG_S2


def test_r2_relation_precedence_underscored_before_connoted(
    signifier: NLPSignifier,
) -> None:
    """Among relations sharing a signature, UNDERSIGNED is picked over CONNOTED.

    Both relations share one signature (built so neither is a canon) — the
    synthesizer orders them by op, not compilation order.
    """
    a, b = 0x1, 0x2
    shared_sig = 0xFF00  # not == make_signature([a]); so neither is a canon
    connoted = KLine(shared_sig, [a], dbg=KDbg(op="CONNOTED", label="c"))
    underscored = KLine(shared_sig, [b], dbg=KDbg(op="UNDERSIGNED", label="u"))
    compiled = [KValue(connoted, SIG_S3), KValue(underscored, SIG_S3)]

    emitted = synthesize(compiled, _identity(signifier, shared_sig), signifier)

    assert emitted.kline == underscored


def test_r2_no_decomposition_is_stalemate_s4(signifier: NLPSignifier) -> None:
    """A signature with no compiled decomposition → identity at S4 (stalemate)."""
    unknown = 0xDEAD
    a, b = 0x1, 0x2
    compiled = [_value(signifier, "CANONIZED", [a, b])]  # unrelated

    emitted = synthesize(compiled, _identity(signifier, unknown), signifier)

    assert emitted.kline == KLine(unknown, [])
    assert emitted.significance == SIG_S4


# ── R3 — Echo a matching compiled kline ──────────────────────────────────


def test_r3_matching_relation_ratified_at_s1(signifier: NLPSignifier) -> None:
    """A proposed relation matching a compiled relation → echoed at S1."""
    a = 0x1
    relation_sig = 0xBEEF  # not == make_signature([a]); a genuine relation
    relation = KLine(relation_sig, [a], dbg=KDbg(op="UNDERSIGNED", label="rel"))
    compiled = [KValue(relation, SIG_S3)]
    proposal = KValue(KLine(relation_sig, [a]), SIG_S3)

    emitted = synthesize(compiled, proposal, signifier)

    assert emitted.kline == relation
    assert emitted.significance == SIG_S1


def test_r3_matching_canon_confirmed_at_s2(signifier: NLPSignifier) -> None:
    """A proposed canon matching a compiled canon → echoed at S2 (confirmed)."""
    a, b, c = 0x1, 0x2, 0x4
    canon = _kline(signifier, "CANONIZED", [a, b, c])  # signature == make_signature
    compiled = [KValue(canon, SIG_S2)]
    proposal = KValue(KLine(canon.signature, [a, b, c]), SIG_S2)

    emitted = synthesize(compiled, proposal, signifier)

    assert emitted.kline == canon
    assert emitted.significance == SIG_S2


def test_r3_no_match_emits_s4(signifier: NLPSignifier) -> None:
    """A proposed kline with no compiled match → CONNOTED,S4."""
    a = 0x1
    compiled = [_value(signifier, "CANONIZED", [a, 0x2])]
    proposal = KValue(KLine(0xCAFE, [a]), SIG_S3)

    emitted = synthesize(compiled, proposal, signifier)

    assert emitted.kline == KLine(0xCAFE, [a])
    assert emitted.significance == SIG_S4


def test_r3_match_is_by_nodes_not_just_signature(signifier: NLPSignifier) -> None:
    """A signature match with different nodes is NOT a match (→ S4)."""
    a, b = 0x1, 0x2
    canon = _kline(signifier, "CANONIZED", [a, b])
    compiled = [KValue(canon, SIG_S2)]
    proposal = KValue(KLine(canon.signature, [a]), SIG_S2)  # same sig, wrong nodes

    emitted = synthesize(compiled, proposal, signifier)

    assert emitted.significance == SIG_S4


# ── Golden master: every MHALL trainer turn reproduces ───────────────────


def test_mhall_trainer_turns_reproduce(signifier: NLPSignifier) -> None:
    """synthesize reproduces all 16 MHALL trainer turns against the compiled script.

    The dialogue table is the validation oracle; the synthesizer reads only the
    compiled script and the preceding trainee turn. This is the rule set's
    end-to-end proof independent of the Actor wrapper (Phase 4.2 adds the
    runner-driven version).
    """
    from ks.compiler import compile_source

    raw = mhall_table()
    table = load_table(raw)
    compiled = compile_source(table.script, tokenizer=None, signifier=signifier, dev=True)
    decoded = decode(table, tokenizer=None, signifier=signifier)

    incoming: KValue | None = None
    mismatches: list[str] = []
    for i, turn in enumerate(decoded):
        if turn.role != "T":
            incoming = turn.value
            continue
        emitted = synthesize(compiled, incoming, signifier)
        expected = turn.value
        if emitted.kline != expected.kline or emitted.significance != expected.significance:
            mismatches.append(
                f"cursor {i}: expected sig={expected.significance:#x}, "
                f"emitted sig={emitted.significance:#x}"
            )
        incoming = expected

    assert not mismatches, "trainer turn mismatches:\n  " + "\n  ".join(mismatches)
