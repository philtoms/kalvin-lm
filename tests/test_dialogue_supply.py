"""Phase 2 — stateless Trainer supply rule tests.

Spec: ``@specs/dialogue-driven-training.md`` DDT-5..14, DDT-19. The Trainer is a
pure function of ``(incoming turn, compiled script)`` — no provenance, no
open-proposal ledger (D1). This module covers the held index, canon-first full
shadowing, node-terminality significance, ratification, the no-op terminal, and
the computed opening.

The decisive acceptance check is that :func:`respond` reproduces **every** T
turn of the canonical table from the K turns alone (the Model A T-side proof,
executed here without the loop).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source
from training.dialogue.decoder import DecodedTurn, decode, load_table
from training.dialogue.supply import (
    SupplyMiss,
    build_held_index,
    opening,
    respond,
    supply,
    terminal_significance,
)

MHALL = Path(__file__).resolve().parent.parent / "scripts" / "dialogue-mhall.json"


@pytest.fixture(scope="module")
def kit():
    tok, sigf = NLPTokenizer(), NLPSignifier()
    table = load_table(json.loads(MHALL.read_text()))
    turns = decode(table, tokenizer=tok, signifier=sigf)
    entries = compile_source(table.script, tokenizer=tok, signifier=sigf, dev=True)
    held = build_held_index(entries)
    return turns, held, table


def _turn_eq(a: DecodedTurn, b: DecodedTurn) -> bool:
    return (
        a.actor == b.actor
        and a.op == b.op
        and a.value.kline == b.value.kline
        and a.value.significance == b.value.significance
    )


# ── DDT-5: the trainer is stateless ───────────────────────────────────────


def test_respond_is_pure(kit):
    """DDT-5: same inputs -> same output; respond() mutates no shared state."""
    turns, held, _ = kit
    # Any K turn: calling respond twice yields equal responses, and the held
    # index is unchanged (frozen dataclass; no mutation surface exists).
    k_turn = next(t for t in turns if t.actor == "K")
    r1 = respond(k_turn, held)
    r2 = respond(k_turn, held)
    assert (r1.turn == r2.turn) if (r1.turn and r2.turn) else (r1.turn is r2.turn)
    # A second distinct K turn does not depend on the first having been called.
    other = next(t for t in turns if t.actor == "K" and t is not k_turn)
    assert respond(other, held) == respond(other, held)


# ── DDT-7: computed opening (primary half at S2) ──────────────────────────


def test_opening_is_primary_half_at_S2(kit):
    """DDT-7: opening() computes the primary ``==`` half at S2, not read from table."""
    turns, held, _ = kit
    primary = turns[0].value  # the canonical opening is the primary COUNTERSIGNED at S2
    computed = opening(primary, held)
    assert computed.actor == "T"
    assert computed.op == "COUNTERSIGNED"
    assert computed.value.significance == SIG_S2
    assert _turn_eq(computed, turns[0])


# ── DDT-8/9/11: canon-first supply with full shadowing ────────────────────


def test_supply_returns_canon_for_compound_label(kit):
    """DDT-8: a request for a compound label supplies its canon (canon-first)."""
    _, held, _ = kit
    kv = supply("Mary", held)
    assert kv.kline.dbg.op == "CANONIZED"  # the canon, not the relation


def test_shadowed_relation_is_never_supplied(kit):
    """DDT-9: a relation whose label carries a canon is never supplied (full shadowing)."""
    _, held, _ = kit
    # Mary has both a canon and an UNDERSIGNED relation; supply must return the canon.
    kv = supply("Mary", held)
    assert kv.kline.dbg.op == "CANONIZED"
    assert kv.kline.nodes == [0x400000004D, 0x8000000005F4]  # [M, ary] — the canon decomposition


def test_relation_with_no_canon_is_supplied(kit):
    """DDT-11: a label with a relation but no canon is supplied (not shadowed)."""
    _, held, _ = kit
    # 'a' has a CONNOTED relation but no canon -> supplied (the {a:[Det]} relation).
    kv = supply("a", held)
    assert kv.kline.dbg.op == "CONNOTED"


def test_supply_miss_for_unknown_label(kit):
    """DDT-20: a request with no held kline raises SupplyMiss (escalation path)."""
    _, held, _ = kit
    with pytest.raises(SupplyMiss):
        supply("Nonexistent", held)


# ── DDT-12: node-terminality significance ─────────────────────────────────


def test_terminal_nodes_get_S1(kit):
    """DDT-12: supplied entries with terminal (atomic) nodes are marked S1."""
    _, held, _ = kit
    # Mary canon: [M, ary] atoms -> terminal -> S1.
    assert terminal_significance(supply("Mary", held), held) == SIG_S1


def test_nonterminal_nodes_get_S2(kit):
    """DDT-12: supplied entries with non-terminal (compound) nodes are marked S2."""
    _, held, _ = kit
    # MHALL canon: [Mary, had, a, little, lamb] compounds -> non-terminal -> S2.
    assert terminal_significance(supply("MHALL", held), held) == SIG_S2


# ── DDT-13: ratification; DDT-14: terminal no-op ──────────────────────────


def test_ratify_responds_with_reciprocal_at_S1(kit):
    """DDT-13: on a K S3 proposal, respond() ratifies with the reciprocal at S1."""
    turns, held, _ = kit
    proposal = next(t for t in turns if t.actor == "K" and t.value.significance == SIG_S3)
    resp = respond(proposal, held)
    assert resp.turn is not None
    assert resp.turn.actor == "T"
    assert resp.turn.value.significance == SIG_S1


def test_terminal_K_turn_is_a_noop(kit):
    """DDT-14: on a K S1 terminal, respond() takes no action (advances)."""
    turns, held, _ = kit
    terminal = next(t for t in turns if t.actor == "K" and t.value.significance == SIG_S1)
    resp = respond(terminal, held)
    assert resp.turn is None and resp.miss is None


# ── DDT-19: subword canons withheld & supplied ────────────────────────────


def test_subword_canon_is_withheld_and_supplied(kit):
    """DDT-19: subword canons are not filtered — they are withheld and supplied."""
    _, held, _ = kit
    # 'Det' is a subword canon (Det => D, et); it is withheld and supplied on request.
    kv = supply("Det", held)
    assert kv.kline.dbg.op == "CANONIZED"
    assert terminal_significance(kv, held) == SIG_S1  # [D, et] atoms -> terminal


# ── Decisive acceptance: respond() reproduces every T turn (Model A T-side) ─


def test_respond_reproduces_every_table_T_turn(kit):
    """Every non-opening T turn is reproduced by respond() from the preceding K
    turn alone; the opening is reproduced by opening(). This is the Model A
    T-side proof (executed without the loop): a mismatch would be a
    supply-function bug (DDT-25)."""
    turns, held, _ = kit
    assert _turn_eq(opening(turns[0].value, held), turns[0])

    mismatches = 0
    checked = 0
    for i, t in enumerate(turns):
        if t.actor != "K":
            continue
        nxt = next(
            (turns[j] for j in range(i + 1, len(turns)) if turns[j].actor == "T"),
            None,
        )
        if nxt is None:
            continue
        checked += 1
        resp = respond(t, held)
        if resp.turn is None or not _turn_eq(resp.turn, nxt):
            mismatches += 1
    assert checked > 0
    assert mismatches == 0
