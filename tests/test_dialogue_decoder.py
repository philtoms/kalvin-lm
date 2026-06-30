"""Phase 1 — dialogue-table decoder tests.

Spec: ``@specs/dialogue-driven-training.md`` DDT-1..4, DDT-28. The decoder is a
single-stage configuration-time function that turns a ``DialogueTable`` into a
flat ordered ``list[DecodedTurn]``.

The canonical acceptance input is ``scripts/dialogue-mhall.json`` (the
"Mary had a little lamb" reference dialogue).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from training.dialogue.decoder import (
    DecodeError,
    decode,
    load_table,
)

MHALL = Path(__file__).resolve().parent.parent / "scripts" / "dialogue-mhall.json"

# The decoder's default tokenizer/signifier build is deterministic; a
# session-scoped fixture compiles once for the whole module.
pytestmark = pytest.mark.usefixtures("_tok_sig")


@pytest.fixture(scope="module")
def _tok_sig():
    return NLPTokenizer(), NLPSignifier()


# decode() accepts optional tokenizer/signifier; inject via a small wrapper so
# tests stay declarative. (The signature takes them as kwargs.)
@pytest.fixture(scope="module")
def decoded_mhall(_tok_sig):
    tok, sigf = _tok_sig
    table = load_table(json.loads(MHALL.read_text()))
    return decode(table, tokenizer=tok, signifier=sigf), table


# ── DDT-1: table/turn structure ───────────────────────────────────────────


def test_load_parses_table_and_turn_fields():
    """DDT-1: loader parses ``script`` + ``turns[]``; each turn has the typed fields."""
    table = load_table(json.loads(MHALL.read_text()))
    assert isinstance(table.script, str) and table.script
    assert len(table.turns) >= 2
    first = table.turns[0]
    assert first.actor == "T"
    assert first.op == "COUNTERSIGNED"
    assert first.signature == "MHALL"
    assert first.significance == "S2"
    assert first.nodes == ("SVO",)


def test_load_rejects_malformed_table():
    """DDT-1: a missing script/turns or a bad actor/op is a decode error."""
    with pytest.raises(DecodeError):
        load_table({"turns": []})
    with pytest.raises(DecodeError):
        load_table({"script": "x"})
    with pytest.raises(DecodeError):
        load_table({"script": "x", "turns": [{"actor": "X", "op": "IDENTITY"}]})
    with pytest.raises(DecodeError):
        load_table({"script": "x", "turns": [{"actor": "T", "op": "BOGUS"}]})


# ── DDT-2: decode() returns a flat ordered list ───────────────────────────


def test_decode_returns_flat_ordered_list(decoded_mhall):
    """DDT-2: ``decode`` yields a flat ordered ``list[DecodedTurn]``."""
    turns, table = decoded_mhall
    assert isinstance(turns, list)
    # Order is preserved: the decoded sequence interleaves T/K exactly as the
    # table's structural turns (annotation-only turns removed).
    expected_actors = [t.actor for t in table.turns if not t.is_annotation_only]
    assert [t.actor for t in turns] == expected_actors


# ── DDT-3: single-stage — kline from script + sig lookup + actor/op passthrough ─


def test_decode_passes_through_actor_op_and_attaches_significance(decoded_mhall):
    """DDT-3: actor/op are passed through; significance is the band lookup; notes ignored."""
    turns, table = decoded_mhall
    # Turn 0: T COUNTERSIGNED MHALL:[SVO] at S2.
    t0 = turns[0]
    assert t0.actor == "T"
    assert t0.op == "COUNTERSIGNED"
    assert t0.value.significance == SIG_S2
    # The opening primary's signature is the MHALL MTS signature.
    assert t0.value.kline.signature == turns[-1].value.kline.signature
    # Significance is independent of op (DDT-3): a CANON at S2 and a CANON at S1
    # both appear, distinguished only by the band on the KValue.
    canon_sigs = [t.value.significance for t in turns if t.op == "CANON"]
    assert SIG_S2 in canon_sigs and SIG_S1 in canon_sigs


def test_decode_canon_nodes_are_real_compiled_canons(decoded_mhall):
    """DDT-3/4: a CANON turn resolves to the compiled canon kline (signature + nodes)."""
    turns, _ = decoded_mhall
    # The {Mary:[M,ary]} subword canon at S1.
    mary = next(t for t in turns if t.op == "CANON" and t.value.kline.dbg.label == "Mary")
    assert mary.value.kline.dbg.op == "CANONIZED"  # compiler's aggregate token
    assert len(mary.value.kline.nodes) == 2  # [M, ary]


# ── DDT-4: canons retrieved by node-list match ────────────────────────────


def test_canon_retrieved_by_node_list_match(decoded_mhall):
    """DDT-4: the canon's nodes are the compiled subword/compound decompositions."""
    turns, _ = decoded_mhall
    # SVO canon: nodes are Subject/Verb/Object canonical signatures.
    svo = next(t for t in turns if t.op == "CANON" and t.value.kline.dbg.label == "SVO")
    assert len(svo.value.kline.nodes) == 3
    # MHALL canon: Mary/had/a/little/lamb.
    mhall = next(
        t for t in turns if t.op == "CANON" and t.value.kline.dbg.label == "MHALL"
    )
    assert len(mhall.value.kline.nodes) == 5


def test_decode_rejects_canon_with_no_matching_node_list():
    """DDT-4: a CANON turn whose node-list matches no compiled canon fails loudly."""
    table = load_table(
        {
            "script": "(Mary had a little lamb)\nMHALL == SVO =>",
            "turns": [
                {
                    "actor": "T",
                    "op": "CANON",
                    "signature": "Bogus",
                    "nodes": ["Nope"],
                    "significance": "S2",
                }
            ],
        }
    )
    with pytest.raises(DecodeError):
        decode(table, tokenizer=NLPTokenizer(), signifier=NLPSignifier())


# ── DDT-28: annotation-only turns dropped at decode ───────────────────────


def test_annotation_only_turns_are_dropped(decoded_mhall):
    """DDT-28: turns with notes but no structural fields are dropped at decode time."""
    turns, table = decoded_mhall
    annotation_only = [t for t in table.turns if t.is_annotation_only]
    assert annotation_only, "the canonical table has annotation-only turns"
    # None of them produced a decoded turn.
    assert len(turns) == len(table.turns) - len(annotation_only)


# ── Canonical end state (Phase 1 slice of the acceptance shape) ───────────


def test_canonical_table_opens_on_primary_and_closes_on_countersign(decoded_mhall):
    """The MHALL dialogue opens with the primary half at S2 and closes on K's S1
    countersign of the primary. (Phase 1 resolves the structure; the loop in
    Phase 3 drives and verifies it.)"""
    turns, _ = decoded_mhall
    opening = turns[0]
    closing = turns[-1]
    assert opening.actor == "T" and opening.op == "COUNTERSIGNED"
    assert opening.value.significance == SIG_S2
    assert closing.actor == "K" and closing.op == "COUNTERSIGNED"
    assert closing.value.significance == SIG_S1
    # Opening and closing are the same primary kline.
    assert opening.value.kline == closing.value.kline
