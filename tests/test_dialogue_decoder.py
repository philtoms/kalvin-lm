"""Phase 1 — dialogue-table decoder tests.

Spec: ``@specs/dialogue-driven-training.md`` DDT-1..5. The decoder is a
single-stage configuration-time function that turns a ``DialogueTable`` into a
flat ordered ``list[DecodedTurn]``.

The canonical acceptance input is ``scripts/dialogue-mhall.json`` (the
"Mary had a little lamb" reference dialogue).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalvin.expand import SIG_S1, SIG_S2
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from training.dialogue.decoder import (
    DecodeError,
    decode,
    load_table,
)

MHALL = Path(__file__).resolve().parent.parent / "scripts" / "dialogue-mhall.json"


@pytest.fixture(scope="module")
def _tok_sig():
    return NLPTokenizer(), NLPSignifier()


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
    assert first.role == "T"
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
        load_table({"script": "x", "turns": [{"role": "X", "op": "IDENTITY"}]})
    with pytest.raises(DecodeError):
        load_table({"script": "x", "turns": [{"role": "T", "op": "BOGUS"}]})



# ── DDT-3: decode() returns a flat ordered list ───────────────────────────


def test_decode_returns_flat_ordered_list(decoded_mhall):
    """DDT-3: ``decode`` yields a flat ordered ``list[DecodedTurn]``."""
    turns, table = decoded_mhall
    assert isinstance(turns, list)
    expected_roles = [t.role for t in table.turns if not t.is_annotation_only]
    assert [t.role for t in turns] == expected_roles


def test_decode_passes_through_actor_op_and_attaches_significance(decoded_mhall):
    """DDT-4: actor/op passed through; significance is band lookup; notes ignored."""
    turns, _ = decoded_mhall
    t0 = turns[0]
    assert t0.role == "T"
    assert t0.op == "COUNTERSIGNED"
    assert t0.value.significance == SIG_S2
    # The opening primary's signature is the MHALL MTS signature (== closing).
    assert t0.value.kline.signature == turns[-1].value.kline.signature
    # Significance is independent of op: a CANONIZED at S2 and one at S1 both
    # appear, distinguished only by the band on the KValue.
    canon_sigs = [t.value.significance for t in turns if t.op == "CANONIZED"]
    assert SIG_S2 in canon_sigs and SIG_S1 in canon_sigs


def test_decode_drops_annotation_only_turns(decoded_mhall):
    """DDT-4: annotation-only turns (notes, no op) are dropped at decode."""
    turns, table = decoded_mhall
    annotation_only = [t for t in table.turns if t.is_annotation_only]
    assert annotation_only, "the canonical table has annotation-only turns"
    assert len(turns) == len(table.turns) - len(annotation_only)


# ── DDT-5: CANONIZED retrieved by node-list match ─────────────────────────


def test_canonized_resolves_to_compiled_canon(decoded_mhall):
    """DDT-5: a CANONIZED turn resolves to the compiled canon kline."""
    turns, _ = decoded_mhall
    mary = next(
        t for t in turns if t.op == "CANONIZED" and t.value.kline.dbg.label == "Mary"
    )
    assert mary.value.kline.dbg.op == "CANONIZED"
    assert len(mary.value.kline.nodes) == 2  # [M, ary]


def test_canonized_retrieved_by_node_list_match(decoded_mhall):
    """DDT-5: the canon's nodes are the compiled decompositions."""
    turns, _ = decoded_mhall
    svo = next(
        t for t in turns if t.op == "CANONIZED" and t.value.kline.dbg.label == "SVO"
    )
    assert len(svo.value.kline.nodes) == 3
    mhall = next(
        t for t in turns if t.op == "CANONIZED" and t.value.kline.dbg.label == "MHALL"
    )
    assert len(mhall.value.kline.nodes) == 5


def test_decode_rejects_canonized_with_no_matching_node_list():
    """DDT-5: a CANONIZED turn whose node-list matches no compiled canon fails."""
    table = load_table(
        {
            "script": "(Mary had a little lamb)\nMHALL == SVO =>",
            "turns": [
                {
                    "role": "T",
                    "op": "CANONIZED",
                    "signature": "Bogus",
                    "nodes": ["Nope"],
                    "significance": "S2",
                }
            ],
        }
    )
    with pytest.raises(DecodeError):
        decode(table, tokenizer=NLPTokenizer(), signifier=NLPSignifier())


# ── Canonical end state ───────────────────────────────────────────────────


def test_canonical_table_opens_on_primary_and_closes_on_countersign(decoded_mhall):
    """The MHALL dialogue opens with the primary half at S2 and closes on K's S1
    countersign of the primary."""
    turns, _ = decoded_mhall
    opening = turns[0]
    closing = turns[-1]
    assert opening.role == "T" and opening.op == "COUNTERSIGNED"
    assert opening.value.significance == SIG_S2
    assert closing.role == "K" and closing.op == "COUNTERSIGNED"
    assert closing.value.significance == SIG_S1
    assert opening.value.kline == closing.value.kline
