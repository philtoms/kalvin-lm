"""Phase 1 — dialogue-table decoder tests.

Spec: ``@specs/dialogue-driven-training.md`` DDT-1..5. The decoder is a
single-stage configuration-time function that turns a ``DialogueTable`` into a
flat ordered ``list[DecodedTurn]``.

The canonical acceptance input is the "Mary had a little lamb" reference
dialogue, frozen in :mod:`tests._fixtures` (``scripts/dialogue-mhall.json`` is
user-editable and must not be read by tests).
"""

from __future__ import annotations

import pytest

from kalvin.expand import SIG_S1, SIG_S2
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from tests._fixtures import mhall_table
from training.dialogue.decoder import (
    DecodeError,
    decode,
    load_table,
)


@pytest.fixture(scope="module")
def _tok_sig():
    return NLPTokenizer(), NLPSignifier()


@pytest.fixture(scope="module")
def decoded_mhall(_tok_sig):
    tok, sigf = _tok_sig
    table = load_table(mhall_table())
    return decode(table, tokenizer=tok, signifier=sigf), table


# ── DDT-1: table/turn structure ───────────────────────────────────────────


def test_load_parses_table_and_turn_fields():
    """DDT-1: loader parses ``script`` + ``turns[]``; each turn has the typed fields."""
    table = load_table(mhall_table())
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
        load_table({"script": "A"})
    with pytest.raises(DecodeError):
        load_table({"script": "A", "turns": [{"role": "X", "op": "IDENTITY"}]})
    with pytest.raises(DecodeError):
        load_table({"script": "A", "turns": [{"role": "T", "op": "BOGUS"}]})


def test_load_parses_close_marker():
    """A turn may carry ``close: true`` (a script-boundary marker); it parses to ``True``."""
    table = load_table(
        {"script": "A", "turns": [{"role": "K", "op": "IDENTITY", "signature": "a", "significance": "S1", "close": True}]}
    )
    assert table.turns[0].close is True


def test_load_rejects_bad_close_marker():
    """``close`` must be the boolean ``true`` (not 0, not an int, not a string)."""
    for bad in (0, 1, 2, -1, "1", "true", 1.0, False):
        with pytest.raises(DecodeError):
            load_table(
                {"script": "A", "turns": [{"role": "K", "op": "IDENTITY", "signature": "a", "significance": "S1", "close": bad}]}
            )


def test_decode_accepts_close_on_trainer_row():
    """Closing is a runner concern, not a role constraint: a close may sit on
    a trainer (T) row as well as a trainee (K) row."""
    table = mhall_table()
    table = {**table, "turns": list(table["turns"])}
    table["turns"][0] = {**table["turns"][0], "close": True}  # the T opener carries the close
    decoded = decode(load_table(table))
    closers = [t for t in decoded if t.close]
    assert len(closers) == 1
    assert closers[0].close is True
    assert closers[0].role == "T"


def test_decode_accepts_single_close_marker():
    """``close: true`` survives decode on a trainee (K) row (presence-only)."""
    table = mhall_table()
    table = {**table, "turns": list(table["turns"])}
    table["turns"][-1] = {**table["turns"][-1], "close": True}  # the K closer
    decoded = decode(load_table(table))
    closers = [t for t in decoded if t.close]
    assert len(closers) == 1
    assert closers[0].close is True
    assert closers[0].role == "K"


def test_load_prepends_priors_in_list_order(tmp_path):
    """``priors`` names other dialogue-table files; their turns are inserted
    before this table's own turns, in list order (prior[0] first)."""
    import json

    prior_a = tmp_path / "a.json"
    prior_b = tmp_path / "b.json"
    prior_a.write_text(json.dumps({
        "script": "A",
        "turns": [{"role": "T", "op": "IDENTITY", "signature": "a", "significance": "S1", "notes": "prior-a"}],
    }))
    prior_b.write_text(json.dumps({
        "script": "A",
        "turns": [{"role": "K", "op": "IDENTITY", "signature": "a", "significance": "S1", "notes": "prior-b"}],
    }))
    table = load_table({
        "script": "A",
        "priors": [str(prior_a), str(prior_b)],
        "turns": [{"role": "T", "op": "IDENTITY", "signature": "a", "significance": "S1", "notes": "own"}],
    })
    notes = [t.notes for t in table.turns]
    assert notes == ["prior-a", "prior-b", "own"]


def test_load_priors_resolve_recursively(tmp_path):
    """A prior's own ``priors`` compose: prior-b is pulled in via prior-a, so
    its turns precede prior-a's, which precede the table's own."""
    import json

    prior_b = tmp_path / "b.json"
    prior_b.write_text(json.dumps({
        "script": "A",
        "turns": [{"role": "K", "op": "IDENTITY", "signature": "a", "significance": "S1", "notes": "prior-b"}],
    }))
    prior_a = tmp_path / "a.json"
    prior_a.write_text(json.dumps({
        "script": "A",
        "priors": [str(prior_b)],
        "turns": [{"role": "T", "op": "IDENTITY", "signature": "a", "significance": "S1", "notes": "prior-a"}],
    }))
    table = load_table({
        "script": "A",
        "priors": [str(prior_a)],
        "turns": [{"role": "T", "op": "IDENTITY", "signature": "a", "significance": "S1", "notes": "own"}],
    })
    notes = [t.notes for t in table.turns]
    assert notes == ["prior-b", "prior-a", "own"]


def test_load_rejects_missing_prior_file(tmp_path):
    """A prior path that cannot be read is a hard error (no silent skip)."""
    with pytest.raises(DecodeError):
        load_table({
            "script": "A",
            "priors": [str(tmp_path / "missing.json")],
            "turns": [{"role": "T", "op": "IDENTITY", "signature": "a", "significance": "S1"}],
        })


def test_load_rejects_malformed_priors_field():
    """``priors`` must be a list of path strings."""
    for bad in ("scripts/x.json", [1, 2], {"a": 1}):
        with pytest.raises(DecodeError):
            load_table({
                "script": "A",
                "priors": bad,
                "turns": [{"role": "T", "op": "IDENTITY", "signature": "a", "significance": "S1"}],
            })


def test_primaries_from_source_extracts_each_top_level_script():
    """primaries_from_source returns one primary per top-level KScript scope,
    in source order - the klines a multi-script trainer opens (R1) per script."""
    from pathlib import Path

    from training.dialogue.decoder import primaries_from_source

    tok, sigf = NLPTokenizer(), NLPSignifier()
    # The real two-script file: MHALL then WDMH as separate top-level scopes.
    source = Path("data/scripts/mhall.ks").read_text()
    primaries = primaries_from_source(source, tokenizer=tok, signifier=sigf)
    assert len(primaries) == 2
    # Each primary is a distinct compiled kline (MHALL's, then WDMH's).
    assert primaries[0] != primaries[1]



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


# ── DDT-5: CANONIZED resolves nodes, builds the declared signature verbatim ──


def test_canonized_resolves_to_compiled_canon(decoded_mhall):
    """DDT-5: a CANONIZED turn resolves each node label to its canonical signature."""
    turns, _ = decoded_mhall
    mary = next(
        t for t in turns if t.op == "CANONIZED" and t.value.kline.dbg.label == "Mary"
    )
    assert mary.value.kline.dbg.op == "CANONIZED"
    assert len(mary.value.kline.nodes) == 2  # [M, ary]


def test_canonized_nodes_are_compiled_decompositions(decoded_mhall):
    """DDT-5: the kline's nodes are the resolved compiled decompositions."""
    turns, _ = decoded_mhall
    svo = next(
        t for t in turns if t.op == "CANONIZED" and t.value.kline.dbg.label == "SVO"
    )
    assert len(svo.value.kline.nodes) == 3
    mhall = next(
        t for t in turns if t.op == "CANONIZED" and t.value.kline.dbg.label == "MHALL"
    )
    assert len(mhall.value.kline.nodes) == 5


def test_decode_rejects_canonized_with_unknown_node_label():
    """DDT-5: a CANONIZED turn whose node label is not in the script fails.

    The decoder is a resolver, not a gatekeeper: it does not check that the
    declared signature matches the canon its nodes would form (an author may
    declare a deliberate misfit). It only requires every label to resolve.
    """
    table = load_table(
        {
            "script": "(Mary had a little lamb)\nMHALL == SVO =>",
            "turns": [
                {
                    "role": "T",
                    "op": "CANONIZED",
                    "signature": "MHALL",
                    "nodes": ["Nope"],
                    "significance": "S2",
                }
            ],
        }
    )
    with pytest.raises(DecodeError):
        decode(table, tokenizer=NLPTokenizer(), signifier=NLPSignifier())


def test_decode_admits_canonized_signature_misfit():
    """DDT-5: a CANONIZED turn may declare a signature that differs from the
    canon its nodes form — a deliberate misfit (e.g. a K-generated leap).

    The decoder builds the kline as written: declared signature verbatim, nodes
    resolved to their canonical signatures. No consistency check. See
    ``scripts/dialogue-rationalisation-behaviours.md``.
    """
    table = load_table(
        {
            # MHALL canon and WDMH canon both compile; their nodes differ.
            "script": open("data/scripts/mhall.ks").read(),
            "turns": [
                {
                    "role": "K",
                    "op": "CANONIZED",
                    # The signature is the question; the nodes are the answer's
                    # atoms — a signature-changing leap, declared as written.
                    "signature": "WDMH",
                    "nodes": ["Mary", "had", "a", "little", "lamb"],
                    "significance": "S2",
                }
            ],
        }
    )
    turns = decode(table, tokenizer=NLPTokenizer(), signifier=NLPSignifier())
    assert len(turns) == 1
    kline = turns[0].value.kline
    # The declared signature is honoured verbatim (it is WDMH's compiled sig),
    # and the five nodes resolve to the MHALL atoms' canonical signatures.
    assert kline.dbg.label == "WDMH"
    assert len(kline.nodes) == 5


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
