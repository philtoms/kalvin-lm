"""Dialogue sub-project smoke tests.

These tests cover **basic operation only** — that the core path works
(decode a script; run the canonical dialogue end-to-end through the runner).
They deliberately do not pin current behaviour, so the implementation can be
re-explored without a wall of tests failing on every refactor. Add tests as
fresh discoveries demand them.
"""

from __future__ import annotations

import pytest

from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from tests._fixtures import mhall_script
from training.dialogue import ScriptTrainee, ScriptTrainer, decode, load_script, run
from training.dialogue.decoder import DecodeError


@pytest.fixture(scope="module")
def _tok_sig():
    return NLPTokenizer(), NLPSignifier()


# ── decode smoke ──────────────────────────────────────────────────────────


def test_decode_turns_a_script_into_a_flat_ordered_list(_tok_sig):
    """Basic operation: decode() returns one DecodedTurn per structural turn,
    in script order, each carrying its role/op/significance."""
    tok, sigf = _tok_sig
    script = load_script(mhall_script())
    decoded = decode(script, tokenizer=tok, signifier=sigf)
    assert isinstance(decoded, list) and len(decoded) >= 2
    assert {t.role for t in decoded} <= {"T", "K"}
    # Significance was attached (band lookup ran) — distinct bands appear.
    assert len({t.value.significance for t in decoded}) > 1


def test_loader_rejects_a_malformed_script():
    """Basic guard: a script missing script/turns is a decode error."""
    with pytest.raises(DecodeError):
        load_script({"turns": []})
    with pytest.raises(DecodeError):
        load_script({"script": "A"})


# ── the canonical end-to-end run ──────────────────────────────────────────


@pytest.fixture(scope="module")
def _decoded_mhall(_tok_sig):
    tok, sigf = _tok_sig
    return decode(load_script(mhall_script()), tokenizer=tok, signifier=sigf)


def test_canonical_mhall_run_covers_the_exchange(_decoded_mhall):
    """The decisive acceptance test: the MHALL dialogue runs through the runner
    with the default script-reading actors and covers the whole exchange
    (zero displacement). If this passes, the core loop is wired correctly."""
    res = run(
        _decoded_mhall,
        lambda sink: ScriptTrainer(_decoded_mhall, sink=sink),
        lambda sink: ScriptTrainee(_decoded_mhall, sink=sink),
    ).run()
    assert res.uncovered == []  # full coverage
    assert res.events  # something actually ran
