"""The ASK marker — identity without measurement (compiled ask form).

An ask keeps its canonical signature with ASK_SIG OR-ed in and its canon's
nodes riding along: ``WDMH|ASK_SIG:[what, did, Mary, have]``. The marker
manufactures the ask's distinctiveness from its canon; every atom-space
measurement masks it out.
"""

from __future__ import annotations

import pytest

from kalvin.kline import ASK_SIG, KLine, is_ask, is_canon, is_exact, sig_level
from kalvin.significance import misfit_mass, word_atom_count
from kalvin.signifier import NLPSignifier
from kalvin.hop import candidate_goals
from ks.compiler import compile_source

SCRIPT = """\
(what did Mary have)
WDMH == MHALL =>
  h(ad) => did have
  ALL > O < W
"""


def _asks(entries):
    return [e for e in entries if e.kline.dbg and e.kline.dbg.op == "ASK"]


def test_countersigns_ask_carries_marker_and_canon_nodes(tokenizer):
    entries = compile_source(
        SCRIPT, tokenizer=tokenizer, signifier=NLPSignifier(), dev=True
    )
    (ask,) = _asks(entries)
    assert is_ask(ask.kline.signature)
    assert [n.label for n in ask.kline.nodes] == ["what", "did", "Mary", "have"]
    assert ask.kline.dbg.goal == "MHALL"


def test_ask_is_never_its_canon(tokenizer):
    entries = compile_source(
        SCRIPT, tokenizer=tokenizer, signifier=NLPSignifier(), dev=True
    )
    (ask,) = _asks(entries)
    canon = next(
        e for e in entries
        if e.kline.dbg and e.kline.dbg.label == "WDMH"
        and e.kline.dbg.op == "CANONICALISES" and e.kline.dbg.scope == 1
    )
    sig = NLPSignifier()
    assert ask.kline.signature != canon.kline.signature  # distinct values
    assert not is_canon(ask.kline, sig)  # the question, not the answer
    assert is_canon(canon.kline, sig)
    assert sig_level(ask.kline, sig) == "S4"  # the marker is the ask
    assert sig_level(canon.kline, sig) == "S1"


def test_marker_is_not_an_atom(tokenizer):
    entries = compile_source(
        SCRIPT, tokenizer=tokenizer, signifier=NLPSignifier(), dev=True
    )
    (ask,) = _asks(entries)
    base = int(ask.kline.signature) & ~ASK_SIG
    assert word_atom_count(ask.kline.signature) == word_atom_count(base)
    assert misfit_mass(ask.kline.signature, base) == 0  # no manufactured gap


def test_marker_never_weighs_as_residual(tokenizer):
    entries = compile_source(
        SCRIPT, tokenizer=tokenizer, signifier=NLPSignifier(), dev=True
    )
    (ask,) = _asks(entries)
    sig = NLPSignifier()
    nodes_sig = sig.signature_of(ask.kline.nodes)
    assert is_exact(ask.kline, sig)  # masked: the ask covers its nodes fully
    assert sig.residual(ask.kline.signature, nodes_sig) == 0
    assert sig.residual(nodes_sig, ask.kline.signature) == 0


def test_ask_never_heads_a_goal_list_not_even_via_its_canon():
    sig = NLPSignifier()

    def bit(n):
        return 1 << (32 + n)

    W, D, M, H = bit(0), bit(1), bit(2), bit(3)
    wdmh = W | D | M | H
    ask = KLine(wdmh | ASK_SIG, [W, D, M, H])
    canon_kl = KLine(wdmh, [W, D, M, H])
    held = KLine(H, [D, H])
    # canon released into memory: the ask's own content is not its goal
    goals = candidate_goals([canon_kl, held, ask], ask, sig)
    assert [int(k.signature) for k in goals] == [int(H)]
    # an ask in memory is never a goal for a plain queued kline
    goals = candidate_goals([canon_kl, held, ask], canon_kl, sig)
    assert all(not is_ask(k.signature) for k in goals)
