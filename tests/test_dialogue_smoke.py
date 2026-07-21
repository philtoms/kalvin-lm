"""Dialogue sub-project smoke tests.

These tests cover **basic operation only** — that the core path works
(decode a script; run the canonical dialogue end-to-end through the runner).
They deliberately do not pin current behaviour, so the implementation can be
re-explored without a wall of tests failing on every refactor. Add tests as
fresh discoveries demand them.
"""

from __future__ import annotations

import pytest

from kalvin.expand import SIG_S1
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.nlp_tokenizer import NLPTokenizer
from kalvin.signifier import NLPSignifier
from tests._fixtures import mhall_script
from training.dialogue import ScriptTrainee, ScriptTrainer, decode, load_script, run
from training.dialogue.decoder import DecodeError
from training.dialogue.runner import Divergence, RationaliseEvent


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


# A signature guaranteed absent from any real coverage row, so an emission
# of it is an "unmatched" divergence.
_BOGUS_SIG = 0x1BAD1DE


class _OnceDivergingTrainee(ScriptTrainee):
    """A ScriptTrainee that emits one bogus (divergent) event on its first
    reply, then behaves normally. Used to exercise divergence policy."""

    def __init__(self, table, sink):
        super().__init__(table, sink=sink)
        self._spoiled = False

    def next_events(self, incoming):
        if not self._spoiled and incoming:
            self._spoiled = True
            bogus = KValue(KLine(_BOGUS_SIG, []), SIG_S1)
            yield RationaliseEvent(kind="frame", query=bogus, proposal=bogus, role="K")
        yield from super().next_events(incoming)


def test_accept_divergence_continues_the_run(_decoded_mhall):
    """Under on_divergence='accept', a divergent emission is recorded in
    RunResult.unmatched and the run continues (more events follow it). Under
    'fail', the same divergence stops the run by raising."""
    trainer = lambda sink: ScriptTrainer(_decoded_mhall, sink=sink)

    # accept: no raise, the divergent emission is recorded, and the run kept
    # going (events arrived after the divergence).
    res_accept = run(
        _decoded_mhall,
        trainer,
        lambda sink: _OnceDivergingTrainee(_decoded_mhall, sink=sink),
        on_divergence="accept",
    ).run()
    assert any(
        e.proposal.kline.signature == _BOGUS_SIG for e in res_accept.unmatched
    )
    bogus_idx = next(
        i for i, e in enumerate(res_accept.events)
        if e.proposal.kline.signature == _BOGUS_SIG
    )
    assert bogus_idx + 1 < len(res_accept.events)  # run continued past it

    # fail: the same setup raises Divergence.
    with pytest.raises(Divergence):
        run(
            _decoded_mhall,
            trainer,
            lambda sink: _OnceDivergingTrainee(_decoded_mhall, sink=sink),
            on_divergence="fail",
        ).run()
