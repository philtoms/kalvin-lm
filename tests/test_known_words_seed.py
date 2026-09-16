"""Cross-script word binding: the prior state's words seed a compile.

An underfit question script (its word list holds only the question's
words) cannot bind the answer compound's chars — they mint as fresh
char-words and the two scripts' decompositions drift apart. The seeded
known_words list resolves them: first-letter matching with the
occurrence counter, exactly the script's own word-list tier, searched
after the script's own lists.
"""

from __future__ import annotations

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.signifier import NLPSignifier
from ks.compiler import compile_source

WDMH_SCRIPT = """(what did Mary have)
WDMH == MHALL =>
  h(ad) => did have
  W = O(bject)
  ALL = O(bject)
"""

KNOWN = ["Mary", "had", "a", "little", "lamb", "Subject", "Verb", "Object"]


def _mhall_mts(entries):
    return next(
        e.kline for e in entries
        if e.kline.dbg and e.kline.dbg.label == "MHALL"
        and e.kline.dbg.op == "CANONICALISES"
        and len(e.kline.nodes) == 5
    )


def test_unseeded_chars_mint_fresh():
    sig = NLPSignifier()
    entries = compile_source(WDMH_SCRIPT, tokenizer=BPETokenizer(),
                             signifier=sig, dev=True)
    mhall = _mhall_mts(entries)
    # A and L unbound: char nodes, not the answer's words
    assert [n.label for n in mhall.nodes] == ["Mary", "had", "A", "L", "L"]


def test_seeded_words_bind_the_answer_chars():
    sig = NLPSignifier()
    tok = BPETokenizer()
    word_bits: dict[str, int] = {}
    from pathlib import Path
    first = compile_source(
        Path("data/scripts/mhall.ks").read_text(), tokenizer=tok,
        signifier=sig, dev=True, word_bits=word_bits,
    )
    entries = compile_source(WDMH_SCRIPT, tokenizer=tok, signifier=sig,
                             dev=True, word_bits=word_bits,
                             known_words=list(word_bits))
    mhall = _mhall_mts(entries)
    # the occurrence counter walks little then lamb; the script's own
    # words still bind W/D/M/H ahead of the seed
    assert [n.label for n in mhall.nodes] == [
        "Mary", "had", "a", "little", "lamb"
    ]
    # value continuity: the seeded compile's MHALL equals the answer
    # script's MHALL (the drift is gone)
    answer = _mhall_mts(first)
    assert int(mhall.signature) == int(answer.signature)
