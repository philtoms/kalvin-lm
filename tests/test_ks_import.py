"""``import <module>`` — one script compiles under another's entries and bindings.

The module's compiled entries prepend in import order (its encounter
precedes the script's own), and — because it emits through the same root
binding frame — its word lists seed the importing script exactly as
known_words do: the script's own lists are searched first and always win.
Compound canons register once across the boundary, so a reference reuses
the module's decomposition and signature value (no per-compile drift).
"""

from __future__ import annotations

import pytest

from ks.compiler import CompileError, compile_source, path_resolver
from ks.lexer import Lexer
from ks.parser import ParseError, Parser

from tests.conftest import requires_tokenizer_data

pytestmark = requires_tokenizer_data

MHALL = """(Mary had a little lamb)
MHALL == SVO =>
   Subject < M
   Verb < H
   Object < Query < ALL =>
     A = Det
     L = Mod
     L = O
"""

USER_SCRIPT = """import mhall
(what Object)
W = O

(what did Mary have)
WDMH == MHALL =>
  h(ad) => did have
  ALL = O(bject)
"""


def _compile(tmp_path, source, word_bits=None):
    from kalvin.bpe_tokenizer import BPETokenizer
    from kalvin.signifier import NLPSignifier

    (tmp_path / "mhall.ks").write_text(MHALL)
    return compile_source(
        source, tokenizer=BPETokenizer(), signifier=NLPSignifier(),
        dev=True, word_bits=word_bits, resolver=path_resolver(tmp_path),
    )


def _mhall_canon(entries):
    return next(
        e.kline for e in entries
        if e.kline.dbg and e.kline.dbg.label == "MHALL"
        and e.kline.dbg.op == "CANONICALISES"
        and len(e.kline.nodes) == 5
    )


# ── Parsing ───────────────────────────────────────────────────────────


def test_import_parses_as_directive():
    file = Parser(Lexer("import mhall\nA = B").tokenize()).parse()
    from ks.ast import Import

    assert isinstance(file.constructs[0], Import)
    assert file.constructs[0].module == "mhall"
    assert len(file.constructs) == 2  # the import does not shadow `A = B`


def test_import_after_construct_is_parse_error():
    with pytest.raises(ParseError, match="precede"):
        Parser(Lexer("A = B\nimport mhall").tokenize()).parse()


def test_nested_import_is_parse_error():
    with pytest.raises(ParseError, match="precede"):
        Parser(Lexer("A =>\n  import mhall").tokenize()).parse()


def test_import_requires_module_name():
    with pytest.raises(ParseError, match="module name"):
        Parser(Lexer("import\nA = B").tokenize()).parse()


def test_import_rejects_operators():
    with pytest.raises(ParseError, match="bare module name"):
        Parser(Lexer("import mhall => S").tokenize()).parse()


def test_import_word_stays_a_word_in_items():
    file = Parser(Lexer("(w import)\nW > import").tokenize()).parse()
    assert not any(
        getattr(c, "module", None) for c in file.constructs
    )


# ── Compilation ───────────────────────────────────────────────────────


def test_import_prefixes_module_entries(tmp_path):
    entries = _compile(tmp_path, "import mhall\nW = O")
    labels = [e.kline.dbg.label for e in entries if e.kline.dbg]
    # mhall's encounter comes first; the script's own entry follows it
    assert labels.index("MHALL") < labels.index("W")
    assert labels[0] == "MHALL"  # the module's opening ask leads the run
    assert not any(l == "import" for l in labels)  # no ask for the keyword


def test_import_seeds_word_binding_and_values(tmp_path):
    standalone_bits: dict[str, int] = {}
    standalone = compile_source(MHALL, dev=True, word_bits=standalone_bits)
    imported = _compile(tmp_path, USER_SCRIPT, word_bits={})
    canon = _mhall_canon(imported)
    # the module's decomposition is reused: no fresh char-words
    assert [n.label for n in canon.nodes] == ["Mary", "had", "a", "little", "lamb"]
    # value continuity: the imported MHALL equals the standalone module's
    assert int(canon.signature) == int(_mhall_canon(standalone).signature)


def test_import_seeds_binding_for_script_chars(tmp_path):
    # no script word list: the chars resolve through the module's words,
    # the occurrence counter walking L to little then lamb (S itself is
    # bound to the module's Subject by its outlived inline binding)
    entries = _compile(tmp_path, "import mhall\nS = L\nP = L")
    denotes = {
        e.kline.dbg.label: [n.label for n in e.kline.nodes]
        for e in entries if e.kline.dbg and e.kline.dbg.op == "DENOTES"
    }
    assert denotes["Subject"] == ["little"]
    assert denotes["P"] == ["lamb"]


def test_goal_wiring_survives_import(tmp_path):
    entries = _compile(tmp_path, USER_SCRIPT)
    ask = next(
        e.kline for e in entries
        if e.kline.dbg and e.kline.dbg.label == "WDMH" and e.kline.dbg.op == "ASK"
    )
    assert ask.dbg.goal == "MHALL"
    # the goal kline is among the compiled entries (the module's canon)
    assert _mhall_canon(entries) is not None


def test_script_lists_win_over_module_words(tmp_path):
    entries = _compile(tmp_path, "import mhall\n(wrong widget)\nW = O")
    labels = [e.kline.dbg.label for e in entries if e.kline.dbg]
    # the script's own [wrong, widget] binds W before the module's words
    assert "wrong" in labels


def test_duplicate_import_emits_once(tmp_path):
    once = _compile(tmp_path, "import mhall\nW = O")
    twice = _compile(tmp_path, "import mhall\nimport mhall\nW = O")
    assert len(twice) == len(once)


def test_nested_import_depth_first(tmp_path):
    from kalvin.bpe_tokenizer import BPETokenizer
    from kalvin.signifier import NLPSignifier

    (tmp_path / "mhall.ks").write_text(MHALL)
    (tmp_path / "inner.ks").write_text("I = N\n")
    (tmp_path / "outer.ks").write_text("import inner\nC = E\n")
    entries = compile_source(
        "import outer\nimport mhall\nF = G",
        tokenizer=BPETokenizer(), signifier=NLPSignifier(),
        dev=True, resolver=path_resolver(tmp_path),
    )
    labels = [e.kline.dbg.label for e in entries if e.kline.dbg]
    assert labels.index("I") < labels.index("C") < labels.index("MHALL") < labels.index("F")


def test_diamond_import_emits_once(tmp_path):
    from kalvin.bpe_tokenizer import BPETokenizer
    from kalvin.signifier import NLPSignifier

    (tmp_path / "mhall.ks").write_text(MHALL)
    (tmp_path / "left.ks").write_text("import mhall\nL = M\n")
    (tmp_path / "right.ks").write_text("import mhall\nR = M\n")
    entries = compile_source(
        "import left\nimport right\nQ = S",
        tokenizer=BPETokenizer(), signifier=NLPSignifier(),
        dev=True, resolver=path_resolver(tmp_path),
    )
    canons = [e for e in entries if e.kline.dbg and e.kline.dbg.label == "MHALL"
              and e.kline.dbg.op == "CANONICALISES" and e.kline.nodes]
    assert len(canons) == 1  # mhall emits once despite two paths to it


def test_circular_import_raises(tmp_path):
    from kalvin.bpe_tokenizer import BPETokenizer
    from kalvin.signifier import NLPSignifier

    (tmp_path / "mhall.ks").write_text("import loop\nL = M\n")
    (tmp_path / "loop.ks").write_text("import mhall\nP = Q\n")
    with pytest.raises(CompileError, match="circular"):
        compile_source(
            "import mhall\nX = Y",
            tokenizer=BPETokenizer(), signifier=NLPSignifier(),
            dev=True, resolver=path_resolver(tmp_path),
        )


def test_import_without_resolver_raises(tmp_path):
    from kalvin.bpe_tokenizer import BPETokenizer

    with pytest.raises(CompileError, match="resolver"):
        compile_source("import mhall\nW = O", tokenizer=BPETokenizer(), dev=True)


def test_unresolvable_module_raises(tmp_path):
    from kalvin.bpe_tokenizer import BPETokenizer

    with pytest.raises(CompileError, match="cannot resolve"):
        compile_source(
            "import nosuch\nW = O", tokenizer=BPETokenizer(),
            dev=True, resolver=path_resolver(tmp_path),
        )
