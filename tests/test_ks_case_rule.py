"""Case-rule regressions: word expansion and uppercase-only attraction.

An identifier's case frames its reading (CONTEXT.md, Word Binding):
ALL-UPPER is a compound (MHALL), Capitalized carries its own expansion
(``Mood`` ≡ ``M(ood)``; an explicit annotation suppresses it), and
lowercase-first is a literal word (``had``). Ambient attraction applies
only to bare single chars in sig case — a lowercase single char is the
literal word (the article ``a``) and never attracts.
"""

from __future__ import annotations

import pytest

from ks.compiler import compile_source
from ks.parser import ParseError


def _decoded(source: str) -> list[str]:
    """The compiled klines' decoded forms (op + sig:[nodes])."""
    return [
        f"{kv.kline.dbg.op} | {kv.kline.dbg.decoded}"
        for kv in compile_source(source, dev=True)
    ]


def _klines(source: str) -> list:
    return [kv.kline for kv in compile_source(source, dev=True)]


class TestWordExpansion:
    """Capitalized multi-char words expand as Initial(tail)."""

    def test_bracketless_equivalent_to_bracketed(self):
        # The klines — values included — must be identical.
        assert _klines("Mood = Happy\n") == _klines("M(ood) = H(appy)\n")

    def test_expansion_binds_the_initial(self):
        # After Mood expands, a bare M resolves to the word. The
        # statements end in Happy and X — words with no identity/canon of
        # their own — so each also mints its own S4 unknown.
        assert _decoded("Mood = Happy\nM = X\n") == [
            "DENOTES | Mood:[Happy]",
            "UNKNOWN | Happy:[]",
            "DENOTES | Mood:[X]",
            "UNKNOWN | X:[]",
        ]

    def test_multichar_annotation_is_an_error(self):
        # An inline annotation attaches to a single character's tail —
        # Word(x) is an authoring error (once a silent literal sig-side,
        # a mangled concatenation node-side).
        with pytest.raises(ParseError):
            compile_source("Subject(x) = Y\n", dev=True)
        with pytest.raises(ParseError):
            compile_source("A = had(x)\n", dev=True)

    def test_single_char_annotation_still_attaches(self):
        assert _decoded("M(2) = X\n") == ["DENOTES | M2:[X]", "UNKNOWN | X:[]"]

    def test_whitespace_tail_is_an_error(self):
        # A spaced "word" can never match a list word (lists are
        # whitespace-split) — phrasal content belongs in a prefix annotation.
        with pytest.raises(ParseError):
            compile_source("M(ary had) = X\n", dev=True)
        with pytest.raises(ParseError):
            compile_source("A = M(ary\nhad)\n", dev=True)

    def test_all_upper_stays_compound(self):
        assert "CANONICALISES | MHALL:[M, H, A, L, L]" in _decoded("MHALL = X\n")

    def test_lowercase_stays_literal_word(self):
        # A lowercase node is the word itself, never an expansion. The
        # terminal word 'had' has no identity of its own → its own S4.
        assert _decoded("A = had\n") == ["DENOTES | A:[had]", "UNKNOWN | had:[]"]

    def test_bare_capitalized_word_is_identity(self):
        # A self-carried word is a known word: identity, not the ask.
        assert _decoded("Mood\n") == ["IDENTITY | Mood:[Mood]"]

    def test_expansion_patches_parent_mts_canon(self):
        # The user's Mary script, bracketless: Subject/Object expansions
        # bind S and O, patching the SVO canon and the bare O node.
        src = (
            "(Mary had a little lamb)\n"
            "MHALL == SVO =>\n"
            "  Subject < M\n"
            "  Object < Query < ALL =>\n"
            "    A = Det\n"
            "    L = Mod\n"
            "    L = O\n"
        )
        out = _decoded(src)
        assert "CANONICALISES | SVO:[Subject, V, Object]" in out
        assert "DENOTES | lamb:[Object]" in out


class TestSigCaseAttraction:
    """Ambient attraction is uppercase-only; lowercase chars are literal."""

    def test_uppercase_attracts_case_insensitively_on_word_side(self):
        assert _decoded("(mary had a little lamb)\nM = X\n") == [
            "DENOTES | mary:[X]",
            "UNKNOWN | X:[]",
        ]

    def test_lowercase_single_char_never_attracts(self):
        # The article 'a' must not rebind to a hostile (axe) word list.
        assert _decoded("(a little lamb)\n(axe)\nL = a\n") == [
            "ASK | ALL:[a, little, lamb]",
            "DENOTES | little:[a]",
            "IDENTITY | a:[a]",
            "IDENTITY | little:[little]",
            "IDENTITY | lamb:[lamb]",
        ]

    def test_lowercase_single_char_unbound_stays_literal(self):
        assert _decoded("(axe)\nL = a\n") == ["DENOTES | L:[a]", "UNKNOWN | a:[]"]

    def test_occurrence_counter_still_disambiguates(self):
        # L attracts little then lamb; A attracts the article a; B stays raw.
        # Each statement's terminal has no identity of its own → its own S4.
        assert _decoded("(Mary had a little lamb)\nL = A\nL = B\n") == [
            "DENOTES | little:[a]",
            "UNKNOWN | a:[]",
            "DENOTES | lamb:[B]",
            "UNKNOWN | B:[]",
        ]

    def test_authored_witness_fires_regardless_of_position(self):
        # h(ad) means 'had' whether top-level or inside a subscript —
        # the lowercase gate removed the ambient 'have' that used to
        # block the top-level form. The terminal 'have' has no identity
        # here → its own S4.
        assert _decoded("(what did Mary have)\nh(ad) => did have\n") == [
            "CANONICALISES | had:[did, have]",
            "UNKNOWN | have:[]",
        ]

    def test_authored_witness_binds_compound_char(self):
        # The witness doctrine: h(ad) binds the compound char H.
        out = _decoded("(what did Mary have)\nWDMH == MHALL =>\n  h(ad) => did have\n  ALL = O(bject)\n")
        assert "CANONICALISES | MHALL:[Mary, had, A, L, L]" in out


class TestTerminalUnknowns:
    """A statement terminal with no identity/canon of its own → its own S4."""

    MHALL = """(Mary had a little lamb) == SVO =>
   Subject < M
   Verb < H
   Object < Query < ALL =>
     A = Det
     L = Mod
     L = O
"""

    def test_mhall_terminals_mint_s4_unknowns(self):
        out = _decoded(self.MHALL)
        # Det, Mod, O(→Object) end their statements undefined — each mints
        # its own empty unknown, placed right after its statement.
        assert out == [
            "ASK | MHALL:[Mary, had, a, little, lamb]",
            "CONNOTES | MarySubject:[Subject]",
            "CONNOTES | hadVerb:[Verb]",
            "CONNOTES | QueryObject:[Object]",
            "CONNOTES | ALLQuery:[Query]",
            "DENOTES | a:[Det]",
            "UNKNOWN | Det:[]",
            "DENOTES | little:[Mod]",
            "UNKNOWN | Mod:[]",
            "DENOTES | lamb:[Object]",
            "UNKNOWN | Object:[]",
            "IDENTITY | Mary:[Mary]",
            "IDENTITY | had:[had]",
            "IDENTITY | a:[a]",
            "IDENTITY | little:[little]",
            "IDENTITY | lamb:[lamb]",
            "CANONICALISES | SVO:[Subject, Verb, Object]",
            "CANONICALISES | ALL:[a, little, lamb]",
        ]

    def test_resolved_terminals_mint_nothing(self):
        # In mhall only Det, Mod, Object mint unknowns — M→Mary and H→had
        # resolve to text words whose identities the ask expansion emits.
        out = _decoded(self.MHALL)
        assert [l for l in out if l.startswith("UNKNOWN")] == [
            "UNKNOWN | Det:[]",
            "UNKNOWN | Mod:[]",
            "UNKNOWN | Object:[]",
        ]

    def test_unknowns_are_additive_and_fed_s4(self):
        from kalvin.significance import SIG_S4

        vals = compile_source(self.MHALL, dev=True)
        unknowns = [kv for kv in vals if kv.kline.dbg.op == "UNKNOWN"]
        assert [kv.kline.dbg.decoded for kv in unknowns] == [
            "Det:[]", "Mod:[]", "Object:[]"
        ]
        for kv in unknowns:
            assert kv.kline.nodes == []
            assert kv.significance == SIG_S4
        # The denotations/connotations are untouched.
        assert "DENOTES | a:[Det]" in _decoded(self.MHALL)

    def test_canon_exact_ask_fed_s1(self):
        # The MHALL ask is a full canon — its fed band is S1 (content),
        # not the ask's S4 marker band.
        from kalvin.significance import SIG_S1

        vals = compile_source(self.MHALL)
        asks = [kv for kv in vals if kv.kline.dbg.op == "ASK"]
        assert [kv.significance for kv in asks] == [SIG_S1]

    def test_one_unknown_per_word(self):
        out = _decoded("A = Q\nB = Q\n")
        assert out == [
            "DENOTES | A:[Q]",
            "UNKNOWN | Q:[]",
            "DENOTES | B:[Q]",
        ]

    def test_all_heads_count_as_resolved(self):
        # A word that heads an entry is resolved — a later terminal
        # resolving to it mints nothing.
        assert _decoded("Mood = Happy\nA = Mood\n") == [
            "DENOTES | Mood:[Happy]",
            "UNKNOWN | Happy:[]",
            "DENOTES | A:[Mood]",
        ]

    def test_connotation_head_is_the_specific_side(self):
        # `A < B` heads B: in `Object < Query` Query is the head, Object
        # is not — so Object still mints its own S4 (via the `L = O`
        # terminal), and Query (a head) never does.
        out = _decoded(self.MHALL)
        assert "UNKNOWN | Object:[]" in out
        assert not any("Query" in l for l in out if l.startswith("UNKNOWN"))


class TestCompoundLabelNotation:
    """A compound of 2+ signatures composes its label through merge —
    pipe-separated component words ('Mary|Subject'); the MTS uppercase
    initials (MS) are presentation, rendered by the harness."""

    MHALL = """(Mary had a little lamb) == SVO =>
   Subject < M
   Verb < H
   Object < Query < ALL =>
     A = Det
     L = Mod
     L = O
"""

    def test_concat_compound_labels_as_initials(self):
        # `Subject < M` (M bound to Mary) composes the compound — the KDbg
        # label is the authored identifier; the VALUE label is the merge
        # composition of the component words (test_signature_of_joins), and
        # the identifier never clobbers the composed value.
        out = _decoded(self.MHALL)
        assert "CONNOTES | MarySubject:[Subject]" in out

    def test_lowercase_component_contributes_its_uppercased_initial(self):
        # had + Verb → hadVerb — every component word carries into the
        # authored identifier, as the sentence form does (had → H).
        out = _decoded(self.MHALL)
        assert "CONNOTES | hadVerb:[Verb]" in out

    def test_signature_of_joins_initials(self):
        from kalvin.kline import KNode
        from kalvin.signifier import NLPSignifier

        sig = NLPSignifier().signature_of(
            [KNode(1 << 32, "Mary"), KNode(2 << 32, "Subject")]
        )
        assert sig.label == "Mary|Subject"

    def test_single_signature_keeps_its_word_label(self):
        # One node is not a compound — the full label stands.
        from kalvin.kline import KNode
        from kalvin.signifier import NLPSignifier

        sig = NLPSignifier().signature_of([KNode(1 << 32, "Mary")])
        assert sig.label == "Mary"

    def test_authored_compound_identifier_unchanged(self):
        # An authored ALL-UPPER identifier (MHALL, SVO) is already the
        # notation — it stands as authored.
        out = _decoded("MHALL = X\n")
        assert "CANONICALISES | MHALL:[M, H, A, L, L]" in out
