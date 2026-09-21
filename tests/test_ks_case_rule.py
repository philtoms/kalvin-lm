"""Case-rule regressions: word expansion and uppercase-only attraction.

An identifier's case frames its reading (CONTEXT.md, Word Binding):
ALL-UPPER is a compound (MHALL), Capitalized carries its own expansion
(``Mood`` ≡ ``M(ood)``; an explicit annotation suppresses it), and
lowercase-first is a literal word (``had``). Ambient attraction applies
only to bare single chars in sig case — a lowercase single char is the
literal word (the article ``a``) and never attracts.
"""

from __future__ import annotations

from ks.compiler import compile_source


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
        # After Mood expands, a bare M resolves to the word.
        assert _decoded("Mood = Happy\nM = X\n") == [
            "DENOTES | Mood:[Happy]",
            "DENOTES | Mood:[X]",
        ]

    def test_explicit_annotation_suppresses_expansion(self):
        # Mood(x) stays the literal word Mood; M is never bound.
        assert _decoded("Mood(x) = Y\nM = Z\n") == [
            "DENOTES | Mood:[Y]",
            "DENOTES | M:[Z]",
        ]

    def test_all_upper_stays_compound(self):
        assert "CANONICALISES | MHALL:[M, H, A, L, L]" in _decoded("MHALL = X\n")

    def test_lowercase_stays_literal_word(self):
        # A lowercase node is the word itself, never an expansion.
        assert _decoded("A = had\n") == ["DENOTES | A:[had]"]

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
            "DENOTES | mary:[X]"
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
        assert _decoded("(axe)\nL = a\n") == ["DENOTES | L:[a]"]

    def test_occurrence_counter_still_disambiguates(self):
        # L attracts little then lamb; A attracts the article a; B stays raw.
        assert _decoded("(Mary had a little lamb)\nL = A\nL = B\n") == [
            "DENOTES | little:[a]",
            "DENOTES | lamb:[B]",
        ]

    def test_authored_witness_fires_regardless_of_position(self):
        # h(ad) means 'had' whether top-level or inside a subscript —
        # the lowercase gate removed the ambient 'have' that used to
        # block the top-level form.
        assert _decoded("(what did Mary have)\nh(ad) => did have\n") == [
            "CANONICALISES | had:[did, have]"
        ]

    def test_authored_witness_binds_compound_char(self):
        # The witness doctrine: h(ad) binds the compound char H.
        out = _decoded("(what did Mary have)\nWDMH == MHALL =>\n  h(ad) => did have\n  ALL = O(bject)\n")
        assert "CANONICALISES | MHALL:[Mary, had, A, L, L]" in out
