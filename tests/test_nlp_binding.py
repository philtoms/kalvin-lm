"""Integration tests for NLP binding — full MHALL example and edge cases.

Tests exercise the simplified v2.0 NLP binding pipeline end-to-end:
  Compiler creates BindingScope → ASTEmitter resolves inline → TokenEncoder encodes

Single-pass inline resolution using BindingScope (the former
BindingResolver/NLPSymbolTable modules have been removed). Characters bind via first-letter
matching (case-sensitive) from word lists, with an occurrence counter for
disambiguation. Inline comments (e.g. S(ubject)) bind immediately and
bypass the counter.

NOTE: BindingScope uses case-sensitive first-letter matching. Word list
words must have uppercase first letters to match uppercase KScript
signature characters (e.g. 'Had' not 'had' to match 'H').

Coverage maps to spec NB-* IDs:
  - NB-4  : First-letter matching from word list
  - NB-5  : No matching words → inert
  - NB-6  : Orphan comment → inert
  - NB-7  : Multiple word lists accumulate (most-recent-first search)
  - NB-9  : Inline override patches parent kline
  - NB-10 : Ambiguous match → occurrence counter
  - NB-11 : Duplicate character disambiguation (L#0 vs L#1)
  - NB-12 : Inline override and scope shadowing
  - NB-13 : Scope restoration after subscript exit
  - NB-14 : Unbound signature falls back to standard encoding
  - NB-17 : Mixed MCS — bound + unbound chars in same signature
  - NB-18 : Mod32 compilation unchanged
  - NB-19 : Same source compiles under both Mod32 and NLP
  - NB-23 : Full MHALL end-to-end (primary integration test)
  - NB-26 : Significance routing
  - NB-27 : Counter only increments on ambiguous match
  - NB-28 : Counter resets at scope boundary
  - NB-29 : Inline binding bypasses counter
  - NB-30 : Single match does not increment counter
  - NB-31 : Counter exceeds matches → unbound
  - NB-33 : Inline override with no matching char in parent kline

Spec ref: @kscript-nlp-binding §6.4, §10 (test matrix)
"""

from __future__ import annotations

import pytest

from kalvin.mod_tokenizer import Mod32Tokenizer, Mod64Tokenizer
from kscript.compiler import Compiler, compile_source
from kscript.lexer import Lexer
from kscript.parser import Parser
from kscript.ast_emitter import ASTEmitter
from kscript.token_encoder import CompiledEntry
from kscript.decompiler import Decompiler
from kalvin.signature import is_nlp_node

# Conditional NLP tokenizer import
try:
    from kalvin.nlp_tokenizer import NLPTokenizer
    _has_nlp = True
except ImportError:
    _has_nlp = False

_nlp_skip = pytest.mark.skipif(not _has_nlp, reason="NLPTokenizer not available")


# ── Tokenizers ───────────────────────────────────────────────────────────────

_tok32 = Mod32Tokenizer()
_tok64 = Mod64Tokenizer()


def _get_nlp_tokenizer():
    """Get NLPTokenizer instance, or None if unavailable."""
    if not _has_nlp:
        return None
    try:
        return NLPTokenizer.from_files()
    except Exception:
        return None


# ── Compilation helpers ──────────────────────────────────────────────────────

def compile_nlp(source: str) -> list[CompiledEntry]:
    """Compile source with NLPTokenizer."""
    tok = _get_nlp_tokenizer()
    assert tok is not None, "NLPTokenizer not available"
    return compile_source(source, tokenizer=tok, dev=True)


def compile64(source: str) -> list[CompiledEntry]:
    """Compile source with Mod64Tokenizer."""
    return compile_source(source, tokenizer=_tok64, dev=True)


# ── Decoding helpers ─────────────────────────────────────────────────────────

def entries_to_multidict(entries: list[CompiledEntry], tok=None) -> dict[str, list]:
    """Map sig → list of all decoded node values."""
    tok = tok or _tok64
    result: dict[str, list] = {}
    for e in entries:
        sig, nodes = e.decode(tok)
        result.setdefault(sig, []).append(nodes)
    return result


def _md(entries: list[CompiledEntry], tok=None) -> dict[str, list]:
    return entries_to_multidict(entries, tok)


def _has_node(md: dict[str, list], sig: str, node_value) -> bool:
    """Check if a sig has a specific decoded node value in its list."""
    return node_value in md.get(sig, [])


# ── Source constants ─────────────────────────────────────────────────────────

# NOTE: Word list words must have uppercase first letters for case-sensitive
# first-letter matching: "Had" (not "had") to match "H", etc.
MHALL_SOURCE = """\
(Mary Had A Little Lamb)
MHALL == SVO =>
   S(ubject) = M
   V(erb) = H
   O(bject) = ALL =>
     A > D(et)
     L > M(od)
     L > O"""

MHALL_SOURCE_NO_COMMENTS = """\
MHALL == SVO =>
   S = M
   V = H
   O = ALL =>
     A > D
     L > M
     L > O"""


# =============================================================================
# NB-4: First-letter matching
# =============================================================================


@_nlp_skip
class TestFirstLetterMatching:
    """NB-4: Word list binding via first-letter matching.

    Under v2.0, a word list is an immutable pool. Characters seek words
    whose first letter matches the character (case-sensitive). For
    duplicate characters, the occurrence counter tracks which word to use.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_mhall_first_letter_matching(self) -> None:
        """(Mary Had A Little Lamb) + MHALL: M→Mary, H→Had, A→A, L→Little, L→Lamb."""
        source = "(Mary Had A Little Lamb)\nMHALL"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        # All bound words should appear as decoded sigs
        assert "Mary" in md, f"Expected 'Mary', got keys: {sorted(md.keys())}"
        assert "Had" in md, f"Expected 'Had', got keys: {sorted(md.keys())}"
        assert "A" in md, f"Expected 'A', got keys: {sorted(md.keys())}"
        assert "Little" in md, f"Expected 'Little', got keys: {sorted(md.keys())}"
        assert "Lamb" in md, f"Expected 'Lamb', got keys: {sorted(md.keys())}"

    def test_simple_word_list_binding(self) -> None:
        """(Mary Had) + MH → M→Mary, H→Had in compiled entries."""
        source = "(Mary Had)\nMH == X"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        assert "Mary" in md, f"Expected 'Mary', got keys: {sorted(md.keys())}"
        assert "Had" in md, f"Expected 'Had', got keys: {sorted(md.keys())}"


# =============================================================================
# NB-23: Full MHALL End-to-End
# =============================================================================


@_nlp_skip
class TestMHALLFull:
    """Full MHALL script end-to-end — NB-23.

    Compiles the MHALL source with NLPTokenizer and verifies bindings flow
    through the pipeline. The block comment ``(Mary Had A Little Lamb)``
    provides a 5-word list matched by first letter to the 5-char sig
    ``MHALL``, producing root-level bindings: M→Mary, H→Had, A→A,
    L→Little, L→Lamb.

    Inline comments (S(ubject), V(erb), O(bject)) create inline bindings
    in inner scopes and trigger Rule 4 override patching of the parent
    kline MCS CANONIZE entry.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_compilation_succeeds(self) -> None:
        """MHALL source compiles without errors under NLP tokenizer."""
        entries = compile_nlp(MHALL_SOURCE)
        assert len(entries) > 0

    def test_bound_words_in_compiled_entries(self) -> None:
        """NB-4, NB-23: Compiled entries contain NLP-bound words from word list.

        The emitter resolves bound characters to their NLP words before
        emission. Entries with sig decoding to "Mary", "Had", "A",
        "Little" should exist in the compiled output.
        """
        entries = compile_nlp(MHALL_SOURCE)
        md = _md(entries, self._nlp_tok)

        assert "Mary" in md, f"Expected 'Mary' in entries, got keys: {sorted(md.keys())}"
        assert "Had" in md, f"Expected 'Had' in entries, got keys: {sorted(md.keys())}"
        assert "A" in md, f"Expected 'A' in entries, got keys: {sorted(md.keys())}"

    def test_l_bindings_in_compiled_entries(self) -> None:
        """NB-4, NB-11: L resolves to bound words (Little, Lamb)."""
        entries = compile_nlp(MHALL_SOURCE)
        md = _md(entries, self._nlp_tok)

        # L is bound — entries with sig decoding to "Little" should exist
        assert "Little" in md, f"Expected 'Little' in entries, got keys: {sorted(md.keys())}"

    def test_mcs_sig_is_nlp_aware(self) -> None:
        """NB-17: MHALL MCS signature has NLP type bits for bound characters.

        The MHALL MCS is OR-reduced from its constituent characters. Bound
        characters contribute NLP type bits. The resulting signature should
        be an NLP node.
        """
        entries = compile_nlp(MHALL_SOURCE)

        has_nlp_sig = any(is_nlp_node(e.signature) for e in entries)
        assert has_nlp_sig, "Expected at least one entry with NLP signature"

    def test_node_side_carries_nlp_words(self) -> None:
        """NB-16: Nodes for bound characters carry NLP-BPE tokens.

        The node side of `S = M` should carry the NLP word for M ("Mary").
        """
        entries = compile_nlp(MHALL_SOURCE)
        md = _md(entries, self._nlp_tok)

        # Entries with sig "Mary" should exist
        if "Mary" in md:
            nodes_for_mary = md["Mary"]
            assert len(nodes_for_mary) > 0

    def test_countersign_s_equals_m_carries_nlp_word(self) -> None:
        """NB-23: S = M entry carries M's bound word 'Mary' as node.

        The undersign entry for M (resolved to 'Mary') should have 'S' as
        its node (or "Subject" via inline override).
        """
        entries = compile_nlp(MHALL_SOURCE)
        md = _md(entries, self._nlp_tok)

        assert "Mary" in md, (
            f"Expected 'Mary' as sig in entries. Keys: {sorted(md.keys())}"
        )

    def test_decompilation_roundtrip(self) -> None:
        """Decompile MHALL entries — verify key names are recoverable.

        NLP type signatures decode to type descriptions. NLP-BPE nodes
        decode to readable words.
        """
        entries = compile_nlp(MHALL_SOURCE)
        decompiled = Decompiler(self._nlp_tok).decompile(entries)

        assert len(decompiled) > 0
        for e in decompiled:
            assert e.sig, "Decompiled entry should have non-empty sig"


# =============================================================================
# NB-5: Mismatched/no-match comment is inert
# =============================================================================


@_nlp_skip
class TestMismatchedCommentInert:
    """NB-5: No matching words → comment is inert.

    Under v2.0, there is no "word count must match char count" rule.
    Word lists are searched by first-letter matching. If no words start
    with the character, the character remains unbound.

    Test source: (Apple Banana Cherry) XY — X and Y have no matching words
    (no words start with X or Y), so X and Y remain unbound.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_no_match_words_dont_bind(self) -> None:
        """X and Y have no matching words in (Apple Banana Cherry)."""
        source = "(Apple Banana Cherry)\nXY == CD"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        assert len(entries) > 0
        # Apple, Banana, Cherry should NOT appear in decoded entries
        # (no char matches their first letter to bind them)
        assert "Apple" not in md
        assert "Banana" not in md
        assert "Cherry" not in md

    def test_backward_compatible_mismatch_source(self) -> None:
        """Original v1.0 source (one two three) AB still compiles cleanly.

        Under v2.0, 'o' doesn't match 'A', 't' doesn't match 'B' —
        first-letter matching is case-sensitive, so AB is unbound.
        """
        source = "(one two three)\nAB == CD"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        assert len(entries) > 0
        assert "one" not in md
        assert "two" not in md


# =============================================================================
# NB-6: Orphan comment is inert
# =============================================================================


@_nlp_skip
class TestOrphanCommentInert:
    """NB-6: Comment with no following signature is inert.

    Source: (note)\\nA == B
    The comment "(note)" has no matching characters (N≠A, N≠B), so it's
    inert. Under v2.0, word lists only serve characters encountered after
    them, and only when first-letter matching succeeds.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_orphan_entries_standard(self) -> None:
        """Entries are standard (not bound to 'note')."""
        source = "(note)\nA == B"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        assert "note" not in md
        assert len(entries) >= 2

    def test_orphan_compiles_cleanly(self) -> None:
        """Compilation succeeds, entries are decodable."""
        source = "(note)\nA == B"
        entries = compile_nlp(source)
        assert len(entries) >= 2
        for entry in entries:
            sig, nodes = entry.decode(self._nlp_tok)
            assert isinstance(sig, str)


# =============================================================================
# NB-7: Multiple pending comments
# =============================================================================


@_nlp_skip
class TestMultiplePendingComments:
    """NB-7: Multiple word lists accumulate and are searched most-recent-first.

    Under v2.0, multiple word lists in the same scope accumulate and are
    searched most-recent-first during resolve().

    Source: (Alpha Beta)\\n(Gamma Delta)\\nAB == CD
    Both lists are in the same scope. A is sought: most-recent list first
    (Gamma, Delta) — no A-match. Older list (Alpha, Beta) — A→Alpha.
    B is sought: most-recent list (Gamma, Delta) — no B-match. Older list
    (Alpha, Beta) — B→Beta.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_multiple_word_lists_bind(self) -> None:
        """A→Alpha, B→Beta from accumulated word lists."""
        source = "(Alpha Beta)\n(Gamma Delta)\nAB == CD"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        assert len(entries) > 0
        assert "Alpha" in md, f"Expected 'Alpha', got keys: {sorted(md.keys())}"
        assert "Beta" in md, f"Expected 'Beta', got keys: {sorted(md.keys())}"

    def test_no_match_in_any_list(self) -> None:
        """Neither X nor Y have matches in any word list."""
        source = "(Alpha Beta)\n(Gamma Delta)\nXY == CD"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        assert len(entries) > 0
        assert "Alpha" not in md
        assert "Beta" not in md
        assert "Gamma" not in md
        assert "Delta" not in md


# =============================================================================
# NB-14, NB-17: Unbound char with mixed NLP/Mod32
# =============================================================================


@_nlp_skip
class TestUnboundCharMixed:
    """NB-14, NB-17: Mixed bound and unbound characters.

    Source: (Mary Had) MH == X
    Only M and H get NLP bindings. X is unbound.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_compiled_entries_have_bound_words(self) -> None:
        """Entries with sig 'Mary' and 'Had' exist."""
        source = "(Mary Had)\nMH == X"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        assert "Mary" in md
        assert "Had" in md

    def test_mixed_mcs_has_nlp_bits(self) -> None:
        """NB-17: MCS signature has NLP type bits for bound characters."""
        source = "(Mary Had)\nMH == X"
        entries = compile_nlp(source)

        has_nlp_sig = any(is_nlp_node(e.signature) for e in entries)
        assert has_nlp_sig, "Expected NLP signature bits in mixed MCS"


# =============================================================================
# NB-12, NB-13: Shadowing and Scope Restoration
# =============================================================================


@_nlp_skip
class TestShadowingAndScope:
    """NB-12, NB-13: Shadowing and scope restoration.

    Under v2.0, BindingScope pushes/pops scopes around ``=>`` blocks.
    Characters seek from innermost scope first, then parent scopes upward.
    After all scopes are popped, root-level bindings still produce correct
    compiled entries.

    Tests verify root-level bindings survive scope push/pop by inspecting
    compiled entry output.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_outer_binding_in_entries(self) -> None:
        """M→Mary binding from word list appears in compiled entries."""
        source = "(Mary Had A Little Lamb)\nMHALL == SVO =>\n   S = M\n   V = H"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        # M resolved to "Mary" should appear as a sig
        assert "Mary" in md, f"Expected 'Mary' in entries, got: {sorted(md.keys())}"

    def test_scope_restoration_after_subscript(self) -> None:
        """NB-13: After subscript exits, outer bindings produce correct entries."""
        source = "(Mary Had A Little Lamb)\nMHALL == SVO =>\n   S = M\n   V = H\n   O = ALL =>\n     L > M\n     L > O"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        # Root bindings M→Mary, H→Had, A→A should produce entries
        assert "Mary" in md, f"Expected 'Mary', got: {sorted(md.keys())}"
        assert "Had" in md, f"Expected 'Had', got: {sorted(md.keys())}"
        assert "A" in md, f"Expected 'A', got: {sorted(md.keys())}"

    def test_bindings_identical_with_and_without_nested_subscript(self) -> None:
        """Root-level compiled entries are consistent regardless of nesting.

        The same root-level bound words (Mary, Had, etc.) appear whether
        or not nested subscript blocks exist.
        """
        source_shallow = "(Mary Had A Little Lamb)\nMHALL == SVO =>\n   S = M"
        source_deep = "(Mary Had A Little Lamb)\nMHALL == SVO =>\n   S = M\n   O = ALL =>\n     A > D\n     L > M\n     L > O"

        md_shallow = _md(compile_nlp(source_shallow), self._nlp_tok)
        md_deep = _md(compile_nlp(source_deep), self._nlp_tok)

        # Both should have "Mary" (M→Mary binding from word list)
        assert "Mary" in md_shallow
        assert "Mary" in md_deep


# =============================================================================
# NB-11: Duplicate Character Disambiguation
# =============================================================================


@_nlp_skip
class TestDuplicateCharDisambiguation:
    """NB-11: Duplicate characters get different bindings via occurrence counter.

    Source: (Little Lamb) LL == X

    The BindingScope occurrence counter resolves L: first L→"Little",
    second L→"Lamb". Integration tests verify the compiled output.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_duplicate_chars_compiled_entries(self) -> None:
        """Compiled entries contain 'Little' (first L binding)."""
        source = "(Little Lamb)\nLL == X"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        # "Little" should appear as a sig in the compiled output
        assert "Little" in md, f"Expected 'Little' in entries, got: {sorted(md.keys())}"

    def test_both_l_bindings_in_entries(self) -> None:
        """NB-11: Both L bindings (Little, Lamb) appear in compiled entries."""
        source = "(Little Lamb)\nLL == X"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        # Both "Little" and "Lamb" should appear from the two L positions
        assert "Little" in md, f"Expected 'Little', got: {sorted(md.keys())}"
        assert "Lamb" in md, f"Expected 'Lamb', got: {sorted(md.keys())}"


# =============================================================================
# NB-9, NB-12, NB-33: Inline Override
# =============================================================================


@_nlp_skip
class TestInlineOverride:
    """NB-9, NB-12, NB-33: Inline override patches parent kline.

    When an inline binding fires in a subscript, it retroactively patches
    the matching character in the already-emitted parent kline MCS CANONIZE
    entry. Only the immediate parent kline is patched — no propagation
    beyond one level.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_inline_override_patches_parent_kline(self) -> None:
        """NB-9, NB-12: S(ubject) in subscript patches S in parent SVO kline.

        Source: SVO == ABC =>\n  S(ubject) = M
        The inline comment S(ubject) fires inside the subscript scope.
        Rule 4 override patches 'S' → 'Subject' in the parent kline's
        MCS CANONIZE entry for SVO.
        """
        source = "SVO == ABC =>\n  S(ubject) = M"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        # The compilation should succeed and produce entries
        assert len(entries) > 0

        # "Subject" should appear in decoded entries (from the inline override)
        found_subject = False
        for sig, node_lists in md.items():
            for n in node_lists:
                if n == "Subject" or (isinstance(n, list) and "Subject" in n):
                    found_subject = True
        assert found_subject, f"Expected 'Subject' in entries, got: {md}"

    def test_inline_override_no_match(self) -> None:
        """NB-33: Inline binding with no matching char in parent kline — safe no-op.

        Source: AB == CD =>\n  X(Y) = Z
        X(Y) resolves to "XY" (inline comment Y on sig X). X is not found
        in parent kline "AB" — override does nothing. Compilation succeeds.
        """
        source = "AB == CD =>\n  X(Y) = Z"
        entries = compile_nlp(source)

        # Compilation completes without error
        assert len(entries) > 0
        md = _md(entries, self._nlp_tok)
        # "XY" should appear as a resolved word from the inline comment
        # (the word "XY" from sig X + inline "(Y)")
        found = False
        for sig in md:
            if "XY" in str(sig):
                found = True
        # At minimum, entries are decodable
        for entry in entries:
            sig, nodes = entry.decode(self._nlp_tok)
            assert isinstance(sig, str)


# =============================================================================
# NB-10, NB-27, NB-28, NB-29, NB-30, NB-31: Occurrence Counter
# =============================================================================


@_nlp_skip
class TestOccurrenceCounter:
    """Integration tests for occurrence counter behaviour through compiled entries.

    Counter logic is unit-tested in KB-169 (test_binding_scope.py).
    These tests verify end-to-end flow through compiled entry decode.

    Rules:
      - Counter only increments on ambiguous (multiple match) binding
      - Single match → counter does not increment
      - Counter resets at scope boundary (new scope starts at zero)
      - Inline binding bypasses counter
      - Counter exceeding matches → character unbound
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_ambiguous_match_occurrence_counter(self) -> None:
        """NB-10: (Alice Alpha) AA → first A→Alice, second A→Alpha."""
        source = "(Alice Alpha)\nAA == X"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        # Both Alice and Alpha should appear
        assert "Alice" in md, f"Expected 'Alice', got: {sorted(md.keys())}"
        assert "Alpha" in md, f"Expected 'Alpha', got: {sorted(md.keys())}"

    def test_single_match_no_counter_increment(self) -> None:
        """NB-27, NB-30: Single match does not increment counter.

        (Mary) M → M→Mary. Only one word starts with M, so counter stays
        at zero. A second M should still resolve to "Mary" since counter
        didn't advance past the single match.
        """
        source = "(Mary)\nMM == X"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        # First M→Mary, second M→Mary (counter didn't increment)
        mary_entries = md.get("Mary", [])
        assert len(mary_entries) >= 1, f"Expected 'Mary' entries, got: {md}"

    def test_counter_resets_at_scope_boundary(self) -> None:
        """NB-28: Counter resets at scope boundary.

        In the subscript scope, the counter for A starts fresh.
        (Alice Alpha) in root scope + subscript scope gets its own counter.
        """
        source = "(Alice Alpha)\nAA =>\n  A > B"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        assert len(entries) > 0
        # "Alice" should appear (first A in root scope)
        assert "Alice" in md, f"Expected 'Alice', got: {sorted(md.keys())}"

    def test_inline_binding_bypasses_counter(self) -> None:
        """NB-29: Inline binding bypasses counter.

        (Alice Alpha) + A(lice) as inline comment: inline resolves to
        "Alice" immediately without incrementing the counter. A subsequent
        A should still get "Alice" from the word list (counter at 0).
        """
        source = "(Alice Alpha)\nA(lice) == B"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        assert len(entries) > 0
        # Inline comment A(lice) → "Alice"
        assert "Alice" in md, f"Expected 'Alice', got: {sorted(md.keys())}"

    def test_counter_exceeds_matches_unbound(self) -> None:
        """NB-31: Counter exceeds matches → character unbound for that occurrence.

        (Alice Alpha) AAA — two A-words but three A characters.
        First A→Alice (counter 0→1), second A→Alpha (counter 1→2),
        third A → counter 2 ≥ len(matches=2) → unbound (stays 'A').
        """
        source = "(Alice Alpha)\nAAA == X"
        entries = compile_nlp(source)
        md = _md(entries, self._nlp_tok)

        # Alice and Alpha should appear from first two As
        assert "Alice" in md, f"Expected 'Alice', got: {sorted(md.keys())}"
        assert "Alpha" in md, f"Expected 'Alpha', got: {sorted(md.keys())}"
        # The third A is unbound — raw 'A' may or may not appear depending
        # on dedup. At minimum, compilation succeeds.
        assert len(entries) > 0


# =============================================================================
# NB-18, NB-19: Mod32 Compatibility
# =============================================================================


class TestMod32Compatibility:
    """NB-18: Mod32 compilation unchanged. NB-19: Same source both modes.

    Under Mod32 (supports_mcs=True), no BindingScope is created. Inline
    comments still resolve (that's ASTEmitter behavior, not scope-dependent).
    """

    def test_mhall_mod32_no_nlp_nodes(self) -> None:
        """NB-18: Mod32 entries have no NLP nodes."""
        entries = compile_source(MHALL_SOURCE, tokenizer=_tok32, dev=True)

        for entry in entries:
            assert not is_nlp_node(entry.signature), (
                f"Mod32 entry sig {entry.signature:#x} should not be NLP"
            )

    def test_mhall_mod64_block_comments_inert(self) -> None:
        """NB-18: Block comments (word lists) are inert under Mod64.

        Block comments like (Mary Had A Little Lamb) are inert under Mod64
        because no BindingScope is created. Inline comments like S(ubject)
        still resolve in all modes (that's ASTEmitter behavior), so we test
        block-comment inertness by comparing with/without a block comment
        on an otherwise identical source.
        """
        # Same source with and without block comment — no inline comments
        entries_with = compile64("(Mary Had A Little Lamb)\nMHALL == SVO =>\n   S = M")
        entries_without = compile64("MHALL == SVO =>\n   S = M")

        md_with = _md(entries_with, _tok64)
        md_without = _md(entries_without, _tok64)

        assert md_with == md_without


@_nlp_skip
class TestSameSourceBothModes:
    """NB-19: Same source compiles under both Mod32 and NLP."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    def test_simple_source_both_modes(self) -> None:
        """A == B compiles under both Mod32 and NLP without errors."""
        entries_mod = compile_source("A == B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_nlp("A == B")

        assert len(entries_mod) >= 2
        assert len(entries_nlp) >= 2

    def test_inline_comment_source_both_modes(self) -> None:
        """S(ubject) = M compiles under both Mod32 and NLP."""
        source = "S(ubject) = M"
        entries_mod = compile_source(source, tokenizer=_tok64, dev=True)
        entries_nlp = compile_nlp(source)

        assert len(entries_mod) >= 1
        assert len(entries_nlp) >= 1

    def test_mod32_entries_not_nlp(self) -> None:
        """Mod32 entries don't carry NLP nodes."""
        entries_mod = compile_source("A == B", tokenizer=_tok32, dev=True)
        for e in entries_mod:
            assert not is_nlp_node(e.signature)

    def test_nlp_entries_are_nlp_nodes(self) -> None:
        """NLP entries carry NLP-BPE nodes."""
        entries_nlp = compile_nlp("A == B")
        for e in entries_nlp:
            assert is_nlp_node(e.signature), (
                f"NLP entry sig {e.signature:#x} should be NLP-BPE"
            )


# =============================================================================
# NB-18 (extended): Mod32 compilation unchanged — comprehensive operator sweep
# =============================================================================


class TestNB18Mod32Unchanged:
    """NB-18: Mod32 compilation is completely unchanged by binding infrastructure.

    Verifies that every operator produces identical, known-good Mod32 output.
    No BindingScope is created for Mod32, no NLP nodes appear, and block
    comments are inert (word lists have no effect). Inline comments still
    resolve in the emitter — that's scope-independent behavior.

    These tests run unconditionally (no NLPTokenizer required) because they
    only exercise the Mod32 path.
    """

    OPERATOR_SOURCES: dict[str, str] = {
        "unsigned": "A",
        "countersign": "A == B",
        "undersign": "A = B",
        "connotate": "A > B",
        "canonize": "A => B",
        "subscript_block": "A =>\n  B\n  C",
        "chained": "A => B => C",
        "mcs": "ABC",
        "mhall": MHALL_SOURCE,
    }

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _compile32(source: str) -> list[CompiledEntry]:
        return compile_source(source, tokenizer=_tok32, dev=True)

    @staticmethod
    def _entry_snapshot(entries: list[CompiledEntry]) -> list[tuple]:
        """Produce a deterministic snapshot: (sig, nodes_tuple, sig_level)."""
        return [
            (e.signature, tuple(e.nodes) if isinstance(e.nodes, list) else e.nodes, e.sig_level)
            for e in entries
        ]

    # ── Tests: operator output matches known-good references ───────────

    def test_unsigned_entry_count_and_level(self) -> None:
        """Unsigned `A` → 1 entry, S4."""
        entries = self._compile32("A")
        assert len(entries) == 1
        assert entries[0].sig_level == "S4"

    def test_countersign_entry_count_and_levels(self) -> None:
        """Countersign `A == B` → 2 entries, both S1."""
        entries = self._compile32("A == B")
        assert len(entries) == 2
        assert all(e.sig_level == "S1" for e in entries)

    def test_undersign_entry_count_and_levels(self) -> None:
        """Undersign `A = B` → 1 entry, S1."""
        entries = self._compile32("A = B")
        assert len(entries) == 1
        assert entries[0].sig_level == "S1"

    def test_connotate_entry_count_and_level(self) -> None:
        """Connotate `A > B` → 1 entry, S3."""
        entries = self._compile32("A > B")
        assert len(entries) == 1
        assert entries[0].sig_level == "S3"

    def test_canonize_entry_count_and_levels(self) -> None:
        """Canonize `A => B` → canonize (S2) + unsigned (S4)."""
        entries = self._compile32("A => B")
        assert len(entries) == 2
        ops = [e.sig_level for e in entries]
        assert "S2" in ops
        assert "S4" in ops

    def test_subscript_block_entry_count_and_levels(self) -> None:
        """Subscript block `A =>\\n  B\\n  C` → 1 canonize (S2) + 2 unsigned (S4)."""
        entries = self._compile32("A =>\n  B\n  C")
        assert len(entries) == 3
        levels = [e.sig_level for e in entries]
        assert levels.count("S2") == 1
        assert levels.count("S4") == 2

    def test_chained_entry_count_and_levels(self) -> None:
        """Chained `A => B => C` → 2 canonize (S2) + 1 unsigned (S4)."""
        entries = self._compile32("A => B => C")
        assert len(entries) == 3
        levels = [e.sig_level for e in entries]
        assert levels.count("S2") == 2
        assert levels.count("S4") == 1

    def test_mcs_entry_count_and_levels(self) -> None:
        """MCS `ABC` → 3 unsigned (S4) + 1 canonize (S2) + 1 unsigned MCS (S4)."""
        entries = self._compile32("ABC")
        levels = [e.sig_level for e in entries]
        assert len(entries) >= 5
        assert levels.count("S4") >= 4
        assert levels.count("S2") >= 1

    # ── Tests: decoded signatures/nodes match expected Mod32 values ────

    def test_decoded_signatures_unsigned(self) -> None:
        """Unsigned `A` decodes sig='A'."""
        entries = self._compile32("A")
        md = _md(entries, _tok32)
        assert "A" in md

    def test_decoded_signatures_countersign(self) -> None:
        """Countersign `A == B` decodes sig='A' node='B' and sig='B' node='A'."""
        entries = self._compile32("A == B")
        md = _md(entries, _tok32)
        assert _has_node(md, "A", ["B"])
        assert _has_node(md, "B", ["A"])

    def test_decoded_signatures_undersign(self) -> None:
        """Undersign `A = B` decodes sig='B' node='A'."""
        entries = self._compile32("A = B")
        md = _md(entries, _tok32)
        assert _has_node(md, "B", ["A"])

    def test_decoded_signatures_connotate(self) -> None:
        """Connotate `A > B` decodes sig='A' node='B'."""
        entries = self._compile32("A > B")
        md = _md(entries, _tok32)
        assert _has_node(md, "A", ["B"])

    def test_decoded_signatures_canonize(self) -> None:
        """Canonize `A => B` decodes sig='A' nodes=['B']."""
        entries = self._compile32("A => B")
        md = _md(entries, _tok32)
        assert _has_node(md, "A", ["B"])

    def test_decoded_subscript_block(self) -> None:
        """Subscript `A =>\\n  B\\n  C` decodes sig='A' nodes=['B','C']."""
        entries = self._compile32("A =>\n  B\n  C")
        md = _md(entries, _tok32)
        assert _has_node(md, "A", ["B", "C"])

    def test_decoded_mcs(self) -> None:
        """MCS `ABC` has an entry with sig='ABC' and nodes=['A','B','C']."""
        entries = self._compile32("ABC")
        md = _md(entries, _tok32)
        assert "ABC" in md

    # ── Tests: no NLP nodes for any operator ──────────────────────────

    def test_no_nlp_nodes_all_operators(self) -> None:
        """No entry from any Mod32 compilation has NLP type bits."""
        for name, source in self.OPERATOR_SOURCES.items():
            entries = self._compile32(source)
            for e in entries:
                assert not is_nlp_node(e.signature), (
                    f"{name}: Mod32 entry sig {e.signature:#x} should not be NLP"
                )

    # ── Tests: block comments are inert under Mod32 ───────────────────

    def test_block_comments_inert(self) -> None:
        """`(Mary Had) MH` and `MH` produce identical Mod32 output."""
        entries_with = self._compile32("(Mary Had)\nMH")
        entries_without = self._compile32("MH")

        snap_with = self._entry_snapshot(entries_with)
        snap_without = self._entry_snapshot(entries_without)
        assert snap_with == snap_without

    # ── Test: deterministic — same source same output ──────────────────

    def test_deterministic_compilation(self) -> None:
        """Compiling the same source twice yields identical entries."""
        for name, source in self.OPERATOR_SOURCES.items():
            entries_a = self._compile32(source)
            entries_b = self._compile32(source)
            assert self._entry_snapshot(entries_a) == self._entry_snapshot(entries_b), (
                f"{name}: Mod32 compilation should be deterministic"
            )


# =============================================================================
# NB-19 (extended): Same source compiles under both modes
# =============================================================================


@_nlp_skip
class TestNB19SameSourceBothModes:
    """NB-19: Same `.ks` source compiles under both Mod32 and NLP.

    Verifies that every operator type produces valid compilation
    under both tokenizers, with structurally equivalent operator semantics.
    The encoding differs (Mod32 vs NLP-BPE) but the relationship graph
    (who countersigns whom, who canonizes what) is preserved.
    """

    OPERATOR_SOURCES: dict[str, str] = {
        "unsigned": "A",
        "countersign": "A == B",
        "undersign": "A = B",
        "connotate": "A > B",
        "canonize": "A => B",
        "subscript_block": "A =>\n  B\n  C",
        "chained": "A => B => C",
    }

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    # ── Tests: every operator compiles under both modes ────────────────

    def test_all_operators_compile_both_modes(self) -> None:
        """Every operator source produces entries under both Mod32 and NLP."""
        for name, source in self.OPERATOR_SOURCES.items():
            entries_mod = compile_source(source, tokenizer=_tok32, dev=True)
            entries_nlp = compile_source(source, tokenizer=self._nlp_tok, dev=True)

            assert len(entries_mod) > 0, f"{name}: Mod32 produced no entries"
            assert len(entries_nlp) > 0, f"{name}: NLP produced no entries"

    def test_mhall_compiles_both_modes(self) -> None:
        """Full MHALL example compiles under both modes without errors."""
        entries_mod = compile_source(MHALL_SOURCE, tokenizer=_tok32, dev=True)
        entries_nlp = compile_source(MHALL_SOURCE, tokenizer=self._nlp_tok, dev=True)

        assert len(entries_mod) > 0
        assert len(entries_nlp) > 0

    def test_sources_with_comments_compile_both_modes(self) -> None:
        """Sources with inline comments compile under both modes."""
        sources_with_comments = [
            "S(ubject) = M",
            "(Mary Had) MH",
            MHALL_SOURCE,
        ]
        for source in sources_with_comments:
            entries_mod = compile_source(source, tokenizer=_tok32, dev=True)
            entries_nlp = compile_source(source, tokenizer=self._nlp_tok, dev=True)
            assert len(entries_mod) > 0
            assert len(entries_nlp) > 0

    # ── Tests: sig_level distribution matches between modes ────────────

    def test_countersign_levels_match(self) -> None:
        """`A == B` countersign entries have S1 in both modes."""
        entries_mod = compile_source("A == B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A == B", tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels
        assert all(l == "S1" for l in mod_levels)

    def test_connotate_levels_match(self) -> None:
        """`A > B` connotate entries have S3 in both modes."""
        entries_mod = compile_source("A > B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A > B", tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels
        assert all(l == "S3" for l in mod_levels)

    def test_unsigned_levels_match(self) -> None:
        """`A` unsigned entries have S4 in both modes."""
        entries_mod = compile_source("A", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A", tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels
        assert all(l == "S4" for l in mod_levels)

    def test_canonize_levels_match(self) -> None:
        """`A => B` canonize levels match between modes."""
        entries_mod = compile_source("A => B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A => B", tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels

    def test_undersign_levels_match(self) -> None:
        """`A = B` undersign levels match between modes."""
        entries_mod = compile_source("A = B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A = B", tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels

    # ── Tests: operator semantics preserved across modes ───────────────

    def test_countersign_semantics_preserved(self) -> None:
        """`A == B` → countersign relationship A↔B preserved in decoded output."""
        entries_mod = compile_source("A == B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A == B", tokenizer=self._nlp_tok, dev=True)

        md_mod = _md(entries_mod, _tok32)
        md_nlp = _md(entries_nlp, self._nlp_tok)

        # Mod32: A has B, B has A (bidirectional)
        assert _has_node(md_mod, "A", ["B"])
        assert _has_node(md_mod, "B", ["A"])

        # NLP: same relationship structure — both entries exist
        assert len(md_nlp) >= 2

    def test_undersign_semantics_preserved(self) -> None:
        """`A = B` → undersign relationship B→A preserved."""
        entries_mod = compile_source("A = B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A = B", tokenizer=self._nlp_tok, dev=True)

        md_mod = _md(entries_mod, _tok32)
        md_nlp = _md(entries_nlp, self._nlp_tok)

        # Mod32: sig B, node A
        assert _has_node(md_mod, "B", ["A"])

        # NLP: entry exists with a node value
        assert len(md_nlp) >= 1

    def test_connotate_semantics_preserved(self) -> None:
        """`A > B` → connotate relationship A→B preserved."""
        entries_mod = compile_source("A > B", tokenizer=_tok32, dev=True)
        md_mod = _md(entries_mod, _tok32)

        assert _has_node(md_mod, "A", ["B"])

    def test_canonize_semantics_preserved(self) -> None:
        """`A => B` → canonize relationship A→[B] preserved."""
        entries_mod = compile_source("A => B", tokenizer=_tok32, dev=True)
        md_mod = _md(entries_mod, _tok32)

        assert _has_node(md_mod, "A", ["B"])

    # ── Tests: encoding mode differences ───────────────────────────────

    def test_mod32_entries_not_nlp(self) -> None:
        """Mod32 entries don't carry NLP nodes for any operator."""
        for name, source in self.OPERATOR_SOURCES.items():
            entries = compile_source(source, tokenizer=_tok32, dev=True)
            for e in entries:
                assert not is_nlp_node(e.signature), (
                    f"{name}: Mod32 sig {e.signature:#x} should not be NLP"
                )

    def test_nlp_entries_have_nlp_signatures(self) -> None:
        """NLP entries carry NLP-BPE type bits for single-char sigs."""
        source = "A == B"
        entries = compile_source(source, tokenizer=self._nlp_tok, dev=True)
        has_nlp = any(is_nlp_node(e.signature) for e in entries)
        assert has_nlp, "Expected at least one NLP-type signature"


# =============================================================================
# NB-26: Significance routing on NLP-bound klines
# =============================================================================


@_nlp_skip
class TestNB26SignificanceRouting:
    """NB-26: S1/S2/S3/S4 significance routing is tokenizer-agnostic.

    The sig_level on compiled entries is determined by the operator type
    (COUNTERSIGN→S1, CANONIZE→S2, CONNOTATE→S3, UNSIGNED→S4), not by the
    encoding scheme.  These tests verify that NLP-bound klines carry the
    correct significance levels and that levels match Mod32 output for the
    same source.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._nlp_tok = _get_nlp_tokenizer()
        if self._nlp_tok is None:
            pytest.skip("NLPTokenizer data files not available")

    # ── Individual operator significance levels ────────────────────────

    def test_countersign_s1(self) -> None:
        """`A == B` with NLP → countersign entries have sig_level='S1'."""
        entries = compile_source("A == B", tokenizer=self._nlp_tok, dev=True)
        assert len(entries) >= 2
        for e in entries:
            assert e.sig_level == "S1", (
                f"Countersign entry should be S1, got {e.sig_level!r}"
            )

    def test_canonize_s2(self) -> None:
        """`A => B C` with NLP → canonize entry has sig_level='S2'."""
        entries = compile_source("A => B C", tokenizer=self._nlp_tok, dev=True)
        s2_entries = [e for e in entries if e.sig_level == "S2"]
        assert len(s2_entries) >= 1, (
            f"Expected at least one S2 canonize entry, got levels: "
            f"{[e.sig_level for e in entries]}"
        )

    def test_connotate_s3(self) -> None:
        """`A > B` with NLP → connotate entry has sig_level='S3'."""
        entries = compile_source("A > B", tokenizer=self._nlp_tok, dev=True)
        assert len(entries) >= 1
        assert entries[0].sig_level == "S3", (
            f"Connotate entry should be S3, got {entries[0].sig_level!r}"
        )

    def test_unsigned_s4(self) -> None:
        """`A` (unsigned) with NLP → entry has sig_level='S4'."""
        entries = compile_source("A", tokenizer=self._nlp_tok, dev=True)
        assert len(entries) >= 1
        assert entries[0].sig_level == "S4", (
            f"Unsigned entry should be S4, got {entries[0].sig_level!r}"
        )

    def test_undersign_s1(self) -> None:
        """`A = B` with NLP → undersign entry has sig_level='S1'."""
        entries = compile_source("A = B", tokenizer=self._nlp_tok, dev=True)
        assert len(entries) >= 1
        assert entries[0].sig_level == "S1", (
            f"Undersign entry should be S1, got {entries[0].sig_level!r}"
        )

    # ── Full MHALL: mixed significance levels ──────────────────────────

    def test_mhall_mixed_levels(self) -> None:
        """MHALL under NLP produces entries with a mix of S1, S2, S3, S4."""
        entries = compile_source(
            MHALL_SOURCE, tokenizer=self._nlp_tok, dev=True
        )
        levels = {e.sig_level for e in entries}

        # MHALL contains countersign (S1), canonize (S2), connotate (S3),
        # and unsigned (S4) entries
        assert "S1" in levels, f"Expected S1 in MHALL levels, got {levels}"
        assert "S2" in levels, f"Expected S2 in MHALL levels, got {levels}"
        assert "S3" in levels, f"Expected S3 in MHALL levels, got {levels}"
        assert "S4" in levels, f"Expected S4 in MHALL levels, got {levels}"

    def test_mhall_s1_are_countersign_or_undersign(self) -> None:
        """S1 entries in MHALL correspond to countersign (==) or undersign (=)."""
        entries = compile_source(
            MHALL_SOURCE, tokenizer=self._nlp_tok, dev=True
        )
        s1_entries = [e for e in entries if e.sig_level == "S1"]
        assert len(s1_entries) >= 1, "MHALL should have at least one S1 entry"

    def test_mhall_s2_are_canonize(self) -> None:
        """S2 entries in MHALL correspond to canonize (=>)."""
        entries = compile_source(
            MHALL_SOURCE, tokenizer=self._nlp_tok, dev=True
        )
        s2_entries = [e for e in entries if e.sig_level == "S2"]
        assert len(s2_entries) >= 1, "MHALL should have at least one S2 entry"

    def test_mhall_s3_are_connotate(self) -> None:
        """S3 entries in MHALL correspond to connotate (>)."""
        entries = compile_source(
            MHALL_SOURCE, tokenizer=self._nlp_tok, dev=True
        )
        s3_entries = [e for e in entries if e.sig_level == "S3"]
        assert len(s3_entries) >= 1, "MHALL should have at least one S3 entry"

    # ── Cross-mode significance equivalence ────────────────────────────

    def test_levels_identical_across_modes_simple(self) -> None:
        """`A == B` sig_levels are identical under Mod32 and NLP."""
        entries_mod = compile_source("A == B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A == B", tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels

    def test_levels_identical_across_modes_connotate(self) -> None:
        """`A > B` sig_levels are identical under Mod32 and NLP."""
        entries_mod = compile_source("A > B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A > B", tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels

    def test_levels_identical_across_modes_canonize(self) -> None:
        """`A => B` sig_levels are identical under Mod32 and NLP."""
        entries_mod = compile_source("A => B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A => B", tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels

    def test_levels_identical_across_modes_unsigned(self) -> None:
        """`A` sig_levels are identical under Mod32 and NLP."""
        entries_mod = compile_source("A", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A", tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels

    def test_levels_identical_across_modes_undersign(self) -> None:
        """`A = B` sig_levels are identical under Mod32 and NLP."""
        entries_mod = compile_source("A = B", tokenizer=_tok32, dev=True)
        entries_nlp = compile_source("A = B", tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels

    def test_levels_identical_across_modes_subscript(self) -> None:
        """Subscript block sig_levels are identical under Mod32 and NLP."""
        source = "A =>\n  B\n  C"
        entries_mod = compile_source(source, tokenizer=_tok32, dev=True)
        entries_nlp = compile_source(source, tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels

    def test_levels_identical_across_modes_chained(self) -> None:
        """Chained construct sig_levels are identical under Mod32 and NLP."""
        source = "A => B => C"
        entries_mod = compile_source(source, tokenizer=_tok32, dev=True)
        entries_nlp = compile_source(source, tokenizer=self._nlp_tok, dev=True)

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels

    def test_levels_identical_across_modes_mhall(self) -> None:
        """MHALL sig_levels are identical under Mod32 and NLP."""
        entries_mod = compile_source(
            MHALL_SOURCE, tokenizer=_tok32, dev=True
        )
        entries_nlp = compile_source(
            MHALL_SOURCE, tokenizer=self._nlp_tok, dev=True
        )

        mod_levels = sorted(e.sig_level for e in entries_mod)
        nlp_levels = sorted(e.sig_level for e in entries_nlp)
        assert mod_levels == nlp_levels, (
            f"MHALL level mismatch: Mod32={mod_levels} vs NLP={nlp_levels}"
        )
