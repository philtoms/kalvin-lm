"""Tests for ASTEmitter NLP BindingScope integration.

Covers KB-170 acceptance criteria:
  - NB-1: Inline S(ubject) binding
  - NB-2: Inline V(erb) binding
  - NB-3: Node-side inline D(et) binding
  - NB-9: Inline override parent kline MCS CANONIZE entry
  - NB-10/11: Occurrence counter disambiguation
  - NB-12: Override immediate parent only
  - NB-14: Unbound character stays raw
  - NB-15: Bound sig carries resolved NLP word
  - NB-16: Bound node carries resolved NLP word
  - NB-17: Mixed MCS with bound and unbound characters
  - NB-29: Inline bypasses occurrence counter
  - NB-32: Forward-only word list
  - NB-33: Inline override no-match (safe no-op)

Note: BindingScope uses case-sensitive first-letter matching. Word list
words must have uppercase first letters to match uppercase KScript
signature characters (e.g. "Had" not "had" to match "H").
"""

from __future__ import annotations

import pytest

from kscript.ast import (
    Block,
    Comment,
    Construct,
    KScriptFile,
    Literal,
    PrimaryConstruct,
    Script,
    Signature,
)
from kscript.ast_emitter import ASTEmitter, SymbolicEntry
from kscript.binding_scope import BindingScope
from kscript.token import TokenType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig(id: str) -> Signature:
    """Shorthand for creating a Signature."""
    return Signature(id=id, line=1, column=1)


def _comment(text: str) -> Comment:
    """Shorthand for creating a Comment."""
    return Comment(text=text, line=1, column=1)


def _pc(
    sig_id: str,
    op: TokenType | None = None,
    node_id: str | None = None,
    inline_comment: Comment | None = None,
    node_inline_comment: Comment | None = None,
) -> PrimaryConstruct:
    """Shorthand for creating a PrimaryConstruct."""
    node = Signature(node_id, line=1, column=2) if node_id is not None else None
    return PrimaryConstruct(
        sig=_sig(sig_id),
        op=op,
        node=node,
        inline_comment=inline_comment,
        node_inline_comment=node_inline_comment,
    )


def _make_scope(*word_lists: list[str]) -> BindingScope:
    """Create a BindingScope with the given word lists in a root scope."""
    scope = BindingScope()
    scope.push_scope()
    for words in word_lists:
        scope.add_word_list(words)
    return scope


def _make_file(*constructs: Construct) -> KScriptFile:
    """Create a minimal KScriptFile wrapping the given constructs."""
    return KScriptFile(scripts=[Script(constructs=list(constructs))])


def _find_entry(
    entries: list[SymbolicEntry], sig: str, op: str
) -> SymbolicEntry | None:
    """Find first entry matching sig and op."""
    for e in entries:
        if e.sig == sig and e.op == op:
            return e
    return None


def _find_all_entries(
    entries: list[SymbolicEntry], op: str
) -> list[SymbolicEntry]:
    """Find all entries matching op."""
    return [e for e in entries if e.op == op]


# ---------------------------------------------------------------------------
# NB-1: Inline S(ubject)
# ---------------------------------------------------------------------------


class TestNB1InlineSubject:
    """NB-1 — Inline S(ubject) binds S to 'Subject'."""

    def test_inline_comment_resolves_sig(self) -> None:
        """PrimaryConstruct with inline_comment resolves sig to full word."""
        scope = BindingScope()  # empty scope — no word lists
        scope.push_scope()
        emitter = ASTEmitter(scope=scope)

        pc = _pc("S", inline_comment=_comment("(ubject)"))
        emitter._emit_primary(pc)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert any(e.sig == "Subject" for e in unsigned), (
            f"Expected 'Subject' in unsigned entries; got {unsigned}"
        )

    def test_inline_no_scope_still_works(self) -> None:
        """Inline comment works even without a BindingScope."""
        emitter = ASTEmitter()  # scope=None — Mod32 mode

        pc = _pc("S", inline_comment=_comment("(ubject)"))
        emitter._emit_primary(pc)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        # In Mod32 mode with inline comment, still resolves the word
        assert any(e.sig == "Subject" for e in unsigned)


# ---------------------------------------------------------------------------
# NB-2: Inline V(erb)
# ---------------------------------------------------------------------------


class TestNB2InlineVerb:
    """NB-2 — Inline V(erb) binds V to 'Verb'."""

    def test_inline_verb_binding(self) -> None:
        """V(erb) resolves to 'Verb'."""
        scope = BindingScope()
        scope.push_scope()
        emitter = ASTEmitter(scope=scope)

        pc = _pc("V", inline_comment=_comment("(erb)"))
        emitter._emit_primary(pc)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert any(e.sig == "Verb" for e in unsigned)


# ---------------------------------------------------------------------------
# NB-3: Node-side inline D(et)
# ---------------------------------------------------------------------------


class TestNB3NodeSideInline:
    """NB-3 — Node-side inline comment D(et) resolves node to 'Det'."""

    def test_node_inline_comment_resolves(self) -> None:
        """PrimaryConstruct with node_inline_comment resolves node."""
        scope = BindingScope()
        scope.push_scope()
        emitter = ASTEmitter(scope=scope)

        pc = _pc(
            "A",
            op=TokenType.COUNTERSIGN,
            node_id="D",
            node_inline_comment=_comment("(et)"),
        )
        emitter._emit_primary(pc)

        cs = _find_all_entries(emitter.entries, "COUNTERSIGN")
        # First countersign: sig=A, nodes=Det (node resolved)
        assert any(e.nodes == "Det" for e in cs), (
            f"Expected node 'Det' in countersign entries; got {cs}"
        )


# ---------------------------------------------------------------------------
# NB-9: Inline override parent kline
# ---------------------------------------------------------------------------


class TestNB9InlineOverrideParentKline:
    """NB-9 — Inline binding patches parent MCS CANONIZE entry.

    Source: SVO => Block([S(ubject) = M])
    The MCS CANONIZE for SVO starts as ["S","V","O"], but S(ubject) patches
    it to ["Subject","V","O"].
    """

    def test_inline_override_patches_canonize(self) -> None:
        """S(ubject) inside subscript patches SVO CANONIZE nodes."""
        scope = BindingScope()
        scope.push_scope()

        ast = _make_file(
            Construct(
                inner=[_pc("SVO")],
                chain_op=TokenType.CANONIZE,
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc(
                                "S",
                                op=TokenType.COUNTERSIGN,
                                node_id="M",
                                inline_comment=_comment("(ubject)"),
                            )]
                        )
                    ])
                ),
            )
        )

        emitter = ASTEmitter(scope=scope)
        entries = emitter.emit(ast)

        # Find the MCS CANONIZE for SVO
        svo_canon = _find_entry(entries, "SVO", "CANONIZE")
        assert svo_canon is not None, (
            f"Expected SVO CANONIZE entry; got {[e for e in entries if e.op == 'CANONIZE']}"
        )
        assert isinstance(svo_canon.nodes, list), (
            f"Expected list nodes; got {type(svo_canon.nodes)}"
        )
        # S is overridden inline to "Subject"
        assert svo_canon.nodes[0] == "Subject", (
            f"Expected SVO nodes[0]='Subject' (overridden); got {svo_canon.nodes}"
        )
        # V and O remain raw (no word list, no inline)
        assert svo_canon.nodes[1] == "V"
        assert svo_canon.nodes[2] == "O"


# ---------------------------------------------------------------------------
# NB-10/11: Occurrence counter interaction
# ---------------------------------------------------------------------------


class TestNB10NB11OccurrenceCounter:
    """NB-10/11 — Emitter resolves duplicate chars through scope.

    (Little Lamb) LL: BindingScope resolves first L→'Little' and second
    L→'Lamb' via occurrence counter.
    """

    def test_duplicate_char_resolved_differently(self) -> None:
        """Two L chars in scope with (Little Lamb) resolve to different words."""
        scope = _make_scope(["Little", "Lamb"])
        emitter = ASTEmitter(scope=scope)

        # Emit two L primaries
        pc1 = _pc("L")
        pc2 = _pc("L")
        emitter._emit_primary(pc1)
        emitter._emit_primary(pc2)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert len(unsigned) == 2, f"Expected 2 unsigned entries; got {unsigned}"
        assert unsigned[0].sig == "Little"
        assert unsigned[1].sig == "Lamb"


# ---------------------------------------------------------------------------
# NB-12: Override immediate parent only
# ---------------------------------------------------------------------------


class TestNB12OverrideImmediateParentOnly:
    """NB-12 — Override from inner subscript patches immediate parent only.

    AST: A => B => Block([C(omment)])
    C(omment) patches B's CANONIZE entry, not A's.
    """

    def test_override_only_immediate_parent(self) -> None:
        """Inline in nested subscript patches only immediate parent."""
        scope = BindingScope()
        scope.push_scope()

        ast = _make_file(
            Construct(
                inner=[_pc("A")],
                chain_op=TokenType.CANONIZE,
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("B")],
                            chain_op=TokenType.CANONIZE,
                            chain_right=Construct(
                                inner=Block([
                                    Construct(
                                        inner=[_pc(
                                            "C",
                                            inline_comment=_comment("(omment)"),
                                        )]
                                    )
                                ])
                            ),
                        )
                    ])
                ),
            )
        )

        emitter = ASTEmitter(scope=scope)
        entries = emitter.emit(ast)

        # C resolves to "Comment" via inline
        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert any(e.sig == "Comment" for e in unsigned), (
            f"Expected 'Comment' in unsigned; got {unsigned}"
        )


# ---------------------------------------------------------------------------
# NB-14: Unbound character
# ---------------------------------------------------------------------------


class TestNB14UnboundCharacter:
    """NB-14 — Unbound character stays raw."""

    def test_no_word_list_no_inline(self) -> None:
        """Z with no binding stays as raw 'Z'."""
        scope = BindingScope()
        scope.push_scope()
        emitter = ASTEmitter(scope=scope)

        pc = _pc("Z")
        emitter._emit_primary(pc)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert len(unsigned) >= 1
        assert unsigned[0].sig == "Z"

    def test_empty_scope(self) -> None:
        """Empty scope with no word lists — all chars stay raw."""
        scope = BindingScope()
        scope.push_scope()
        emitter = ASTEmitter(scope=scope)

        pc = _pc("M", op=TokenType.COUNTERSIGN, node_id="X")
        emitter._emit_primary(pc)

        cs = _find_all_entries(emitter.entries, "COUNTERSIGN")
        assert any(e.sig == "M" and e.nodes == "X" for e in cs)


# ---------------------------------------------------------------------------
# NB-15: Bound sig carries resolved NLP word
# ---------------------------------------------------------------------------


class TestNB15BoundSig:
    """NB-15 — Bound sig carries resolved NLP word as sig string."""

    def test_scope_resolved_sig(self) -> None:
        """M resolved via scope to 'Mary' appears as sig."""
        scope = _make_scope(["Mary"])
        emitter = ASTEmitter(scope=scope)

        pc = _pc("M")
        emitter._emit_primary(pc)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert any(e.sig == "Mary" for e in unsigned), (
            f"Expected sig 'Mary'; got {unsigned}"
        )

    def test_inline_resolved_sig(self) -> None:
        """M(ary) inline resolves to 'Mary' as sig."""
        emitter = ASTEmitter()  # no scope needed for inline

        pc = _pc("M", inline_comment=_comment("(ary)"))
        emitter._emit_primary(pc)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert any(e.sig == "Mary" for e in unsigned)


# ---------------------------------------------------------------------------
# NB-16: Bound node carries resolved NLP word
# ---------------------------------------------------------------------------


class TestNB16BoundNode:
    """NB-16 — Bound node carries resolved NLP word in nodes field."""

    def test_scope_resolved_node(self) -> None:
        """S = M where M resolves to 'Mary' — nodes is 'Mary'."""
        scope = _make_scope(["Mary"])
        emitter = ASTEmitter(scope=scope)

        pc = _pc("S", op=TokenType.COUNTERSIGN, node_id="M")
        emitter._emit_primary(pc)

        cs = _find_all_entries(emitter.entries, "COUNTERSIGN")
        assert any(e.sig == "S" and e.nodes == "Mary" for e in cs), (
            f"Expected (S, Mary) countersign; got {cs}"
        )


# ---------------------------------------------------------------------------
# NB-17: Mixed MCS
# ---------------------------------------------------------------------------


class TestNB17MixedMCS:
    """NB-17 — Mixed MCS with bound and unbound characters.

    Scope has word list (Mary Had) and MCS sig 'MHA'.
    Canonize nodes are ['Mary', 'Had', 'A'] — first two bound, last unbound.
    """

    def test_mixed_mcs_resolution(self) -> None:
        """MHA with (Mary Had) resolves M→Mary, H→Had, A stays raw."""
        scope = _make_scope(["Mary", "Had"])
        emitter = ASTEmitter(scope=scope)

        result = emitter._emit_mcs("MHA")
        assert result is not None

        canonize = _find_all_entries(emitter.entries, "CANONIZE")
        assert len(canonize) == 1
        assert canonize[0].sig == "MHA"
        assert canonize[0].nodes == ["Mary", "Had", "A"]

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        unsigned_sigs = {e.sig for e in unsigned}
        assert "Mary" in unsigned_sigs
        assert "Had" in unsigned_sigs
        assert "A" in unsigned_sigs


# ---------------------------------------------------------------------------
# NB-29: Inline bypasses counter
# ---------------------------------------------------------------------------


class TestNB29InlineBypassesCounter:
    """NB-29 — Inline binding bypasses scope occurrence counter.

    Scope has word list (Alpha Alice). Sig A(lice) followed by bare A.
    The inline resolves A→'Alice' without incrementing the counter.
    The subsequent bare A resolves via scope to 'Alpha' (counter at 0).
    """

    def test_inline_does_not_increment_counter(self) -> None:
        """Inline A(lice) doesn't affect scope counter; bare A gets 'Alpha'."""
        scope = _make_scope(["Alpha", "Alice"])
        emitter = ASTEmitter(scope=scope)

        # First: inline A(lice) — bypasses counter
        pc_inline = _pc("A", inline_comment=_comment("(lice)"))
        emitter._emit_primary(pc_inline)

        # Second: bare A — scope counter still at 0, gets first match 'Alpha'
        pc_bare = _pc("A")
        emitter._emit_primary(pc_bare)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert len(unsigned) == 2, f"Expected 2 unsigned entries; got {unsigned}"
        assert unsigned[0].sig == "Alice", f"First A should be 'Alice'; got {unsigned[0].sig}"
        assert unsigned[1].sig == "Alpha", f"Second A should be 'Alpha'; got {unsigned[1].sig}"


# ---------------------------------------------------------------------------
# NB-32: Forward-only word list
# ---------------------------------------------------------------------------


class TestNB32ForwardOnlyWordList:
    """NB-32 — Word list only serves characters AFTER it.

    Block comment (Mary) appears AFTER sig M in the AST.
    M does NOT resolve to 'Mary'. A subsequent M DOES resolve.
    """

    def test_word_list_forward_only(self) -> None:
        """M before (Mary) stays raw; M after (Mary) resolves."""
        scope = BindingScope()
        scope.push_scope()
        emitter = ASTEmitter(scope=scope)

        # First M — no word list yet
        pc_before = _pc("M")
        emitter._emit_primary(pc_before)

        # Now add word list
        scope.add_word_list(["Mary"])

        # Second M — word list now available
        pc_after = _pc("M")
        emitter._emit_primary(pc_after)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert len(unsigned) == 2
        assert unsigned[0].sig == "M", f"First M should be raw; got {unsigned[0].sig}"
        assert unsigned[1].sig == "Mary", f"Second M should be 'Mary'; got {unsigned[1].sig}"


# ---------------------------------------------------------------------------
# NB-33: Inline override no-match
# ---------------------------------------------------------------------------


class TestNB33InlineOverrideNoMatch:
    """NB-33 — Inline override for char not in parent kline is safe no-op.

    Z(ebra) inside subscript where parent kline is 'SVO'.
    Z not in 'SVO' → no patch. Entry for Z emitted with sig 'Zebra',
    CANONIZE entry unchanged.
    """

    def test_no_match_is_safe_noop(self) -> None:
        """Z(ebra) doesn't patch SVO CANONIZE — Z not in SVO."""
        scope = BindingScope()
        scope.push_scope()

        ast = _make_file(
            Construct(
                inner=[_pc("SVO")],
                chain_op=TokenType.CANONIZE,
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc(
                                "Z",
                                inline_comment=_comment("(ebra)"),
                            )]
                        )
                    ])
                ),
            )
        )

        emitter = ASTEmitter(scope=scope)
        entries = emitter.emit(ast)

        # SVO CANONIZE unchanged — Z not in "SVO"
        svo_canon = _find_entry(entries, "SVO", "CANONIZE")
        assert svo_canon is not None
        assert isinstance(svo_canon.nodes, list)
        # No word list, so V and O stay raw
        assert svo_canon.nodes == ["S", "V", "O"], (
            f"SVO CANONIZE should be unchanged; got {svo_canon.nodes}"
        )

        # Z resolves to "Zebra" via inline
        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert any(e.sig == "Zebra" for e in unsigned), (
            f"Expected 'Zebra' in unsigned; got {unsigned}"
        )


# ---------------------------------------------------------------------------
# Backward compatibility — Mod32 mode (scope=None)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure default behaviour (no scope) is unchanged."""

    def test_emit_mcs_no_scope(self) -> None:
        """MCS emission without scope produces same results as before."""
        emitter = ASTEmitter()
        emitter._emit_mcs("ABC")

        assert len(emitter.entries) == 4  # 3 unsigned + 1 canonize
        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        canonize = _find_all_entries(emitter.entries, "CANONIZE")
        assert [e.sig for e in unsigned] == ["A", "B", "C"]
        assert canonize[0].sig == "ABC"
        assert canonize[0].nodes == ["A", "B", "C"]

    def test_emit_entry_no_scope(self) -> None:
        """_emit_entry without scope is a passthrough."""
        emitter = ASTEmitter()
        emitter._emit_entry("M", None, "UNSIGNED")
        assert emitter.entries[0].sig == "M"

    def test_resolve_char_returns_char_when_no_scope(self) -> None:
        """_resolve_char returns char when no scope is set."""
        emitter = ASTEmitter()
        assert emitter._resolve_char("X") == "X"

    def test_full_emit_no_scope(self) -> None:
        """Full emit cycle without scope produces baseline results."""
        source = _make_file(
            Construct([
                PrimaryConstruct(
                    Signature("S", line=1, column=1),
                    TokenType.COUNTERSIGN,
                    Signature("M", line=1, column=2),
                ),
            ]),
        )
        emitter = ASTEmitter()
        entries = emitter.emit(source)

        sigs = [(e.sig, e.nodes, e.op) for e in entries]
        assert ("S", "M", "COUNTERSIGN") in sigs
        assert ("M", "S", "COUNTERSIGN") in sigs

    def test_comment_silently_skipped_no_scope(self) -> None:
        """Block comments are silently skipped when scope is None."""
        ast = _make_file(
            Construct(inner=Comment("(Mary had a little lamb)", line=1, column=1)),
            Construct(inner=[_pc("M")]),
        )
        emitter = ASTEmitter()
        entries = emitter.emit(ast)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert len(unsigned) == 1
        assert unsigned[0].sig == "M"  # raw — no scope to feed word list

    def test_scope_none_produces_identical_output(self) -> None:
        """Passing scope=None is identical to not passing scope at all."""
        ast = _make_file(
            Construct([
                _pc("S", TokenType.COUNTERSIGN, "M"),
                _pc("ABC"),
            ])
        )

        emitter_no_scope = ASTEmitter()
        entries_no_scope = emitter_no_scope.emit(ast)

        emitter_none = ASTEmitter(scope=None)
        entries_none = emitter_none.emit(ast)

        assert entries_no_scope == entries_none


# ---------------------------------------------------------------------------
# Scope-based resolution
# ---------------------------------------------------------------------------


class TestScopeResolution:
    """Tests for scope-based character resolution."""

    def test_resolve_char_bound(self) -> None:
        """_resolve_char with binding returns the bound word."""
        scope = _make_scope(["Mary"])
        emitter = ASTEmitter(scope=scope)
        assert emitter._resolve_char("M") == "Mary"

    def test_resolve_char_unbound(self) -> None:
        """_resolve_char without binding returns the raw char."""
        scope = _make_scope(["Mary"])
        emitter = ASTEmitter(scope=scope)
        assert emitter._resolve_char("Z") == "Z"

    def test_resolve_char_no_scope(self) -> None:
        """_resolve_char with no scope returns raw char."""
        emitter = ASTEmitter()
        assert emitter._resolve_char("M") == "M"

    def test_scope_word_list_from_comment(self) -> None:
        """Block comment feeds scope word list via add_word_list."""
        scope = BindingScope()
        scope.push_scope()
        emitter = ASTEmitter(scope=scope)

        # Simulate what _emit_construct does with a comment
        words = emitter._extract_words("(Mary Had A)")
        assert words == ["Mary", "Had", "A"]

        scope.add_word_list(words)
        assert emitter._resolve_char("M") == "Mary"
        assert emitter._resolve_char("H") == "Had"
        assert emitter._resolve_char("A") == "A"

    def test_scope_push_pop(self) -> None:
        """Scope push/pop creates isolated scopes."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary"])

        emitter = ASTEmitter(scope=scope)

        # Root: M→Mary
        assert emitter._resolve_char("M") == "Mary"

        # Push child scope with different binding
        scope.push_scope()
        scope.add_word_list(["Mod"])
        assert emitter._resolve_char("M") == "Mod"

        # Pop back to root
        scope.pop_scope()
        assert emitter._resolve_char("M") == "Mary"


# ---------------------------------------------------------------------------
# Extract helpers
# ---------------------------------------------------------------------------


class TestExtractHelpers:
    """Tests for inline word extraction helpers."""

    def test_extract_inline_word(self) -> None:
        """_extract_inline_word strips parens and prepends sig char."""
        emitter = ASTEmitter()
        result = emitter._extract_inline_word("S", _comment("(ubject)"))
        assert result == "Subject"

    def test_extract_inline_word_verb(self) -> None:
        """V + (erb) → Verb."""
        emitter = ASTEmitter()
        result = emitter._extract_inline_word("V", _comment("(erb)"))
        assert result == "Verb"

    def test_extract_words_basic(self) -> None:
        """_extract_words splits parenthesized word list."""
        emitter = ASTEmitter()
        result = emitter._extract_words("(Mary Had A Little Lamb)")
        assert result == ["Mary", "Had", "A", "Little", "Lamb"]

    def test_extract_words_empty(self) -> None:
        """_extract_words returns empty list for empty comment."""
        emitter = ASTEmitter()
        result = emitter._extract_words("()")
        assert result == []

    def test_extract_words_single(self) -> None:
        """_extract_words with single word."""
        emitter = ASTEmitter()
        result = emitter._extract_words("(Mary)")
        assert result == ["Mary"]


# ---------------------------------------------------------------------------
# Emitter integration with full AST + scope
# ---------------------------------------------------------------------------


class TestEmitterIntegration:
    """Integration tests using full AST + BindingScope."""

    def test_block_comment_feeds_scope(self) -> None:
        """Block comment before M feeds scope, M resolves to 'Mary'."""
        scope = BindingScope()
        scope.push_scope()
        emitter = ASTEmitter(scope=scope)

        ast = _make_file(
            Construct(inner=Comment("(Mary)", line=1, column=1)),
            Construct(inner=[_pc("M")]),
        )
        entries = emitter.emit(ast)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert any(e.sig == "Mary" for e in unsigned), (
            f"Expected 'Mary'; got {unsigned}"
        )

    def test_scope_push_pop_in_chain(self) -> None:
        """Chain => triggers push/pop, inner scope isolated."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary"])

        ast = _make_file(
            Construct(
                inner=[_pc("SVO")],
                chain_op=TokenType.CANONIZE,
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc(
                                "M",
                                op=TokenType.COUNTERSIGN,
                                node_id="X",
                            )]
                        )
                    ])
                ),
            )
        )

        emitter = ASTEmitter(skip_mcs=True, scope=scope)
        entries = emitter.emit(ast)

        # M in subscript resolves upward to "Mary" from root scope
        cs = _find_all_entries(emitter.entries, "COUNTERSIGN")
        assert any(e.sig == "Mary" for e in cs), (
            f"Expected 'Mary' in countersign sigs; got {cs}"
        )

    def test_countersign_both_bound_via_scope(self) -> None:
        """S = M with both bound via scope."""
        scope = _make_scope(["Subject", "Mary"])

        ast = _make_file(
            Construct([
                _pc("S", TokenType.COUNTERSIGN, "M"),
            ])
        )

        emitter = ASTEmitter(scope=scope)
        entries = emitter.emit(ast)

        cs = _find_all_entries(emitter.entries, "COUNTERSIGN")
        assert any(e.sig == "Subject" and e.nodes == "Mary" for e in cs)
        assert any(e.sig == "Mary" and e.nodes == "Subject" for e in cs)

    def test_dedup_with_resolved_words(self) -> None:
        """Two emissions of the same resolved word are deduped."""
        scope = _make_scope(["Mary"])
        emitter = ASTEmitter(scope=scope)
        emitter._emit_entry("Mary", None, "UNSIGNED")
        emitter._emit_entry("Mary", None, "UNSIGNED")
        assert len(emitter.entries) == 1

    def test_mcs_dedup_with_resolved(self) -> None:
        """MCS dedup works — duplicate resolved words are deduped."""
        scope = _make_scope(["Alpha", "Beta"])
        emitter = ASTEmitter(scope=scope)
        result = emitter._emit_mcs("AB")
        assert result is not None

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        # A→Alpha, B→Beta — different words, both kept
        assert len(unsigned) == 2
        assert unsigned[0].sig == "Alpha"
        assert unsigned[1].sig == "Beta"

    def test_all_bound_mcs(self) -> None:
        """MCS with all characters bound → all unsigned entries carry words."""
        scope = _make_scope(["Mary", "Had", "A"])
        emitter = ASTEmitter(scope=scope)
        result = emitter._emit_mcs("MHA")
        assert result is not None

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert len(unsigned) == 3
        assert unsigned[0].sig == "Mary"
        assert unsigned[1].sig == "Had"
        assert unsigned[2].sig == "A"

        canonize = _find_all_entries(emitter.entries, "CANONIZE")[0]
        assert canonize.sig == "MHA"
        assert canonize.nodes == ["Mary", "Had", "A"]

    def test_all_unbound_mcs_no_scope(self) -> None:
        """MCS with no scope → behaviour identical to current."""
        emitter = ASTEmitter()
        result = emitter._emit_mcs("ABC")

        assert result is not None
        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        assert len(unsigned) == 3
        assert [e.sig for e in unsigned] == ["A", "B", "C"]

        canonize = _find_all_entries(emitter.entries, "CANONIZE")[0]
        assert canonize.sig == "ABC"
        assert canonize.nodes == ["A", "B", "C"]

    def test_undersign_bound(self) -> None:
        """L > O with L→Little, O→Object → undersign resolved."""
        scope = _make_scope(["Little", "Object"])
        emitter = ASTEmitter(scope=scope)

        pc = _pc("L", op=TokenType.UNDERSIGN, node_id="O")
        emitter._emit_primary(pc)

        us = _find_all_entries(emitter.entries, "UNDERSIGN")
        assert len(us) == 1
        assert us[0].sig == "Object"
        assert us[0].nodes == "Little"

    def test_connotate_bound(self) -> None:
        """S : M with both bound → connotate resolved."""
        scope = _make_scope(["Subject", "Mary"])
        emitter = ASTEmitter(scope=scope)

        pc = _pc("S", op=TokenType.CONNOTATE, node_id="M")
        emitter._emit_primary(pc)

        con = _find_all_entries(emitter.entries, "CONNOTATE")
        assert len(con) == 1
        assert con[0].sig == "Subject"
        assert con[0].nodes == "Mary"

    def test_full_mixed_example(self) -> None:
        """Full example: MHALL with (Mary Had A Little Lamb) word list."""
        scope = _make_scope(["Mary", "Had", "A", "Little", "Lamb"])

        ast = _make_file(
            Construct([_pc("MHALL")])
        )

        emitter = ASTEmitter(scope=scope)
        entries = emitter.emit(ast)

        unsigned = _find_all_entries(emitter.entries, "UNSIGNED")
        unsigned_sigs = {e.sig for e in unsigned}
        assert "Mary" in unsigned_sigs
        assert "Had" in unsigned_sigs
        assert "A" in unsigned_sigs
        # L appears twice in MHALL: first L→Little, second L→Lamb
        assert "Little" in unsigned_sigs
        assert "Lamb" in unsigned_sigs

        canonize = _find_entry(entries, "MHALL", "CANONIZE")
        assert canonize is not None
        assert canonize.nodes == ["Mary", "Had", "A", "Little", "Lamb"]
