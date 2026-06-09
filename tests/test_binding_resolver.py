"""Tests for BindingResolver — NLP binding mechanism (NB-1 through NB-13).

All tests construct ASTs programmatically using Comment, PrimaryConstruct,
Construct, Block, Script, KScriptFile, Signature, and Literal nodes.
No lexer or parser dependency.
"""

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
from kscript.binding_resolver import BindingResolver
from kscript.symbol_table import NLPSymbolTable, Scope


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sig(id: str, line: int = 1, column: int = 1) -> Signature:
    return Signature(id=id, line=line, column=column)


def _pc(
    sig_id: str,
    inline_comment: Comment | None = None,
    op=None,
    node=None,
) -> PrimaryConstruct:
    return PrimaryConstruct(
        sig=_sig(sig_id),
        op=op,
        node=node,
        inline_comment=inline_comment,
    )


def _comment(text: str, line: int = 1, column: int = 1) -> Comment:
    return Comment(text=text, line=line, column=column)


def _make_file(*constructs: Construct) -> KScriptFile:
    return KScriptFile(scripts=[Script(constructs=list(constructs))])


def _resolve(*constructs: Construct) -> NLPSymbolTable:
    """Build a KScriptFile from constructs and resolve it."""
    resolver = BindingResolver()
    return resolver.resolve(_make_file(*constructs))


# =============================================================================
# NB-1 through NB-3: Inline binding
# =============================================================================


class TestInlineBinding:
    """NB-1, NB-2, NB-3 — Inline comment extraction and binding."""

    def test_nb1_inline_binding_subject(self) -> None:
        """S(ubject) binds S to 'Subject'."""
        constructs = [
            Construct(
                inner=[_pc("S", inline_comment=_comment("(ubject)"))]
            )
        ]
        table = _resolve(*constructs)
        assert table.resolve("S") == "Subject"

    def test_nb2_inline_binding_verb(self) -> None:
        """V(erb) binds V to 'Verb'."""
        constructs = [
            Construct(
                inner=[_pc("V", inline_comment=_comment("(erb)"))]
            )
        ]
        table = _resolve(*constructs)
        assert table.resolve("V") == "Verb"

    def test_nb3_inline_binding_det(self) -> None:
        """D(et) binds D to 'Det'."""
        constructs = [
            Construct(
                inner=[_pc("D", inline_comment=_comment("(et)"))]
            )
        ]
        table = _resolve(*constructs)
        assert table.resolve("D") == "Det"


# =============================================================================
# NB-4: Block word list claiming
# =============================================================================


class TestBlockWordListClaiming:
    """NB-4 — Positional zip of word list onto multi-char signature."""

    def test_nb4_block_word_list_claiming(self) -> None:
        """(Mary had a little lamb) followed by MHALL binds all 5 chars."""
        constructs = [
            Construct(inner=_comment("(Mary had a little lamb)")),
            Construct(inner=[_pc("MHALL")]),
        ]
        table = _resolve(*constructs)
        assert table.resolve("M") == "Mary"
        assert table.resolve("H") == "had"
        assert table.resolve("A") == "a"
        assert table.resolve("L") == "little"
        # L appears twice; resolve returns first unconsumed
        assert table.resolve("L") == "little"


# =============================================================================
# NB-5: Word list mismatch
# =============================================================================


class TestWordListMismatch:
    """NB-5 — Mismatched word count leaves comment inert."""

    def test_nb5_word_list_mismatch_inert(self) -> None:
        """(one two three) followed by AB (3 words ≠ 2 chars) → inert."""
        constructs = [
            Construct(inner=_comment("(one two three)")),
            Construct(inner=[_pc("AB")]),
        ]
        table = _resolve(*constructs)
        assert table.resolve("A") is None
        assert table.resolve("B") is None


# =============================================================================
# NB-6: Orphan comment
# =============================================================================


class TestOrphanComment:
    """NB-6 — Comment with no following signature is inert."""

    def test_nb6_orphan_comment_inert(self) -> None:
        """(note) at end of script with no following sig → no bindings."""
        constructs = [
            Construct(inner=_comment("(note)")),
        ]
        table = _resolve(*constructs)
        assert table.resolve("N") is None
        # No bindings at all
        assert not any(
            len(scope.bindings) > 0
            for scope in table._scopes
        )


# =============================================================================
# NB-7: Multiple pending comments
# =============================================================================


class TestMultiplePendingComments:
    """NB-7 — Only the most recent unclaimed comment is pending."""

    def test_nb7_multiple_pending_comments(self) -> None:
        """Two comments then AB — only second comment is pending."""
        constructs = [
            Construct(inner=_comment("(first word)")),
            Construct(inner=_comment("(second word)")),
            Construct(inner=[_pc("AB")]),
        ]
        table = _resolve(*constructs)
        assert table.resolve("A") == "second"
        assert table.resolve("B") == "word"


# =============================================================================
# NB-8: Upward traversal
# =============================================================================


class TestUpwardTraversal:
    """NB-8 — Unbound sig resolves via scope chain to outer binding."""

    def test_nb8_upward_traversal(self) -> None:
        """M in inner subscript resolves to 'Mary' from outer MHALL binding."""
        # Outer: (Mary had a little lamb) + MHALL → binds M→Mary etc.
        # Inner: subscript with bare M (no inline comment)
        constructs = [
            Construct(inner=_comment("(Mary had a little lamb)")),
            Construct(
                inner=[_pc("MHALL")],
                chain_right=Construct(
                    inner=Block([
                        Construct(inner=[_pc("M")]),
                    ])
                ),
            ),
        ]
        table = _resolve(*constructs)

        # After walk, root scope has M→Mary
        assert table.resolve("M") == "Mary"

        # Verify upward traversal: push a child scope (simulating inner subscript)
        # M has no binding in the child scope, but resolves upward
        table.push_scope()
        assert table.resolve("M") == "Mary"  # walks up scope chain
        table.pop_scope()


# =============================================================================
# NB-9: Inline binding in subscript scope
# =============================================================================


class TestInlineBindingInSubscript:
    """NB-9 — Inline binding S(ubject) works correctly inside subscript."""

    def test_nb9_inline_binding_in_subscript_scope(self) -> None:
        """S(ubject) in a subscript scope binds S→'Subject' within that scope.

        Downward traversal (§4.3) is not implemented by the resolver; this
        test verifies the inline binding is correctly scoped.
        """
        # Root: (Mary had a little lamb) + MHALL => Block([S(ubject) = M])
        constructs = [
            Construct(inner=_comment("(Mary had a little lamb)")),
            Construct(
                inner=[_pc("MHALL")],
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("S", inline_comment=_comment("(ubject)"))]
                        ),
                    ])
                ),
            ),
        ]
        table = _resolve(*constructs)

        # Root scope should have MHALL bindings but NOT S (inner scope popped)
        assert table.resolve("M") == "Mary"
        assert table.resolve("S") is None  # S was in inner scope, now popped

        # Verify the scope chain: push child, bind S→"Subject", resolve
        table.push_scope()
        table.bind("S", "Subject")
        assert table.resolve("S") == "Subject"
        assert table.resolve("M") == "Mary"  # still resolves upward
        table.pop_scope()

        # Back at root
        assert table.resolve("S") is None


# =============================================================================
# NB-10: Binding consumption
# =============================================================================


class TestBindingConsumption:
    """NB-10 — Duplicate character bindings consumed in order."""

    def test_nb10_binding_consumption(self) -> None:
        """(Alice Alpha) + AA → A#0→Alice, A#1→Alpha."""
        constructs = [
            Construct(inner=_comment("(Alice Alpha)")),
            Construct(inner=[_pc("AA")]),
        ]
        table = _resolve(*constructs)
        scope = table.current_scope()

        # Two bindings for A, consumed in order
        b0 = scope.claim_next("A")
        assert b0 is not None
        assert b0.word == "Alice"

        b1 = scope.claim_next("A")
        assert b1 is not None
        assert b1.word == "Alpha"

        # No more unconsumed
        assert scope.claim_next("A") is None


# =============================================================================
# NB-11: Duplicate character disambiguation
# =============================================================================


class TestDuplicateCharDisambiguation:
    """NB-11 — L#0→little, L#1→lamb via positional zip."""

    def test_nb11_duplicate_char_disambiguation(self) -> None:
        """(little lamb) + LL → L#0→little, L#1→lamb."""
        constructs = [
            Construct(inner=_comment("(little lamb)")),
            Construct(inner=[_pc("LL")]),
        ]
        table = _resolve(*constructs)
        scope = table.current_scope()

        bindings = scope.bindings["L"]
        assert len(bindings) == 2
        assert bindings[0].word == "little"
        assert bindings[1].word == "lamb"

        # claim_next gives them in order
        assert scope.claim_next("L").word == "little"
        assert scope.claim_next("L").word == "lamb"


# =============================================================================
# NB-12: Lexical shadowing
# =============================================================================


class TestLexicalShadowing:
    """NB-12 — Inner scope M(od) shadows outer M→Mary."""

    def test_nb12_lexical_shadowing(self) -> None:
        """Inner M(od) shadows outer M→Mary within subscript scope."""
        constructs = [
            Construct(inner=_comment("(Mary had a little lamb)")),
            Construct(
                inner=[_pc("MHALL")],
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("M", inline_comment=_comment("(od)"))]
                        ),
                    ])
                ),
            ),
        ]
        table = _resolve(*constructs)

        # After walk, root has M→Mary (inner M→Mod was scoped and popped)
        assert table.resolve("M") == "Mary"

        # Verify shadowing: push child scope, bind M→"Mod"
        table.push_scope()
        table.bind("M", "Mod")
        assert table.resolve("M") == "Mod"  # shadows parent
        table.pop_scope()


# =============================================================================
# NB-13: Scope restoration
# =============================================================================


class TestScopeRestoration:
    """NB-13 — After inner scope exits, outer binding is restored."""

    def test_nb13_scope_restoration(self) -> None:
        """After inner subscript exits, M reverts to 'Mary'."""
        constructs = [
            Construct(inner=_comment("(Mary had a little lamb)")),
            Construct(
                inner=[_pc("MHALL")],
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("M", inline_comment=_comment("(od)"))]
                        ),
                    ])
                ),
            ),
        ]
        table = _resolve(*constructs)

        # Root binding M→Mary is restored after inner scope popped
        assert table.resolve("M") == "Mary"

        # Full shadow/restore cycle
        table.push_scope()
        table.bind("M", "Mod")
        assert table.resolve("M") == "Mod"
        table.pop_scope()
        assert table.resolve("M") == "Mary"  # restored


# =============================================================================
# Additional: Scope depth
# =============================================================================


class TestScopeDepth:
    """Verify single scope per subscript level."""

    def test_scope_stack_balanced_after_walk(self) -> None:
        """After resolution, only root scope remains on the stack."""
        constructs = [
            Construct(inner=_comment("(Mary had a little lamb)")),
            Construct(
                inner=[_pc("MHALL")],
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("S", inline_comment=_comment("(ubject)"))]
                        ),
                        Construct(
                            inner=[_pc("O", inline_comment=_comment("(bject)"))],
                            chain_right=Construct(
                                inner=Block([
                                    Construct(
                                        inner=[_pc("A", inline_comment=_comment("(djective)"))]
                                    ),
                                ])
                            ),
                        ),
                    ])
                ),
            ),
        ]
        table = _resolve(*constructs)
        # Only root scope remains
        assert len(table._scopes) == 1

    def test_block_does_not_push_scope(self) -> None:
        """Block inside chain_right produces exactly one scope, not two.

        This verifies that the resolver does NOT push a scope for Block
        nodes — only chain_right creates scope boundaries.
        """
        # Build AST: A => Block([B])
        constructs = [
            Construct(
                inner=[_pc("A")],
                chain_right=Construct(
                    inner=Block([
                        Construct(inner=[_pc("B")]),
                    ])
                ),
            ),
        ]
        table = _resolve(*constructs)
        # Root scope only
        assert len(table._scopes) == 1

    def test_nested_subscripts_balanced(self) -> None:
        """Nested chain_right pushes/pops scopes correctly."""
        # A => Block([B => Block([C])])
        constructs = [
            Construct(
                inner=[_pc("A")],
                chain_right=Construct(
                    inner=Block([
                        Construct(
                            inner=[_pc("B")],
                            chain_right=Construct(
                                inner=Block([
                                    Construct(inner=[_pc("C")]),
                                ])
                            ),
                        ),
                    ])
                ),
            ),
        ]
        table = _resolve(*constructs)
        assert len(table._scopes) == 1


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Additional edge cases for robustness."""

    def test_empty_file(self) -> None:
        """Empty KScriptFile produces empty table."""
        table = _resolve()
        assert len(table._scopes) == 1  # root scope
        assert not table.is_active()

    def test_comment_clears_pending_on_inline(self) -> None:
        """Inline binding clears the pending word list."""
        constructs = [
            Construct(inner=_comment("(Mary had a little lamb)")),
            Construct(
                inner=[_pc("S", inline_comment=_comment("(ubject)"))]
            ),
            # Pending was cleared by inline, so MHALL won't claim it
            Construct(inner=[_pc("MHALL")]),
        ]
        table = _resolve(*constructs)
        # S bound via inline
        assert table.resolve("S") == "Subject"
        # MHALL didn't claim (pending was cleared)
        assert table.resolve("M") is None

    def test_inline_comment_case_preserved(self) -> None:
        """Case in inline comment is preserved."""
        constructs = [
            Construct(
                inner=[_pc("X", inline_comment=_comment("(YZ)"))]
            ),
        ]
        table = _resolve(*constructs)
        assert table.resolve("X") == "XYZ"

    def test_empty_comment_inert(self) -> None:
        """Empty comment () produces no pending words."""
        constructs = [
            Construct(inner=_comment("()")),
            Construct(inner=[_pc("AB")]),
        ]
        table = _resolve(*constructs)
        assert table.resolve("A") is None
        assert table.resolve("B") is None

    def test_literal_construct_skipped(self) -> None:
        """Literal constructs don't interfere with binding."""
        lit = Literal(id="hello", line=1, column=1)
        constructs = [
            Construct(inner=_comment("(Mary had a little lamb)")),
            Construct(inner=lit),
            Construct(inner=[_pc("MHALL")]),
        ]
        table = _resolve(*constructs)
        # Pending comment survives the literal (not consumed)
        assert table.resolve("M") == "Mary"

    def test_word_list_only_consumed_once(self) -> None:
        """A claimed word list cannot be claimed again."""
        constructs = [
            Construct(inner=_comment("(Alice Alpha)")),
            Construct(inner=[_pc("AA")]),
            # Second MCS sig — pending already cleared
            Construct(inner=[_pc("BB")]),
        ]
        table = _resolve(*constructs)
        # AA claimed the word list
        assert table.resolve("A") == "Alice"
        # BB has no word list to claim
        assert table.resolve("B") is None
