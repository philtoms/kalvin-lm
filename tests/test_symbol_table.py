"""Tests for NLPSymbolTable data structure.

Covers: Binding, Scope, and NLPSymbolTable classes.
Spec ref: @kscript-nlp-binding §6.1, §4.4, §4.5
Test matrix: NB-4, NB-10
"""

from kscript.symbol_table import Binding, NLPSymbolTable, Scope


# =============================================================================
# NB-4: Positional binding
# =============================================================================


class TestPositionalBinding:
    """NB-4: (Mary had a little lamb) + MHALL binds M→Mary, H→had, A→a, L→little, L→lamb."""

    def test_mhall_positional_binding(self) -> None:
        """Bind each char/word in MHALL and verify each resolves correctly."""
        scope = Scope()
        words = ["Mary", "had", "a", "little", "lamb"]
        chars = list("MHALL")

        for char, word in zip(chars, words):
            scope.bind(char, word)

        assert scope.resolve("M") == "Mary"
        assert scope.resolve("H") == "had"
        assert scope.resolve("A") == "a"
        assert scope.resolve("L") == "little"

    def test_each_char_resolves_to_correct_word(self) -> None:
        """Each character independently resolves to its bound word."""
        scope = Scope()
        scope.bind("M", "Mary")
        scope.bind("H", "had")
        scope.bind("A", "a")
        scope.bind("X", "extra")

        assert scope.resolve("M") == "Mary"
        assert scope.resolve("H") == "had"
        assert scope.resolve("A") == "a"
        assert scope.resolve("X") == "extra"


# =============================================================================
# NB-10: Consumption order
# =============================================================================


class TestConsumptionOrder:
    """NB-10: (Alice Alpha) → A#0 binds to "Alice" (claimed), A#1 binds to "Alpha"."""

    def test_duplicate_char_consumption_order(self) -> None:
        """First claim_next returns Alice, second returns Alpha, third returns None."""
        scope = Scope()
        scope.bind("A", "Alice")
        scope.bind("A", "Alpha")

        # First claim
        b1 = scope.claim_next("A")
        assert b1 is not None
        assert b1.word == "Alice"
        assert b1.consumed is True

        # Second claim
        b2 = scope.claim_next("A")
        assert b2 is not None
        assert b2.word == "Alpha"
        assert b2.consumed is True

        # Third claim — no more unconsumed bindings
        b3 = scope.claim_next("A")
        assert b3 is None

    def test_consumed_binding_still_resolves_to_next_unconsumed(self) -> None:
        """After consuming A→Alice, resolve returns A→Alpha."""
        scope = Scope()
        scope.bind("A", "Alice")
        scope.bind("A", "Alpha")

        # resolve returns first unconsumed
        assert scope.resolve("A") == "Alice"

        # consume first
        scope.claim_next("A")

        # resolve now returns next unconsumed
        assert scope.resolve("A") == "Alpha"

        # consume second
        scope.claim_next("A")

        # no more unconsumed
        assert scope.resolve("A") is None


# =============================================================================
# Lexical scoping
# =============================================================================


class TestLexicalScoping:
    """Test scope chain: parent lookup, shadowing, and restoration."""

    def test_parent_scope_resolution(self) -> None:
        """Child scope resolves M→"Mary" via parent chain."""
        parent = Scope()
        parent.bind("M", "Mary")

        child = Scope(parent=parent)
        assert child.resolve("M") == "Mary"

    def test_child_shadowing(self) -> None:
        """Child scope binds M→"Mod" and resolves to "Mod" (shadows parent)."""
        parent = Scope()
        parent.bind("M", "Mary")

        child = Scope(parent=parent)
        child.bind("M", "Mod")

        assert child.resolve("M") == "Mod"

    def test_parent_restored_after_child(self) -> None:
        """After child scope goes away, parent still resolves M→"Mary"."""
        parent = Scope()
        parent.bind("M", "Mary")

        child = Scope(parent=parent)
        child.bind("M", "Mod")

        # Child sees its own binding
        assert child.resolve("M") == "Mod"

        # Parent still sees its own binding
        assert parent.resolve("M") == "Mary"

    def test_grandparent_resolution(self) -> None:
        """Resolve falls through child → parent → grandparent."""
        grandparent = Scope()
        grandparent.bind("X", "grandparent_value")

        parent = Scope(parent=grandparent)
        child = Scope(parent=parent)

        assert child.resolve("X") == "grandparent_value"

    def test_resolve_unbound_returns_none(self) -> None:
        """Resolving a char with no binding anywhere returns None."""
        parent = Scope()
        child = Scope(parent=parent)
        assert child.resolve("Z") is None


# =============================================================================
# Pending comment storage
# =============================================================================


class TestPendingComment:
    """Test pending_comment field on Scope."""

    def test_set_and_read_pending_comment(self) -> None:
        """Set pending_comment and verify it persists."""
        scope = Scope()
        scope.pending_comment = ["Mary", "had"]
        assert scope.pending_comment == ["Mary", "had"]

    def test_clear_pending_comment(self) -> None:
        """Set to None clears pending comment."""
        scope = Scope()
        scope.pending_comment = ["Mary", "had"]
        scope.pending_comment = None
        assert scope.pending_comment is None

    def test_default_pending_comment_is_none(self) -> None:
        """New scope has no pending comment by default."""
        scope = Scope()
        assert scope.pending_comment is None


# =============================================================================
# NLPSymbolTable push/pop
# =============================================================================


class TestNLPSymbolTable:
    """Test NLPSymbolTable scope management."""

    def test_push_pop_basic(self) -> None:
        """Push scope, bind, resolve, pop."""
        table = NLPSymbolTable()
        table.push_scope()
        table.bind("M", "Mary")
        assert table.resolve("M") == "Mary"
        table.pop_scope()

    def test_nested_scopes(self) -> None:
        """Push two scopes, bind in inner, resolve through chain."""
        table = NLPSymbolTable()

        # Root scope
        table.push_scope()
        table.bind("M", "Mary")

        # Inner scope inherits
        table.push_scope()
        assert table.resolve("M") == "Mary"

        # Inner scope shadows
        table.bind("M", "Mod")
        assert table.resolve("M") == "Mod"

        # Pop inner
        table.pop_scope()
        assert table.resolve("M") == "Mary"

        # Pop root
        table.pop_scope()

    def test_pop_empty_stack_raises(self) -> None:
        """Popping an empty scope stack raises AssertionError."""
        table = NLPSymbolTable()
        try:
            table.pop_scope()
            assert False, "Expected AssertionError"
        except AssertionError:
            pass

    def test_current_scope_empty_stack_raises(self) -> None:
        """Accessing current_scope on empty stack raises AssertionError."""
        table = NLPSymbolTable()
        try:
            table.current_scope()
            assert False, "Expected AssertionError"
        except AssertionError:
            pass

    def test_push_scope_returns_scope(self) -> None:
        """push_scope returns the newly created Scope."""
        table = NLPSymbolTable()
        scope = table.push_scope()
        assert isinstance(scope, Scope)
        assert scope is table.current_scope()


# =============================================================================
# is_active
# =============================================================================


class TestIsActive:
    """Test NLPSymbolTable.is_active()."""

    def test_empty_table_not_active(self) -> None:
        """Table with no scopes returns False."""
        table = NLPSymbolTable()
        assert table.is_active() is False

    def test_scope_with_no_bindings_not_active(self) -> None:
        """Pushed scope with no bindings returns False."""
        table = NLPSymbolTable()
        table.push_scope()
        assert table.is_active() is False

    def test_after_binding_is_active(self) -> None:
        """After binding, table is active."""
        table = NLPSymbolTable()
        table.push_scope()
        table.bind("M", "Mary")
        assert table.is_active() is True

    def test_after_pop_to_empty_not_active(self) -> None:
        """After popping all scopes, table is not active."""
        table = NLPSymbolTable()
        table.push_scope()
        table.bind("M", "Mary")
        table.pop_scope()
        assert table.is_active() is False

    def test_parent_scope_keeps_active(self) -> None:
        """After popping inner scope with bindings, outer with bindings stays active."""
        table = NLPSymbolTable()
        table.push_scope()
        table.bind("M", "Mary")
        table.push_scope()
        table.bind("S", "Subject")
        table.pop_scope()
        # Outer scope still has bindings
        assert table.is_active() is True


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge cases for resolve and claim_next."""

    def test_resolve_on_empty_scope(self) -> None:
        """Resolving on scope with no bindings returns None."""
        scope = Scope()
        assert scope.resolve("A") is None

    def test_claim_next_no_bindings(self) -> None:
        """claim_next on char with no bindings returns None."""
        scope = Scope()
        assert scope.claim_next("A") is None

    def test_claim_next_fully_consumed(self) -> None:
        """claim_next on fully consumed char returns None."""
        scope = Scope()
        scope.bind("A", "Alpha")
        b = scope.claim_next("A")
        assert b is not None
        assert scope.claim_next("A") is None

    def test_claim_next_local_only(self) -> None:
        """claim_next does not search parent scope."""
        parent = Scope()
        parent.bind("A", "Alpha")

        child = Scope(parent=parent)
        # Child has no A binding — claim_next returns None
        assert child.claim_next("A") is None

    def test_binding_dataclass_fields(self) -> None:
        """Binding dataclass has correct default consumed=False."""
        b = Binding(char="M", word="Mary")
        assert b.char == "M"
        assert b.word == "Mary"
        assert b.consumed is False

    def test_multiple_chars_independent(self) -> None:
        """Consuming one char doesn't affect another."""
        scope = Scope()
        scope.bind("A", "Alpha")
        scope.bind("B", "Beta")

        scope.claim_next("A")
        assert scope.resolve("B") == "Beta"
        assert scope.resolve("A") is None  # consumed
