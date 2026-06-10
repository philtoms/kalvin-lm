"""Tests for BindingScope — lightweight scope stack for NLP binding resolution.

Covers spec rules NB-4 through NB-31 plus edge cases. Each test constructs
a BindingScope directly — no AST, no parser, no tokenizer.

Spec ref: @kscript-nlp-binding v2.0 §3 (rules 2–3), §7.2
"""

import pytest

from kscript.binding_scope import BindingScope


# =============================================================================
# NB-4: Word list matching (first-letter)
# =============================================================================


class TestNB4WordListMatching:
    """First-letter matching against a single word list."""

    def test_match_first_letter(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary", "had", "a", "little", "lamb"])

        assert scope.resolve("M") == "Mary"
        assert scope.resolve("h") == "had"
        assert scope.resolve("a") == "a"

    def test_first_L_match(self) -> None:
        """When multiple words start with L, the first occurrence is returned."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary", "had", "a", "little", "lamb"])

        assert scope.resolve("l") == "little"


# =============================================================================
# NB-7: Multiple word lists in same scope
# =============================================================================


class TestNB7MultipleWordLists:
    """Most-recent-first ordering across word lists in the same scope."""

    def test_second_list_takes_priority(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["first", "word"])
        scope.add_word_list(["second", "entry"])

        # Second list (most recent) is checked first
        assert scope.resolve("s") == "second"
        assert scope.resolve("e") == "entry"

    def test_fallthrough_to_older_list(self) -> None:
        """Characters not found in newer lists fall through to older ones."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["first", "word"])
        scope.add_word_list(["second", "entry"])

        # "F" not in second list → checks first list
        assert scope.resolve("f") == "first"


# =============================================================================
# NB-8: Upward traversal (scope chain)
# =============================================================================


class TestNB8UpwardTraversal:
    """Resolution walks up to outer scopes when inner scope has no match."""

    def test_walk_up_to_outer_scope(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary", "had"])
        scope.push_scope()  # inner scope, no word lists

        assert scope.resolve("M") == "Mary"
        assert scope.resolve("h") == "had"

    def test_inner_scope_shadows_outer(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary"])
        scope.push_scope()
        scope.add_word_list(["Mod"])

        assert scope.resolve("M") == "Mod"


# =============================================================================
# NB-10: Occurrence counter (ambiguous match advances)
# =============================================================================


class TestNB10OccurrenceCounter:
    """Ambiguous matches advance the occurrence counter."""

    def test_ambiguous_advances_counter(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Alice", "Alpha"])

        assert scope.resolve("A") == "Alice"  # counter 0 → 1
        assert scope.resolve("A") == "Alpha"  # counter 1 → 2


# =============================================================================
# NB-11: Duplicate char disambiguation
# =============================================================================


class TestNB11DuplicateCharDisambiguation:
    """L matching 'little' then 'lamb' via occurrence counter."""

    def test_l_disambiguation(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["little", "lamb"])

        assert scope.resolve("l") == "little"  # counter 0 → 1
        assert scope.resolve("l") == "lamb"  # counter 1 → 2


# =============================================================================
# NB-13: Scope restoration (pop restores outer bindings)
# =============================================================================


class TestNB13ScopeRestoration:
    """Popping a scope restores the outer scope's bindings."""

    def test_pop_restores_outer(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary"])
        scope.push_scope()
        scope.add_word_list(["Mod"])

        assert scope.resolve("M") == "Mod"  # inner shadows outer
        scope.pop_scope()
        assert scope.resolve("M") == "Mary"  # outer restored


# =============================================================================
# NB-27: Counter only increments on ambiguous match
# =============================================================================


class TestNB27CounterOnlyOnAmbiguous:
    """Single-match resolution does not increment the counter."""

    def test_single_match_no_increment(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Alpha"])  # single match for A

        assert scope.resolve("A") == "Alpha"
        # Counter did NOT increment, so still matches "Alpha"
        assert scope.resolve("A") == "Alpha"


# =============================================================================
# NB-28: Counter resets on new scope
# =============================================================================


class TestNB28CounterResetsOnNewScope:
    """Each scope has its own independent counter starting at zero."""

    def test_inner_scope_has_own_counter(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Alice", "Alpha"])
        assert scope.resolve("A") == "Alice"  # outer counter → 1

        scope.push_scope()
        scope.add_word_list(["Aardvark", "Antelope"])
        # Inner scope counter at 0
        assert scope.resolve("A") == "Aardvark"  # inner counter → 1
        assert scope.resolve("A") == "Antelope"  # inner counter → 2

        scope.pop_scope()
        # Back to outer scope, counter still at 1
        assert scope.resolve("A") == "Alpha"  # outer counter was 1 → 2

    def test_inner_scope_counter_used_for_inner_word_lists(self) -> None:
        """Inner scope's counter at 0 when it has its own word list."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Alice", "Alpha"])
        scope.resolve("A")  # outer counter → 1

        scope.push_scope()
        scope.add_word_list(["Aardvark", "Antelope"])
        # Uses inner scope's counter (0), not outer's
        assert scope.resolve("A") == "Aardvark"


# =============================================================================
# NB-30: Single match does not increment counter
# =============================================================================


class TestNB30SingleMatchNoIncrement:
    """Repeated resolution of a single-match word always returns the same word."""

    def test_always_same_word(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary"])

        assert scope.resolve("M") == "Mary"
        assert scope.resolve("M") == "Mary"
        assert scope.resolve("M") == "Mary"


# =============================================================================
# NB-31: Counter exceeds available matches → None (unbound)
# =============================================================================


class TestNB31CounterExceedsMatches:
    """When the counter exceeds available matches, returns None."""

    def test_counter_exhaustion(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Alice", "Alpha"])

        assert scope.resolve("A") == "Alice"  # counter → 1
        assert scope.resolve("A") == "Alpha"  # counter → 2
        assert scope.resolve("A") is None  # counter 2, only 2 matches


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_empty_scope_stack_returns_none(self) -> None:
        scope = BindingScope()
        assert scope.resolve("A") is None

    def test_empty_word_list_has_no_effect(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list([])
        assert scope.resolve("A") is None

    def test_no_matching_first_letter(self) -> None:
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary", "had"])
        assert scope.resolve("Z") is None

    def test_case_sensitive_first_letter(self) -> None:
        """Lowercase 'm' does NOT match 'Mary'."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary"])
        assert scope.resolve("m") is None

    def test_pop_empty_stack_raises(self) -> None:
        scope = BindingScope()
        with pytest.raises(AssertionError):
            scope.pop_scope()

    def test_add_word_list_empty_stack_raises(self) -> None:
        scope = BindingScope()
        with pytest.raises(AssertionError):
            scope.add_word_list(["test"])

    def test_empty_string_word_ignored(self) -> None:
        """Empty string words are skipped (no first character to match)."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["", "Mary"])
        assert scope.resolve("M") == "Mary"

    def test_single_char_word(self) -> None:
        """Single-character words match their own letter."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["A"])
        assert scope.resolve("A") == "A"

    def test_multiple_word_lists_counter_accumulates_across_lists(self) -> None:
        """Counter does not reset between word lists within the same scope."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Alice", "Alpha"])  # 2 A-words
        scope.add_word_list(["Ant"])  # 1 A-word in second list

        # Second list checked first: "Ant" is a single match → no counter increment
        assert scope.resolve("A") == "Ant"
        # Still resolving from second list — counter is 0, still "Ant"
        assert scope.resolve("A") == "Ant"

    def test_fallthrough_after_counter_exhausted_in_newer_list(self) -> None:
        """When counter exceeds matches in newer list, falls through to older list."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Alice", "Alpha"])
        scope.add_word_list(["Ant"])

        # Second list (newer) checked first: single match "Ant" → no increment
        assert scope.resolve("A") == "Ant"
        # Still "Ant" — unambiguous, counter stays at 0
        assert scope.resolve("A") == "Ant"

    def test_scope_with_no_word_lists(self) -> None:
        """Scope with no word lists is transparent to upward traversal."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Mary"])
        scope.push_scope()  # empty inner scope

        assert scope.resolve("M") == "Mary"

    def test_deep_nesting(self) -> None:
        """Multiple levels of scope nesting resolve correctly."""
        scope = BindingScope()
        scope.push_scope()
        scope.add_word_list(["Alpha"])
        scope.push_scope()
        scope.push_scope()
        scope.push_scope()

        assert scope.resolve("A") == "Alpha"  # walks up 3 levels
