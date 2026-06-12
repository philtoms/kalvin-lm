"""Tests for BindingScope — NLP binding resolution scope stack.

Covers acceptance criteria KS-23 through KS-31 plus edge cases:
case-insensitive matching, counter exceeded, multiple word lists,
empty word lists, and error conditions.

Spec reference: §10 (NLP Binding Resolution), §10.1 (Rules B1–B4), §10.3 (BindingScope API).
"""

import pytest

from ks.binding_scope import BindingScope


# ---------------------------------------------------------------------------
# KS-23: First-letter matching
# ---------------------------------------------------------------------------

class TestFirstLetterMatching:
    """KS-23: Words are matched by their first letter (case-insensitive)."""

    def test_mary(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary", "Had", "A", "Little", "Lamb"])
        assert bs.resolve("M") == "Mary"

    def test_had(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary", "Had", "A", "Little", "Lamb"])
        assert bs.resolve("H") == "Had"

    def test_a(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary", "Had", "A", "Little", "Lamb"])
        assert bs.resolve("A") == "A"

    def test_l(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary", "Had", "A", "Little", "Lamb"])
        assert bs.resolve("L") == "Little"


# ---------------------------------------------------------------------------
# KS-24: Occurrence counter disambiguation
# ---------------------------------------------------------------------------

class TestOccurrenceCounter:
    """KS-24: Ambiguous matches (multiple L-words) use occurrence counter."""

    def test_first_l_returns_little(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary", "Had", "A", "Little", "Lamb"])
        assert bs.resolve("L") == "Little"

    def test_second_l_returns_lamb(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary", "Had", "A", "Little", "Lamb"])
        bs.resolve("L")  # Little
        assert bs.resolve("L") == "Lamb"

    def test_counter_only_increments_on_ambiguous(self):
        """Verify counter does not advance for unambiguous single-match chars."""
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary", "Had", "A", "Little", "Lamb"])
        # M is unambiguous — resolving multiple times returns same word
        assert bs.resolve("M") == "Mary"
        assert bs.resolve("M") == "Mary"
        assert bs.resolve("M") == "Mary"


# ---------------------------------------------------------------------------
# KS-25: Unambiguous match does NOT increment counter (inline bypass property)
# ---------------------------------------------------------------------------

class TestInlineBypassProperty:
    """KS-25: Unambiguous match bypasses counter — same word returned repeatedly."""

    def test_single_match_returns_same_word_repeatedly(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Apple", "Banana", "Cherry"])
        # Only one A-word: Apple
        assert bs.resolve("A") == "Apple"
        assert bs.resolve("A") == "Apple"
        assert bs.resolve("A") == "Apple"

    def test_single_match_counter_stays_at_zero(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Only"])
        assert bs.resolve("O") == "Only"
        assert bs.resolve("O") == "Only"


# ---------------------------------------------------------------------------
# KS-27: Scope inheritance (inner scope falls through to outer)
# ---------------------------------------------------------------------------

class TestScopeInheritance:
    """KS-27: Inner scope without a binding inherits from outer scope."""

    def test_inner_finds_outer_word(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary"])
        bs.push_scope()
        # Inner scope has no M-words
        assert bs.resolve("M") == "Mary"


# ---------------------------------------------------------------------------
# KS-28: Scope shadowing
# ---------------------------------------------------------------------------

class TestScopeShadowing:
    """KS-28: Inner scope binding shadows outer scope for same character."""

    def test_inner_shadows_outer(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary"])
        bs.push_scope()
        bs.add_words(["Michael"])
        assert bs.resolve("M") == "Michael"

    def test_outer_visible_after_pop(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary"])
        bs.push_scope()
        bs.add_words(["Michael"])
        assert bs.resolve("M") == "Michael"
        bs.pop_scope()
        assert bs.resolve("M") == "Mary"


# ---------------------------------------------------------------------------
# KS-29: Counter reset on new scope
# ---------------------------------------------------------------------------

class TestCounterReset:
    """KS-29: Each new scope has independent counters."""

    def test_new_scope_independent_counter(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Little", "Lamb"])
        # Advance L-counter in outer scope
        assert bs.resolve("L") == "Little"
        assert bs.resolve("L") == "Lamb"

        # Push new scope — counters are reset in all parent scopes
        bs.push_scope()
        # Inner scope has no L-words, so it falls through to outer
        # Outer scope's counter was reset to 0 on push_scope
        result = bs.resolve("L")
        assert result == "Little"  # Counter reset to 0, first match again

    def test_new_scope_with_own_words(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Little", "Lamb"])
        assert bs.resolve("L") == "Little"
        assert bs.resolve("L") == "Lamb"

        # New scope with its own L-words
        bs.push_scope()
        bs.add_words(["Lemon"])
        assert bs.resolve("L") == "Lemon"


# ---------------------------------------------------------------------------
# KS-30: Unbound returns None
# ---------------------------------------------------------------------------

class TestUnboundReturnsNone:
    """KS-30: Resolving an unbound character returns None."""

    def test_no_match(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Mary", "Had"])
        assert bs.resolve("Z") is None

    def test_empty_scope(self):
        bs = BindingScope()
        bs.push_scope()
        assert bs.resolve("A") is None

    def test_no_scopes(self):
        bs = BindingScope()
        assert bs.resolve("A") is None


# ---------------------------------------------------------------------------
# KS-31: Inert annotation (no matching chars → no side effects)
# ---------------------------------------------------------------------------

class TestInertAnnotation:
    """KS-31: No matching characters causes no side effects on counters."""

    def test_no_match_no_counter_side_effect(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Apple", "Banana", "Cherry"])
        # Z has no match
        assert bs.resolve("Z") is None
        # Resolving A afterward should still work — counter unaffected
        assert bs.resolve("A") == "Apple"
        assert bs.resolve("A") == "Apple"

    def test_inert_word_list_no_counter_effects(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Apple", "Banana"])
        # Resolve Z — no match, no counter change
        assert bs.resolve("Z") is None
        # Now add words with Z
        bs.add_words(["Zebra"])
        assert bs.resolve("Z") == "Zebra"


# ---------------------------------------------------------------------------
# Case-insensitive matching
# ---------------------------------------------------------------------------

class TestCaseInsensitive:
    """First-letter matching is case-insensitive."""

    def test_lowercase_word_uppercase_char(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["mary", "had"])
        assert bs.resolve("M") == "mary"

    def test_lowercase_char(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["mary", "had"])
        assert bs.resolve("m") == "mary"

    def test_mixed_case_words(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Alpha", "beta", "Gamma"])
        assert bs.resolve("a") == "Alpha"
        assert bs.resolve("B") == "beta"
        assert bs.resolve("g") == "Gamma"


# ---------------------------------------------------------------------------
# Counter exceeded
# ---------------------------------------------------------------------------

class TestCounterExceeded:
    """When counter exceeds available matches, returns None."""

    def test_third_resolve_returns_none(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Alice", "Alpha"])
        assert bs.resolve("A") == "Alice"
        assert bs.resolve("A") == "Alpha"
        assert bs.resolve("A") is None

    def test_counter_exceeded_falls_to_next_word_list(self):
        """If counter exceeded in one word list, continues to the next."""
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Alice", "Alpha"])
        assert bs.resolve("A") == "Alice"
        assert bs.resolve("A") == "Alpha"
        # Counter at 2, exceeded for first list — but no second list yet
        assert bs.resolve("A") is None

    def test_counter_exceeded_no_fallthrough_same_scope(self):
        """Counter is per-scope (shared across word lists).

        Once exhausted in newer list, older list in same scope also sees
        the advanced counter and is skipped.
        """
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["OldAardvark"])  # older list
        bs.add_words(["Alice", "Alpha"])  # newer list (searched first)
        assert bs.resolve("A") == "Alice"
        assert bs.resolve("A") == "Alpha"
        # Counter at 2, shared — OldAardvark also skipped (counter >= 1)
        assert bs.resolve("A") is None

    def test_counter_exceeded_falls_through_to_different_scope(self):
        """Counter exceeded in inner scope falls to outer scope."""
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Aardvark"])
        bs.push_scope()
        bs.add_words(["Alice", "Alpha"])
        assert bs.resolve("A") == "Alice"
        assert bs.resolve("A") == "Alpha"
        # Inner scope counter exceeded — outer scope has own counter at 0
        assert bs.resolve("A") == "Aardvark"


# ---------------------------------------------------------------------------
# Multiple word lists in same scope (most-recent-first)
# ---------------------------------------------------------------------------

class TestMultipleWordLists:
    """Multiple add_words calls in same scope: most-recent-first search."""

    def test_newer_list_takes_priority(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Alpha"])
        bs.add_words(["Alice"])
        # Most-recent-first: "Alice" from newer list
        assert bs.resolve("A") == "Alice"

    def test_unambiguous_exhausted_then_older_list(self):
        """When newer list has no A-words, older list is searched."""
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Alpha"])
        bs.add_words(["Beta", "Gamma"])
        # Newer list has no A-words — falls to older list
        assert bs.resolve("A") == "Alpha"

    def test_older_list_not_searched_if_newer_matches(self):
        """When newer list has a match, older list is not searched."""
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Alpha"])
        bs.add_words(["Alice"])
        # Alice found in newer list — Alpha in older not considered
        assert bs.resolve("A") == "Alice"


# ---------------------------------------------------------------------------
# Empty word list
# ---------------------------------------------------------------------------

class TestEmptyWordList:
    """Empty word lists are skipped during resolution."""

    def test_empty_list_skipped(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words([])
        bs.add_words(["Apple"])
        assert bs.resolve("A") == "Apple"

    def test_only_empty_lists(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words([])
        assert bs.resolve("A") is None


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------

class TestErrorConditions:
    """Assert errors on invalid operations."""

    def test_pop_empty_stack_raises(self):
        bs = BindingScope()
        with pytest.raises(AssertionError):
            bs.pop_scope()

    def test_add_words_empty_stack_raises(self):
        bs = BindingScope()
        with pytest.raises(AssertionError):
            bs.add_words(["test"])

    def test_pop_then_add_raises(self):
        bs = BindingScope()
        bs.push_scope()
        bs.pop_scope()
        with pytest.raises(AssertionError):
            bs.add_words(["test"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_single_character_words(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["A", "B", "C"])
        assert bs.resolve("A") == "A"
        assert bs.resolve("B") == "B"
        assert bs.resolve("C") == "C"

    def test_resolve_after_pop_returns_outer(self):
        """After popping inner scope, outer scope is searched."""
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Ostrich"])
        bs.push_scope()
        bs.add_words(["Owl"])
        assert bs.resolve("O") == "Owl"
        bs.pop_scope()
        assert bs.resolve("O") == "Ostrich"

    def test_word_with_empty_string_is_skipped(self):
        """Empty strings in a word list should be skipped (falsy check)."""
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["", "Apple"])
        assert bs.resolve("A") == "Apple"

    def test_push_pop_push_keeps_outer(self):
        bs = BindingScope()
        bs.push_scope()
        bs.add_words(["Alpha"])
        bs.push_scope()
        bs.pop_scope()
        # Back to outer scope
        assert bs.resolve("A") == "Alpha"
