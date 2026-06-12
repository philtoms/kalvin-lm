"""Lightweight scope stack for NLP binding resolution.

Implements the BindingScope data structure described in spec §10
(NLP Binding Resolution), specifically §10.1 Rule B3 (First-Letter
Matching) and §10.3 (BindingScope API).

Resolution algorithm (§10.1 Rule B3):
  - Walk scopes innermost-first (reversed stack).
  - Within each scope, walk word lists most-recent-first (reversed).
  - For each word list, collect words where ``word[0].lower() == char.lower()``.
  - If matches found: read ``scope.counters[char.lower()]`` (default 0).
  - If counter >= len(matches): skip to next word list (counter exceeded).
  - Otherwise: word = matches[counter].
  - If ambiguous (len(matches) > 1): increment counter.
  - If unambiguous (len(matches) == 1): do NOT increment counter.
  - Return the matched word, or None if no match in any scope.

Rules B1 (once-bound immutability) and B4 (inline override) are
enforced by the ASTEmitter, not by BindingScope.

Counter reset on scope push: calling ``push_scope()`` clears the
occurrence counters in all existing (parent) scopes, so that when
resolution falls through from a child scope to a parent scope, the
parent's counter starts fresh rather than retaining stale state.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Scope:
    """Internal scope representation.

    Attributes:
        word_lists: Ordered collection of word lists added to this scope.
        counters: Per-character occurrence counter for disambiguation.
            Keyed on lowercase character, value is the current counter.
    """

    word_lists: list[list[str]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)


class BindingScope:
    """Lightweight scope stack for NLP binding resolution.

    Manages a stack of scopes, each containing ordered word lists.
    Resolution walks the stack innermost-first, searching word lists
    most-recent-first within each scope, using first-letter matching
    with an occurrence counter for disambiguation.

    Usage::

        scope = BindingScope()
        scope.push_scope()
        scope.add_words(["Mary", "had", "a", "little", "lamb"])
        assert scope.resolve("M") == "Mary"
        assert scope.resolve("m") == "Mary"  # case-insensitive
        assert scope.resolve("L") == "little"

    The caller must call ``push_scope()`` to create the root scope
    before adding word lists or resolving.
    """

    def __init__(self) -> None:
        """Initialize an empty scope stack."""
        self._stack: list[_Scope] = []

    def push_scope(self) -> None:
        """Push a new scope onto the stack.

        Resets occurrence counters in all parent scopes so that
        fallthrough resolution starts fresh.  Each new scope starts
        with empty word lists and counters at zero.
        """
        for s in self._stack:
            s.counters.clear()
        self._stack.append(_Scope())

    def pop_scope(self) -> None:
        """Pop the top scope off the stack.

        Raises:
            AssertionError: If the scope stack is empty.
        """
        assert self._stack, "Cannot pop from empty scope stack"
        self._stack.pop()

    def add_words(self, words: list[str]) -> None:
        """Append a word list to the current (top) scope.

        Multiple calls accumulate — a scope may have multiple word lists.
        Word lists are searched most-recent-first during ``resolve()``.

        Args:
            words: Ordered list of words to add.

        Raises:
            AssertionError: If the scope stack is empty.
        """
        assert self._stack, "No current scope — stack is empty"
        self._stack[-1].word_lists.append(list(words))

    def resolve(self, char: str) -> str | None:
        """Resolve a character to a word by walking the scope stack.

        Walks scopes from innermost to outermost.  For each scope,
        iterates through word lists in reverse order (most-recent-first).
        For each word list, collects words whose first letter matches
        ``char`` (case-insensitive).  Uses the scope's occurrence counter
        for disambiguation per §10.1 Rule B3.

        Args:
            char: Single character to resolve.

        Returns:
            The matched word, or ``None`` if unbound.
        """
        for scope in reversed(self._stack):
            result = self._resolve_in_scope(scope, char)
            if result is not None:
                return result
        return None

    def _resolve_in_scope(self, scope: _Scope, char: str) -> str | None:
        """Try to resolve char within a single scope.

        Iterates through the scope's word lists in reverse order
        (most-recent-first).  Returns the first match found using
        occurrence-counter disambiguation, or None.

        Args:
            scope: The scope to search.
            char: The character to resolve.

        Returns:
            The matched word, or ``None``.
        """
        key = char.lower()
        for word_list in reversed(scope.word_lists):
            matches = [w for w in word_list if w and w[0].lower() == key]
            if not matches:
                continue

            counter = scope.counters.get(key, 0)
            if counter >= len(matches):
                # Counter exceeded — skip to next word list
                continue

            word = matches[counter]
            if len(matches) > 1:
                # Ambiguous: increment counter for next resolution
                scope.counters[key] = counter + 1
            # Unambiguous: counter stays the same
            return word

        return None
