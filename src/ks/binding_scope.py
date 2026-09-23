"""Lightweight scope stack for word binding resolution.

Implements the BindingScope data structure for Word Binding Resolution:
First-Letter Matching and the BindingScope API.

Resolution algorithm:
  - Walk scopes innermost-first (reversed stack).
  - Within each scope, walk word lists most-recent-first (reversed).
  - For each word list, collect words where ``word[0].lower() == char.lower()``.
  - If matches found: read ``scope.counters[char.lower()]`` (default 0).
  - If counter >= len(matches): skip to next word list (counter exceeded).
  - Otherwise: word = matches[counter].
  - If ambiguous (len(matches) > 1): increment counter.
  - If unambiguous (len(matches) == 1): do NOT increment counter.
  - Return the matched word, or None if no match in any scope.

Sig-case gate: ambient attraction (word lists, resolved memory) applies
only to uppercase single chars. A lowercase single char is a literal
word (the article 'a') and never attracts — a previously declared word
list word cannot rebind it. Explicit overrides still apply to any case
(an authored witness binds its char regardless of case).

Rules B1 (once-bound immutability) and B4 (inline override) are
enforced by the ASTEmitter, not by BindingScope.

Binding precedence for a character:
  1. Inline annotation overrides — innermost scope first. An inline
     binding on an uppercase char additionally binds in its immediate
     parent scope (the enclosing scope) — not beyond — so it outlives
     its own scope but does not reach unrelated outer scopes.
  2. Word lists (prefix annotations) — innermost scope first, most-recent
     word list first, occurrence-counter disambiguation.
  3. Resolved bindings — a file-level memory of char → word populated
     by every successful resolution, consulted after all annotations.

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
        overrides: Inline (item) word bindings, keyed on lowercase char.
            An inline annotation binds unconditionally and overrides any
            outer (word-list) binding for that char (Word Binding rule:
            inline binds tighter than top-level). Checked before word-list
            resolution so every code path that resolves the char (compound char
            expansion, identity emission, node resolution) sees the inline
            word, keeping one token per char.
    """

    word_lists: list[list[str]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    overrides: dict[str, str] = field(default_factory=dict)
    #: Bindings that outlived their own scope, registered by child scopes.
    #: Weaker than this scope's own word lists: consulted only after them.
    weak_overrides: dict[str, str] = field(default_factory=dict)


class BindingScope:
    """Lightweight scope stack for word binding resolution.

    Manages a stack of scopes, each containing ordered word lists.
    Resolution walks the stack innermost-first, searching word lists
    most-recent-first within each scope, using first-letter matching
    with an occurrence counter for disambiguation.

    Usage::

        scope = BindingScope()
        scope.push_scope()
        scope.add_words(["Mary", "had", "a", "little", "lamb"])
        assert scope.resolve("M") == "Mary"
        assert scope.resolve("L") == "little"
        assert scope.resolve("a") is None  # lowercase: literal word,
                                            # never attracts

    The caller must call ``push_scope()`` to create the root scope
    before adding word lists or resolving.
    """

    def __init__(self) -> None:
        """Initialize an empty scope stack."""
        self._stack: list[_Scope] = []
        self._resolved: dict[str, str] = {}

    @property
    def resolved_bindings(self) -> dict[str, str]:
        """File-level char → word memory of every successful resolution."""
        return self._resolved

    @property
    def depth(self) -> int:
        """The current frame's nesting level — the root (global) frame is 0."""
        return len(self._stack) - 1

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

    def counters_snapshot(self) -> list[dict[str, int]]:
        """Snapshot per-scope occurrence counters, innermost last."""
        return [dict(s.counters) for s in self._stack]

    def counters_restore(self, snapshot: list[dict[str, int]]) -> None:
        """Restore occurrence counters from a :meth:`counters_snapshot`.

        Used around decoding-aid resolutions (compound char expansion) so they
        don't consume the occurrence counters that belong to the identity
        occurrences emitted as item klines.
        """
        assert len(snapshot) == len(self._stack)
        for scope, counters in zip(self._stack, snapshot):
            scope.counters = dict(counters)

    def reset_counters(self) -> None:
        """Zero the occurrence counters in every scope — a file boundary.

        An imported module's resolutions must not consume the occurrence
        counters its words owe the importing script (the module's words
        reach it like known_words: a fresh walk of little then lamb).
        """
        for s in self._stack:
            s.counters.clear()

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

    def bind_override(self, char: str, word: str) -> None:
        """Register an inline (item) binding for ``char`` in the current scope.

        An inline annotation binds unconditionally and overrides any outer
        (word-list) binding for that char. Registered in the current (topmost)
        scope so it is honored before word-list resolution wherever the char
        is resolved — including compound char expansion and identity emission, not
        only the inline item's own node. This keeps one token per char: the
        inline word wins everywhere, never competing with a looser binding.
        """
        assert self._stack, "No current scope — stack is empty"
        self._stack[-1].overrides[char.lower()] = word
        # An inline annotation additionally binds uppercase chars in its
        # immediate parent scope — the enclosing scope, not beyond. It
        # outlives its own scope without reaching unrelated outer scopes.
        # It registers there as a *weak* binding: word lists added to that
        # parent later (e.g. a following subscript's own annotation) outrank
        # it, so an outlived binding never masquerades as the parent's own.
        if char.isupper() and len(self._stack) >= 2:
            self._stack[-2].weak_overrides[char.lower()] = word

    def resolve(self, char: str) -> str | None:
        """Resolve a character to a word by walking the scope stack.

        Walks scopes from innermost to outermost.  For each scope,
        iterates through word lists in reverse order (most-recent-first).
        For each word list, collects words whose first letter matches
        ``char`` (case-insensitive).  Uses the scope's occurrence counter
        for disambiguation.

        Args:
            char: Single character to resolve.

        Returns:
            The matched word, or ``None`` if unbound. A lowercase char
            never resolves through the ambient tiers (word lists,
            resolved memory) — only through explicit overrides.
        """
        key = char.lower()
        if not char.isupper():
            # Sig-case gate: ambient attraction is uppercase-only. A
            # lowercase single char is a literal word (the article 'a');
            # word lists and resolved memory must not rebind it.
            # Explicit overrides remain visible to any case (an authored
            # witness binds its char regardless of case).
            for scope in reversed(self._stack):
                if key in scope.overrides:
                    return scope.overrides[key]
            return None
        for scope in reversed(self._stack):
            # Inline (item) bindings override word-list bindings for this char
            # (Word Binding rule: inline binds tighter than top-level). Checked
            # first so every resolution path sees the inline word.
            if key in scope.overrides:
                return scope.overrides[key]
            result = self._resolve_in_scope(scope, char)
            if result is not None:
                self._resolved[key] = result
                return result
            # A binding that outlived its own scope (weak, registered in its
            # parent) applies only if the parent's own word lists have nothing.
            if key in scope.weak_overrides:
                return scope.weak_overrides[key]
        word = self._resolved.get(key)
        if word is not None:
            return word
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
                continue

            word = matches[counter]
            if len(matches) > 1:
                scope.counters[key] = counter + 1
            return word

        return None
