"""Lightweight scope stack for NLP binding resolution.

Supersedes the former NLPSymbolTable and BindingResolver with a single data structure
that manages a stack of scopes, each containing ordered word lists.

BindingScope manages a stack of scopes. Each scope holds an ordered
collection of word lists and a per-character occurrence counter.  The
``resolve(char)`` method walks the scope stack from innermost to outermost,
searching each scope's word lists most-recent-first, using first-letter
matching with an occurrence counter for disambiguation.

Key rules (spec @kscript-nlp-binding v2.0 §3 rules 2–3, §7.2):

- First-letter matching: a word matches a character when the first letter
  of the word equals the character, compared case-insensitively
  (``word[0].lower() == char.lower()``).
- Occurrence counter: each scope maintains an independent counter per
  character.  The counter **only** increments on ambiguous matches (when
  multiple words in the same list start with the same letter).  Single
  (unambiguous) matches do not increment the counter.
- Scope stack: ``resolve()`` walks innermost-first.  Each new scope starts
  with all counters at zero.  When resolution finds a match in an outer
  scope, that outer scope's counter is used (not the inner scope's).

This module is the foundation for the simplified single-pass binding
architecture (spec v2.0 §3, §7.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Scope:
    """Internal scope representation.

    Attributes:
        word_lists: Ordered collection of word lists added to this scope.
        counters: Per-character occurrence counter for disambiguation.
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
        scope.add_word_list(["Mary", "had", "a", "little", "lamb"])
        assert scope.resolve("M") == "Mary"
        assert scope.resolve("m") == "Mary"  # case-insensitive match
        assert scope.resolve("L") == "little"

    The caller (Compiler or ASTEmitter) must call ``push_scope()`` to
    create the root scope before adding word lists or resolving.
    """

    def __init__(self) -> None:
        """Initialize an empty scope stack."""
        self._stack: list[_Scope] = []

    def push_scope(self) -> None:
        """Push a new scope onto the stack.

        Each scope starts with an empty word-list collection and an
        occurrence counter dict of ``{char: 0}``.
        """
        self._stack.append(_Scope())

    def pop_scope(self) -> None:
        """Pop the top scope off the stack.

        Raises:
            AssertionError: If the scope stack is empty.
        """
        assert self._stack, "Cannot pop from empty scope stack"
        self._stack.pop()

    def add_word_list(self, words: list[str]) -> None:
        """Append a word list to the current (top) scope.

        Multiple calls accumulate — a scope may have multiple word lists.
        The order matters: word lists are searched most-recent-first
        during ``resolve()``.

        Args:
            words: Ordered list of words to add.

        Raises:
            AssertionError: If the scope stack is empty.
        """
        assert self._stack, "No current scope — stack is empty"
        self._stack[-1].word_lists.append(list(words))

    def resolve(self, char: str) -> str | None:
        """Resolve a character to a word by walking the scope stack.

        Walks the scope stack from innermost to outermost.  For each scope,
        iterates through its word lists in reverse order (most-recent-first).
        For each word list, collects words whose first letter matches
        ``char`` (case-insensitive).  Uses the scope's occurrence counter for
        disambiguation:

        - **Single match** (unambiguous): returns the word at the current
          counter index.  Counter does NOT increment.
        - **Multiple matches** (ambiguous): returns the word at the current
          counter index and increments the counter by 1.
        - **Counter exceeds matches**: skips to the next word list or outer
          scope.
        - **No match found in any scope**: returns ``None``.

        Each scope has its own independent counter dict.  The counter only
        increments on ambiguous matches.  Each new scope starts at counter
        zero.  When resolution finds a match in an outer scope, that outer
        scope's counter is used.

        Args:
            char: Single character to resolve.

        Returns:
            The matched word, or ``None`` if unbound.
        """
        # Walk innermost-first (top of stack first)
        for scope in reversed(self._stack):
            result = self._resolve_in_scope(scope, char)
            if result is not None:
                return result
        return None

    def _resolve_in_scope(self, scope: _Scope, char: str) -> str | None:
        """Try to resolve char within a single scope.

        Iterates through the scope's word lists in reverse order
        (most-recent-first).  Returns the first match found, or None.

        Args:
            scope: The scope to search.
            char: The character to resolve.

        Returns:
            The matched word, or ``None``.
        """
        for word_list in reversed(scope.word_lists):
            matches = [w for w in word_list if w and w[0].lower() == char.lower()]
            if not matches:
                continue

            counter = scope.counters.get(char.lower(), 0)
            is_ambiguous = len(matches) > 1

            if counter < len(matches):
                word = matches[counter]
                if is_ambiguous:
                    scope.counters[char.lower()] = counter + 1
                return word
            # Counter exceeded available matches in this word list;
            # continue to the next (older) word list in this scope.
        return None
