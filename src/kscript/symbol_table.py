"""NLP Symbol Table for KScript binding resolution.

Provides scoped storage for character→word bindings with positional
consumption for word-list claiming. Used by BindingResolver to map
single-character KScript signatures to NLP words.

Re-walk API (KB-166):
  After BindingResolver populates the table and pops all inner scopes,
  leaving only root on the active stack, the emitter needs to re-traverse
  those pre-built scopes.  The re-walk API supports this:

  - ``_all_scopes`` records every scope ever pushed (preserved after pop).
  - ``_scope_index`` maps ``id(scope)`` → index in ``_all_scopes`` for
    O(1) parent lookup during ``exit_scope()``.
  - ``_walk_cursor`` tracks the current position during re-walk.  Default
    ``-1`` means "not in walk mode; use ``_scopes`` stack as before."
  - ``rewind()`` sets the cursor to 0 (root scope), enabling walk mode.
  - ``enter_scope()`` advances the cursor to the next scope in creation
    order.
  - ``exit_scope()`` moves the cursor back to the parent scope's index.
  - ``resolve(char)`` dispatches on walk mode: when cursor ≥ 0 it resolves
    from ``_all_scopes[cursor]``; otherwise from ``current_scope()``.

  Backward compatibility: without calling ``rewind()``, all existing
  behaviour is unchanged.  The active stack (``_scopes``) and all
  public methods (``push_scope``, ``pop_scope``, ``current_scope``,
  ``bind``, ``is_active``) work identically.

Spec ref: @kscript-nlp-binding §6.1
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Binding:
    """A single character→NLP word binding.

    Attributes:
        char: Single character signature, e.g. "M"
        word: NLP word, e.g. "Mary"
        consumed: Whether this binding has been claimed by word-list consumption
    """

    char: str
    word: str
    consumed: bool = False


class Scope:
    """A lexical scope containing character→word bindings.

    Supports duplicate characters via per-character binding lists.
    Each position is a separate Binding, consumed in order by claim_next().

    Attributes:
        parent: Parent scope for lexical chain (None for root)
        bindings: Per-character binding list
        pending_comment: Pending word list awaiting claim by next signature
    """

    def __init__(self, parent: Scope | None = None) -> None:
        self.parent: Scope | None = parent
        self.bindings: dict[str, list[Binding]] = {}
        self.pending_comment: list[str] | None = None

    def bind(self, char: str, word: str) -> None:
        """Add a binding for char in this scope.

        Appends a new Binding to the per-character list, supporting
        duplicate characters (e.g. L#0, L#1 both map to "L").
        """
        binding = Binding(char=char, word=word)
        if char not in self.bindings:
            self.bindings[char] = []
        self.bindings[char].append(binding)

    def resolve(self, char: str) -> str | None:
        """Resolve a character to its NLP word via lexical scope chain.

        Searches this scope for the first unconsumed binding matching char.
        If not found, delegates to parent scope. Returns the word or None.

        Does NOT consume the binding — consumption only happens via claim_next().
        """
        if char in self.bindings:
            for binding in self.bindings[char]:
                if not binding.consumed:
                    return binding.word
        if self.parent is not None:
            return self.parent.resolve(char)
        return None

    def is_bound_to(self, char: str, word: str) -> bool:
        """Check if char is already bound to the given word in this scope chain.

        Searches this scope for the first unconsumed binding matching char
        and compares its word. If not found, delegates to parent scope.

        Returns True if the char is bound to exactly the same word.
        Used by BindingResolver to detect redundant inline annotations
        (e.g. M(ary) when M is already bound to "Mary" in a parent scope).
        """
        if char in self.bindings:
            for binding in self.bindings[char]:
                if not binding.consumed:
                    return binding.word == word
        if self.parent is not None:
            return self.parent.is_bound_to(char, word)
        return False

    def claim_next(self, char: str) -> Binding | None:
        """Claim the next unconsumed binding for char in this scope only.

        Finds the first unconsumed Binding for char, marks it consumed,
        and returns it. Does NOT search parent scopes — claiming is local.

        Returns None if all bindings for char are consumed or char not present.
        """
        if char not in self.bindings:
            return None
        for binding in self.bindings[char]:
            if not binding.consumed:
                binding.consumed = True
                return binding
        return None


class NLPSymbolTable:
    """Maps single-character signatures to NLP words via scoped bindings.

    Maintains a scope stack supporting push/pop for lexical scoping.
    Delegates binding and resolution to the current (top) scope.

    Re-walk mode (KB-166):
      After the BindingResolver populates the table and pops inner scopes,
      call ``rewind()`` to enable walk mode.  Then use ``enter_scope()`` /
      ``exit_scope()`` to navigate the preserved scopes in creation order.
      ``resolve()`` automatically dispatches to the walk-cursor scope when
      in walk mode.

    The symbol table is populated by BindingResolver and read by ASTEmitter
    and TokenEncoder during compilation.
    """

    def __init__(self) -> None:
        self._scopes: list[Scope] = []
        # Re-walk state (KB-166)
        self._all_scopes: list[Scope] = []  # every scope ever pushed, in creation order
        self._scope_index: dict[int, int] = {}  # id(scope) → index in _all_scopes
        self._walk_cursor: int = -1  # -1 = not in walk mode
        self._next_enter: int = -1  # next unvisited scope index in walk mode

    def push_scope(self) -> Scope:
        """Create a new Scope with current top as parent, push onto stack.

        Records the scope in ``_all_scopes`` for later re-walk navigation.

        Returns the newly created scope.
        """
        parent = self._scopes[-1] if self._scopes else None
        scope = Scope(parent=parent)
        self._scopes.append(scope)
        # Record for re-walk (KB-166)
        self._scope_index[id(scope)] = len(self._all_scopes)
        self._all_scopes.append(scope)
        return scope

    def pop_scope(self) -> None:
        """Pop the top scope off the active stack.

        The scope is preserved in ``_all_scopes`` for re-walk navigation.

        Raises:
            AssertionError: If the stack is empty.
        """
        assert self._scopes, "Cannot pop from empty scope stack"
        self._scopes.pop()

    def current_scope(self) -> Scope:
        """Return the top of the scope stack.

        Raises:
            AssertionError: If the stack is empty.
        """
        assert self._scopes, "No current scope — stack is empty"
        return self._scopes[-1]

    def bind(self, char: str, word: str) -> None:
        """Bind char to word in the current scope."""
        self.current_scope().bind(char, word)

    def resolve(self, char: str) -> str | None:
        """Resolve char through the scope chain.

        In walk mode (cursor ≥ 0), resolves from ``_all_scopes[cursor]``.
        Otherwise resolves from ``current_scope()`` (backward compatible).

        Returns the NLP word or None if unbound.
        """
        if self._walk_cursor >= 0:
            return self._all_scopes[self._walk_cursor].resolve(char)
        return self.current_scope().resolve(char)

    def is_bound_to(self, char: str, word: str) -> bool:
        """Check if char is already bound to the given word in the scope chain.

        In walk mode (cursor ≥ 0), checks from ``_all_scopes[cursor]``.
        Otherwise checks from ``current_scope()`` (backward compatible).

        Returns True if the char is bound to exactly the same word, meaning
        an inline annotation creating this binding would be redundant.
        Used to avoid unnecessary shadow bindings in curriculum annotations.
        """
        if self._walk_cursor >= 0:
            return self._all_scopes[self._walk_cursor].is_bound_to(char, word)
        return self.current_scope().is_bound_to(char, word)

    def is_active(self) -> bool:
        """Return True if any scope in the stack has any bindings."""
        return any(
            len(scope.bindings) > 0 and any(len(bl) > 0 for bl in scope.bindings.values())
            for scope in self._scopes
        )

    # ------------------------------------------------------------------
    # Re-walk API (KB-166)
    # ------------------------------------------------------------------

    def rewind(self) -> None:
        """Enable walk mode, starting at the root scope (index 0).

        After calling rewind(), enter_scope() / exit_scope() navigate the
        preserved scope tree in creation order.  resolve() dispatches to
        the walk-cursor scope instead of the active stack.

        Raises:
            AssertionError: If ``_all_scopes`` is empty (no scopes pushed).
        """
        assert self._all_scopes, "Cannot rewind — no scopes recorded"
        self._walk_cursor = 0
        self._next_enter = 1  # next unvisited scope index

    def enter_scope(self) -> None:
        """Advance the walk cursor to the next unvisited scope.

        The cursor always moves forward in ``_all_scopes`` creation order,
        never re-visiting a previously entered scope.  ``_next_enter``
        tracks the high-water mark so sibling scopes are visited correctly.

        Used to navigate into a child scope during re-walk.  The emitter
        calls this before processing a subscript block.

        Raises:
            AssertionError: If not in walk mode or no more scopes to enter.
        """
        assert self._walk_cursor >= 0, "Not in walk mode — call rewind() first"
        assert self._next_enter < len(self._all_scopes), (
            f"enter_scope() would exceed _all_scopes "
            f"(next_enter={self._next_enter}, len={len(self._all_scopes)})"
        )
        self._walk_cursor = self._next_enter
        self._next_enter += 1

    def exit_scope(self) -> None:
        """Move the walk cursor back to the parent scope.

        Looks up the parent of the current scope via ``_scope_index`` for
        O(1) navigation.  If the parent is None (root), sets cursor to 0.
        The ``_next_enter`` counter is unchanged — the next enter_scope()
        call will advance past this exited scope.

        Raises:
            AssertionError: If not in walk mode.
        """
        assert self._walk_cursor >= 0, "Not in walk mode — call rewind() first"
        current = self._all_scopes[self._walk_cursor]
        parent = current.parent
        if parent is None:
            self._walk_cursor = 0
        else:
            self._walk_cursor = self._scope_index[id(parent)]

    def peek_next_bindings(self) -> dict[str, str]:
        """Preview bindings visible in the next child scope without advancing.

        Returns a dict mapping each character to its first unconsumed word
        as seen from the next unvisited scope (``_all_scopes[_next_enter]``).
        This uses the scope's ``resolve()`` method which walks the parent
        chain, so inherited bindings are included.

        This method does NOT advance ``_next_enter`` or ``_walk_cursor`` —
        it is a read-only peek for downward traversal (NB-9).

        Returns:
            Dict of char→word bindings visible in the next scope.
            Empty dict if no more scopes are available.

        Raises:
            AssertionError: If not in walk mode.
        """
        assert self._walk_cursor >= 0, "Not in walk mode — call rewind() first"
        if self._next_enter >= len(self._all_scopes):
            return {}
        scope = self._all_scopes[self._next_enter]
        result: dict[str, str] = {}
        # Collect all unique chars from this scope and its parents
        visited: set[str] = set()
        current: Scope | None = scope
        while current is not None:
            for char in current.bindings:
                if char not in visited:
                    word = scope.resolve(char)
                    if word is not None:
                        result[char] = word
                    visited.add(char)
            current = current.parent
        return result

    @property
    def in_walk_mode(self) -> bool:
        """True when the table is in walk mode (cursor ≥ 0)."""
        return self._walk_cursor >= 0
