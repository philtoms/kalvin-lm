"""NLP Symbol Table for KScript binding resolution.

Provides scoped storage for character→word bindings with positional
consumption for word-list claiming. Used by BindingResolver to map
single-character KScript signatures to NLP words.

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

    The symbol table is populated by BindingResolver and read by ASTEmitter
    and TokenEncoder during compilation.
    """

    def __init__(self) -> None:
        self._scopes: list[Scope] = []

    def push_scope(self) -> Scope:
        """Create a new Scope with current top as parent, push onto stack.

        Returns the newly created scope.
        """
        parent = self._scopes[-1] if self._scopes else None
        scope = Scope(parent=parent)
        self._scopes.append(scope)
        return scope

    def pop_scope(self) -> None:
        """Pop the top scope off the stack.

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
        """Resolve char through the scope chain starting at current scope.

        Returns the NLP word or None if unbound.
        """
        return self.current_scope().resolve(char)

    def is_active(self) -> bool:
        """Return True if any scope in the stack has any bindings."""
        return any(
            len(scope.bindings) > 0 and any(len(bl) > 0 for bl in scope.bindings.values())
            for scope in self._scopes
        )
