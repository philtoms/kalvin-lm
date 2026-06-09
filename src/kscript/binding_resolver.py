"""Binding resolver for KScript NLP binding.

Walks a KScript AST, processes comment word lists (both block comments and
inline comments on signatures), and builds an NLPSymbolTable mapping
single-character signatures to NLP words.

The resolver performs a single recursive AST walk with lexical scoping
(push/pop around subscript blocks via chain_right), positional word-list
claiming, and inline-binding extraction. It stays entirely in the string
domain — no NLP-BPE tokenisation occurs here.

Algorithm overview (spec @kscript-nlp-binding §6.2–§6.4):

1. Create NLPSymbolTable, push root scope, walk each script's constructs.
2. For each Construct:
   - Comment: extract word list, store as pending_comment (overwrites previous).
   - Block: recurse into block's constructs — no scope push/pop.
   - Literal: skip.
   - list[PrimaryConstruct]: process each PC, then if chain_right exists,
     push scope, recurse, pop scope.
3. For each PrimaryConstruct:
   - Inline comment: extract word, bind each sig char to it, clear pending.
   - Multi-char sig without inline: try to claim pending word list positionally.
   - Single-char sig without inline: no action (resolved at encoding time).
4. On scope exit, discard that scope's bindings. Parent bindings restored.

Spec ref: @kscript-nlp-binding §6
"""

from __future__ import annotations

from .ast import (
    Block,
    Comment,
    Construct,
    KScriptFile,
    Literal,
    PrimaryConstruct,
    Signature,
)
from .symbol_table import NLPSymbolTable


class BindingResolver:
    """Walks a KScript AST and builds an NLP symbol table from comment word lists.

    Usage::

        resolver = BindingResolver()
        table = resolver.resolve(kscript_file)
        word = table.resolve("S")  # "Subject" if bound
    """

    def resolve(self, file: KScriptFile) -> NLPSymbolTable:
        """Build an NLPSymbolTable from a KScriptFile AST.

        Performs a single recursive walk, processing comments as word lists
        and inline comments as direct bindings. Returns a populated symbol
        table with the root scope still on the stack.

        Args:
            file: The parsed KScriptFile AST.

        Returns:
            NLPSymbolTable with bindings populated from the AST.
        """
        table = NLPSymbolTable()
        table.push_scope()  # Root scope
        for script in file.scripts:
            self._resolve_constructs(script.constructs, table)
        # Root scope remains on the stack so downstream consumers can resolve.
        return table

    # ------------------------------------------------------------------
    # Internal walk
    # ------------------------------------------------------------------

    def _resolve_constructs(
        self, constructs: list[Construct], table: NLPSymbolTable
    ) -> None:
        """Process a sequence of constructs."""
        for construct in constructs:
            self._resolve_construct(construct, table)

    def _resolve_construct(
        self, construct: Construct, table: NLPSymbolTable
    ) -> None:
        """Process a single construct."""
        inner = construct.inner

        if isinstance(inner, Comment):
            self._handle_comment(inner, table)
            return

        if isinstance(inner, Block):
            # Block is just a container — no scope push/pop.
            # Scope boundaries are created by chain_right (=>).
            self._resolve_constructs(inner.constructs, table)
            return

        if isinstance(inner, Literal):
            return  # Literals don't participate in binding

        # list[PrimaryConstruct]
        for pc in inner:
            self._resolve_primary(pc, table)

        # Chain right (subscript block) — push one scope
        if construct.chain_right is not None:
            table.push_scope()
            self._resolve_construct(construct.chain_right, table)
            table.pop_scope()

    def _resolve_primary(
        self, pc: PrimaryConstruct, table: NLPSymbolTable
    ) -> None:
        """Process a primary construct for binding."""
        sig = pc.sig.id

        if pc.inline_comment is not None:
            # Inline binding: S(ubject) → bind S to "Subject"
            word = self._extract_inline_word(sig, pc.inline_comment)
            for char in sig:
                table.bind(char, word)
            # Clear any pending word list (inline binding takes precedence)
            table.current_scope().pending_comment = None
        elif len(sig) > 1:
            # Multi-char sig without inline: try to claim pending word list
            self._try_claim_word_list(sig, table)
        # Single-char sig without inline: no action.
        # Resolution happens at encoding time via table.resolve().

        # Node-side inline comment: D(et) on right side of A = D(et)
        if pc.node_inline_comment is not None and isinstance(pc.node, Signature):
            node_id = pc.node.id
            word = self._extract_inline_word(node_id, pc.node_inline_comment)
            for char in node_id:
                table.bind(char, word)
            # Clear any pending word list (inline binding takes precedence)
            table.current_scope().pending_comment = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _handle_comment(self, comment: Comment, table: NLPSymbolTable) -> None:
        """Store comment as pending word list for the current scope.

        Overwrites any previous pending comment — only the most recent
        unclaimed comment is available (§3.3).
        """
        words = self._extract_words(comment.text)
        if words:
            table.current_scope().pending_comment = words

    def _extract_inline_word(self, sig_char: str, comment: Comment) -> str:
        """Extract word from an inline comment.

        Strips outer parentheses from comment text and prepends sig_char.
        E.g. "S" + "(ubject)" → "Subject", "V" + "(erb)" → "Verb".
        Case is preserved.
        """
        text = comment.text
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return sig_char + text

    def _extract_words(self, comment_text: str) -> list[str]:
        """Extract word list from a block comment.

        Strips outer parentheses and splits on whitespace.
        E.g. "(Mary had a little lamb)" → ["Mary", "had", "a", "little", "lamb"].
        Returns empty list for empty comments.
        """
        text = comment_text.strip()
        if text.startswith("("):
            text = text[1:]
        if text.endswith(")"):
            text = text[:-1]
        text = text.strip()
        if not text:
            return []
        return text.split()

    def _try_claim_word_list(self, sig: str, table: NLPSymbolTable) -> None:
        """Try to claim the pending word list for this multi-char signature.

        If the current scope has a pending word list and the word count
        matches the character count, performs a positional zip binding each
        sig[i] → words[i]. Clears the pending word list on successful claim.

        If counts don't match, does nothing — the comment is inert (§3.3).
        """
        scope = table.current_scope()
        if scope.pending_comment is None:
            return
        words = scope.pending_comment
        if len(words) != len(sig):
            return  # Mismatch — comment is inert
        # Positional zip
        for char, word in zip(sig, words):
            table.bind(char, word)
        scope.pending_comment = None  # Consumed
