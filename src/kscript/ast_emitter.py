"""AST Emitter — walks KScript AST and yields symbolic entries.

This module is responsible for AST traversal and determining the correct
operator semantics for each construct. It yields symbolic tuples
(sig_str, nodes_strs, op) that a TokenEncoder then converts to token IDs.

NLP binding integration (KB-170):
  When a BindingScope is provided, the emitter resolves single-character
  signatures inline during its single AST walk. Resolution order:

  1. **Inline comment first**: if a PrimaryConstruct has an inline_comment
     (sig-side) or node_inline_comment (node-side), the word is extracted
     and bound immediately — bypassing scope.resolve() and the occurrence
     counter entirely. E.g. S(ubject) → "Subject".

  2. **Scope fallback**: if no inline comment, scope.resolve(char) walks
     the scope stack innermost-first, using first-letter matching with an
     occurrence counter for disambiguation.

  Block comments feed the scope's word list via scope.add_word_list().
  Chain-right (=>) boundaries trigger scope.push_scope()/pop_scope() with
  save/restore of parent kline tracking for Rule 4 override patching.

  **Rule 4 — inline override**: when an inline binding fires inside a
  subscript, it retroactively patches the matching character in the already-
  emitted MCS CANONIZE entry for the parent kline's nodes list.

  Example: SVO => Block([S(ubject) = M])
    Before override: CANONIZE("SVO", ["S","V","O"])
    After override:  CANONIZE("SVO", ["Subject","V","O"])

  Only the immediate parent kline is patched — no propagation beyond one
  level (NB-12). If the inline char is not found in the parent kline, it's
  a safe no-op (NB-33).

  When scope=None (Mod32 mode), all binding logic is skipped — complete
  backward compatibility with existing Mod32 compilation.

No tokenizer dependency — pure AST logic.
"""

from __future__ import annotations

from typing import NamedTuple

from .ast import (
    Block,
    Comment,
    Construct,
    ConstructItem,
    KScriptFile,
    Literal,
    Node,
    PrimaryConstruct,
    Script,
    Signature,
)
from .binding_scope import BindingScope
from .token import TokenType


class SymbolicEntry(NamedTuple):
    """A symbolic (not yet tokenized) compilation entry."""
    sig: str
    nodes: str | None | list[str]
    op: str  # COUNTERSIGN, CANONIZE, CONNOTATE, UNDERSIGN, UNSIGNED, IDENTITY


class ASTEmitter:
    """Walks KScript AST and emits symbolic entries.

    Each emitted entry is a (sig_str, nodes_strs, op) tuple. No token
    encoding happens here — that's TokenEncoder's job.

    Args:
        dev: Enable development/diagnostic mode.
        skip_mcs: Skip MCS decomposition for tokenizers that don't support it.
        scope: Optional BindingScope for NLP binding resolution.
            When provided, single-character signatures are resolved inline
            via scope.resolve() (e.g. M → "Mary"). When None (default),
            all signatures pass through as raw characters — backward
            compatible with existing Mod32 compilation.
    """

    def __init__(self, dev: bool = False, skip_mcs: bool = False,
                 scope: BindingScope | None = None):
        self.entries: list[SymbolicEntry] = []
        self.dev = dev
        self._skip_mcs = skip_mcs
        self._scope = scope
        # Per-call-stack tracking for Rule 4 inline override patching
        self._parent_kline_chars: str | None = None
        self._parent_kline_canonize_idx: int | None = None
        self._sig_levels = {
            "COUNTERSIGN": "S1",
            "CANONIZE": "S2",
            "CONNOTATE": "S3",
            "UNDERSIGN": "S1",
            "UNSIGNED": "S4",
            "IDENTITY": "S1",
        }
        # Symbolic dedup — on (sig_str, nodes_strs) before encoding
        self._seen: set[tuple[str, None | str | tuple[str, ...]]] = set()

    def _resolve_char(self, char: str) -> str:
        """Resolve a single character via the BindingScope.

        When the scope has a binding for *char*, returns the resolved word
        (e.g. "Mary" for "M").  When the scope is absent (Mod32 mode) or
        the character is unbound, returns *char* unchanged.
        """
        if self._scope is not None:
            word = self._scope.resolve(char)
            if word is not None:
                return word
        return char

    def emit(self, file: KScriptFile) -> list[SymbolicEntry]:
        """Walk a KScriptFile and collect symbolic entries."""
        for script in file.scripts:
            self._emit_script(script)
        return self.entries

    def _emit_script(self, script: Script) -> None:
        for construct in script.constructs:
            self._emit_construct(construct)

    def _emit_construct(self, construct: Construct) -> None:
        if isinstance(construct.inner, Block):
            for c in construct.inner.constructs:
                self._emit_construct(c)
            return

        if isinstance(construct.inner, Comment):
            # Comments feed the scope's word list via add_word_list, not emitted as entries
            if self._scope is not None:
                words = self._extract_words(construct.inner.text)
                if words:
                    self._scope.add_word_list(words)
            return

        if isinstance(construct.inner, Literal):
            self._emit_entry(construct.inner.id, None, "UNSIGNED")
            return

        primaries = construct.inner

        if construct.chain_op is None:
            self._process_primaries(primaries)
        else:
            self._process_chain(primaries, construct.chain_op, construct.chain_right)

    def _process_primaries(self, primaries: list[PrimaryConstruct]) -> None:
        for pc in primaries:
            self._emit_primary(pc)

    def _emit_mcs(self, sig: str) -> int | None:
        """Emit MCS entries for multi-character signatures.

        Each constituent character is resolved via _resolve_char before
        emitting unsigned entries. The canonization entry keeps the original
        composed sig string but uses resolved words for its nodes list.

        Returns the index of the CANONIZE entry in self.entries, or None
        if no MCS was emitted (single char or skip_mcs).
        """
        if self._skip_mcs:
            return None

        if len(sig) <= 1:
            return None

        chars = [self._resolve_char(c) for c in sig]
        for resolved_char in chars:
            self._emit_entry(resolved_char, None, "UNSIGNED")
        self._emit_entry(sig, chars, "CANONIZE")
        return len(self.entries) - 1

    def _emit_primary(self, pc: PrimaryConstruct) -> None:
        sig = pc.sig.id

        # Inline-first resolution: check sig-side inline comment
        sig_resolved = self._resolve_inline_or_scope(sig, pc.inline_comment)

        self._emit_mcs(sig)

        if pc.op is None:
            self._emit_entry(sig_resolved, None, "UNSIGNED")
            return

        node = pc.node
        node_str = self._node_to_string(node)
        # Resolve node via inline comment or scope
        node_resolved = self._resolve_inline_or_scope_node(
            node_str, pc.node_inline_comment
        )
        if node_str.isupper() and node_str.isalpha():
            self._emit_mcs(node_str)

        if pc.op == TokenType.COUNTERSIGN:
            self._emit_entry(sig_resolved, node_resolved, "COUNTERSIGN")
            if self._is_signature(node):
                self._emit_entry(node_resolved, sig_resolved, "COUNTERSIGN")

        elif pc.op == TokenType.UNDERSIGN:
            if sig_resolved == node_resolved:
                self._emit_entry(sig_resolved, None, "IDENTITY")
            else:
                self._emit_entry(node_resolved, sig_resolved, "UNDERSIGN")

        elif pc.op == TokenType.CONNOTATE:
            self._emit_entry(sig_resolved, node_resolved, "CONNOTATE")

    def _process_chain(
        self,
        left_primaries: list[PrimaryConstruct],
        chain_op: TokenType,
        right: Construct | None
    ) -> None:
        for pc in left_primaries:
            if pc.op is not None:
                self._emit_primary(pc)

        if right is None:
            return

        right_items = self._flatten_to_items(right)

        if chain_op == TokenType.CANONIZE:
            last = left_primaries[-1]
            owner = self._get_owner(last)
            mcs_idx: int | None = None
            if last.node is None or isinstance(last.node, Signature):
                mcs_idx = self._emit_mcs(owner)
            if right_items:
                item_ids = [self._item_id(item) for item in right_items]
                self._emit_entry(owner, item_ids, "CANONIZE")

        # Save and set parent kline tracking for Rule 4 override
        saved_chars = self._parent_kline_chars
        saved_idx = self._parent_kline_canonize_idx

        if chain_op == TokenType.CANONIZE and self._scope is not None:
            self._parent_kline_chars = owner
            self._parent_kline_canonize_idx = mcs_idx
            self._scope.push_scope()

        self._emit_construct(right)

        if chain_op == TokenType.CANONIZE and self._scope is not None:
            self._scope.pop_scope()
            self._parent_kline_chars = saved_chars
            self._parent_kline_canonize_idx = saved_idx

    def _flatten_to_items(self, construct: Construct) -> list[ConstructItem]:
        if isinstance(construct.inner, Block):
            items: list[ConstructItem] = []
            for c in construct.inner.constructs:
                items.extend(self._flatten_to_items(c))
            return items
        if isinstance(construct.inner, Comment):
            return []  # Comments feed the scope's word list via add_word_list, not emitted
        if isinstance(construct.inner, Literal):
            return [construct.inner]
        return construct.inner

    def _item_id(self, item: ConstructItem) -> str:
        if isinstance(item, PrimaryConstruct):
            return item.sig.id
        elif isinstance(item, Literal):
            return item.id
        return str(item)

    def _get_owner(self, pc: PrimaryConstruct) -> str:
        if pc.node is not None:
            return self._node_to_string(pc.node)
        return pc.sig.id

    def _node_to_string(self, node: Node | None) -> str:
        if isinstance(node, Signature):
            return node.id
        elif isinstance(node, Literal):
            return node.id
        return str(node)

    def _is_signature(self, node: Node | None) -> bool:
        if isinstance(node, Signature):
            return True
        if isinstance(node, Literal):
            return False
        return False

    # ------------------------------------------------------------------
    # Inline resolution helpers
    # ------------------------------------------------------------------

    def _resolve_inline_or_scope(self, sig: str, inline_comment: Comment | None) -> str:
        """Resolve a sig string: inline comment first, then scope fallback.

        For single-char sigs: check inline comment, then scope.resolve().
        For multi-char sigs: return as-is (MCS decomposition handles chars).

        When an inline binding fires for a single character, the method also
        triggers Rule 4 override patching of the parent kline's MCS CANONIZE
        entry if applicable.
        """
        if len(sig) == 1:
            if inline_comment is not None:
                word = self._extract_inline_word(sig, inline_comment)
                self._patch_parent_canonize(sig, word)
                return word
            resolved = self._resolve_char(sig)
            return resolved
        return sig

    def _resolve_inline_or_scope_node(self, node_str: str, node_inline_comment: Comment | None) -> str:
        """Resolve a node string: inline comment first, then scope fallback.

        For single-char nodes: check inline comment, then scope.resolve().
        For multi-char nodes: return as-is (MCS decomposition handles chars).

        When an inline binding fires, triggers Rule 4 override patching.
        """
        if len(node_str) == 1:
            if node_inline_comment is not None:
                word = self._extract_inline_word(node_str, node_inline_comment)
                self._patch_parent_canonize(node_str, word)
                return word
            resolved = self._resolve_char(node_str)
            return resolved
        return node_str

    def _patch_parent_canonize(self, char: str, word: str) -> None:
        """Rule 4 — inline override: patch parent kline MCS CANONIZE entry.

        When an inline binding fires for char C with resolved word W inside
        a subscript, retroactively patch the matching character in the already-
        emitted MCS CANONIZE entry for the parent kline.

        Example:
            Source: SVO => Block([S(ubject) = M])
            Before: CANONIZE("SVO", ["S","V","O"]) at parent_kline_canonize_idx
            After:  CANONIZE("SVO", ["Subject","V","O"]) — S patched at index 0

        Only patches the immediate parent — no propagation beyond one level.
        If char is not found in parent kline chars, this is a safe no-op (NB-33).
        """
        if self._parent_kline_chars is None or self._parent_kline_canonize_idx is None:
            return
        idx = self._parent_kline_chars.find(char)
        if idx < 0:
            return  # NB-33: safe no-op
        entry = self.entries[self._parent_kline_canonize_idx]
        if entry.op != "CANONIZE":
            return
        nodes = entry.nodes
        if isinstance(nodes, list) and idx < len(nodes):
            # Patch in-place via entry._replace — NamedTuple is immutable
            new_nodes = list(nodes)
            new_nodes[idx] = word
            self.entries[self._parent_kline_canonize_idx] = entry._replace(nodes=new_nodes)
        elif isinstance(nodes, str) and nodes == char:
            # Singleton case
            self.entries[self._parent_kline_canonize_idx] = entry._replace(nodes=word)

    # ------------------------------------------------------------------
    # Word extraction helpers
    # ------------------------------------------------------------------

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

    def _emit_entry(self, sig: str, nodes: str | None | list[str], op: str) -> None:
        """Emit a symbolic entry with dedup.

        Singleton rule: if nodes is a list with length 1, unwrap to single value.

        All NLP resolution happens upstream in _emit_primary and _emit_mcs.
        This method receives already-resolved sig/node strings.
        """
        if isinstance(nodes, list) and len(nodes) == 1:
            nodes = nodes[0]

        # Dedup on symbolic values
        key = (sig, None if nodes is None else nodes if isinstance(nodes, str) else tuple(nodes))
        if key in self._seen:
            return
        self._seen.add(key)

        self.entries.append(SymbolicEntry(sig, nodes, op))
