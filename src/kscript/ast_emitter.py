"""AST Emitter — walks KScript AST and yields symbolic entries.

This module is responsible for AST traversal and determining the correct
operator semantics for each construct. It yields symbolic tuples
(sig_str, nodes_strs, op) that a TokenEncoder then converts to token IDs.

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
    """

    def __init__(self, dev: bool = False, skip_mcs: bool = False):
        self.entries: list[SymbolicEntry] = []
        self.dev = dev
        self._skip_mcs = skip_mcs
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

    def _emit_mcs(self, sig: str) -> bool:
        """Emit MCS entries for multi-character signatures."""
        if self._skip_mcs:
            return False

        if len(sig) <= 1:
            return False

        chars = list(sig)
        for char in chars:
            self._emit_entry(char, None, "UNSIGNED")
        self._emit_entry(sig, chars, "CANONIZE")
        return True

    def _emit_primary(self, pc: PrimaryConstruct) -> None:
        sig = pc.sig.id
        self._emit_mcs(sig)

        if pc.op is None:
            self._emit_entry(sig, None, "UNSIGNED")
            return

        node = pc.node
        node_str = self._node_to_string(node)
        if node_str.isupper() and node_str.isalpha():
            self._emit_mcs(node_str)

        if pc.op == TokenType.COUNTERSIGN:
            self._emit_entry(sig, node_str, "COUNTERSIGN")
            if self._is_signature(node):
                self._emit_entry(node_str, sig, "COUNTERSIGN")

        elif pc.op == TokenType.UNDERSIGN:
            if sig == node_str:
                self._emit_entry(sig, None, "IDENTITY")
            else:
                self._emit_entry(node_str, sig, "UNDERSIGN")

        elif pc.op == TokenType.CONNOTATE:
            self._emit_entry(sig, node_str, "CONNOTATE")

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
            if last.node is None or isinstance(last.node, Signature):
                self._emit_mcs(owner)
            if right_items:
                item_ids = [self._item_id(item) for item in right_items]
                self._emit_entry(owner, item_ids, "CANONIZE")
            self._emit_construct(right)

    def _flatten_to_items(self, construct: Construct) -> list[ConstructItem]:
        if isinstance(construct.inner, Block):
            items: list[ConstructItem] = []
            for c in construct.inner.constructs:
                items.extend(self._flatten_to_items(c))
            return items
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

    def _emit_entry(self, sig: str, nodes: str | None | list[str], op: str) -> None:
        """Emit a symbolic entry with dedup.

        Singleton rule: if nodes is a list with length 1, unwrap to single value.
        """
        if isinstance(nodes, list) and len(nodes) == 1:
            nodes = nodes[0]

        # Dedup on symbolic values
        key = (sig, None if nodes is None else nodes if isinstance(nodes, str) else tuple(nodes))
        if key in self._seen:
            return
        self._seen.add(key)

        self.entries.append(SymbolicEntry(sig, nodes, op))
