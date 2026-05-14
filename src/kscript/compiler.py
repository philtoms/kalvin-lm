"""KScript compiler with construct binding semantics.

Op mappings (using abbreviated property chains):
  COUNTERSIGN  -> {sig: node}, {node: sig}
  CANONIZE     -> {p[-1].(node or sig): p.sig for p in r.primaries}
  CONNOTATE    -> {sig: node}
  UNDERSIGN    -> {node: sig}
  UNSIGNED     -> sig | S4 (S4=0, no bits)

Where:
  p = primary construct (left side)
  r = right (chain_right's primaries)

Singleton rule: nodes lists with length 1 are unwrapped to single values.

Significance encoding (bitwise OR on signature position):
  COUNTERSIGN  -> sig | S1
  UNDERSIGN    -> sig | S1
  CANONIZE     -> sig | S2
  CONNOTATE    -> sig | S3
  UNSIGNED     -> sig | S4 (S4=0, no bits)
"""

from __future__ import annotations

from typing import TypeAlias

from kalvin.kline import KLine, KNodes, KSig
from kalvin.mod_tokenizer import Mod32Tokenizer, ModTokenizer
from kalvin.model import D_MAX
from kalvin.signature import is_literal_node

from .ast import (
    Block,
    Construct,
    ConstructItem,
    KScriptFile,
    Literal,
    Node,
    PrimaryConstruct,
    Script,
    Signature,
)
from .parser import Parser
from .token import TokenType

class CompiledEntry(KLine):
    """A compiled KLine entry with decode support.

    Nodes semantics:
    - None: unsigned entry (sig exists with no children)
    - int: single token ID link (countersign, undersign)
    - list[int]: nodes list (connotate, canonize)
    """

    def __init__(self, signature: KSig, nodes: KNodes, dbg_text: str = ""):
        super().__init__(signature=signature, nodes=nodes, dbg_text=dbg_text)

    @classmethod
    def encode(
        cls,
        sig: str,
        nodes: str | None | list[str],
        tokenizer: ModTokenizer,
        *,
        sig_level: str = "S4",
        significance: object | None = None,
        dbg_text: str = ""
    ) -> "CompiledEntry":
        """Encode string signature/nodes to token IDs.

        Signatures (uppercase strings) are packed automatically.
        Literals are encoded as literal nodes automatically.
        """
        # significance param kept for API compat; encoding is pure tokenizer now
        sig_id = tokenizer.encode(sig)[0]
        if nodes is None:
            return cls(signature=sig_id, nodes=None, dbg_text=dbg_text)
        elif isinstance(nodes, str):
            if nodes.isupper() and nodes.isalpha():
                node_id = tokenizer.encode(nodes)[0]
                return cls(signature=sig_id, nodes=node_id, dbg_text=dbg_text)
            else:
                node_ids = tokenizer.encode(nodes)
                return cls(signature=sig_id, nodes=node_ids, dbg_text=dbg_text)
        else:
            all_node_ids: list[int] = []
            for n in nodes:
                if n.isupper() and n.isalpha():
                    all_node_ids.append(tokenizer.encode(n)[0])
                else:
                    all_node_ids.extend(tokenizer.encode(n))
            return cls(signature=sig_id, nodes=all_node_ids, dbg_text=dbg_text)

    def decode(self, tokenizer: ModTokenizer) -> tuple[str, str | None | list[str]]:
        """Decode token IDs back to strings."""
        # Use signature token directly (significance bits not yet implemented)
        sig = tokenizer.decode([self.signature])

        if self.nodes is None:
            return sig, None
        elif isinstance(self.nodes, int):
            # Check if literal or packed signature
            if is_literal_node(self.nodes):
                return sig, tokenizer.decode([self.nodes])
            else:
                decoded = tokenizer.decode([self.nodes])
                return sig, decoded
        else:
            all_literals = all(is_literal_node(n) for n in self.nodes)

            if all_literals:
                decoded = tokenizer.decode(list(self.nodes))
                return sig, decoded
            else:
                decoded_nodes: list[str] = []
                literal_chars: list[int] = []

                for n in self.nodes:
                    if is_literal_node(n):
                        literal_chars.append(n)
                    else:
                        if literal_chars:
                            decoded_nodes.append(tokenizer.decode(literal_chars))
                            literal_chars = []
                        decoded_nodes.append(tokenizer.decode([n]))

                if literal_chars:
                    decoded_nodes.append(tokenizer.decode(literal_chars))

                return sig, decoded_nodes


class Compiler:
    """Compiles KScript AST to CompiledEntry list.

    Traverses the AST and emits CompiledEntry based on operator type.
    """

    def __init__(self, tokenizer: ModTokenizer | None = None, dev: bool = False):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev
        self._sig_levels = {
            "COUNTERSIGN": "S1",
            "CANONIZE": "S2",
            "CONNOTATE": "S3",
            "UNDERSIGN": "S1",
            "UNSIGNED": "S4",
            "IDENTITY": "S1",
        }
        self._seen: set[tuple[int, None | int | tuple[int, ...]]] = set()

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile to entries."""
        for script in file.scripts:
            self._compile_script(script)
        return self.entries

    def _compile_script(self, script: Script) -> None:
        """Compile a script's constructs."""
        for construct in script.constructs:
            self._compile_construct(construct)

    def _compile_construct(self, construct: Construct) -> None:
        """Compile a construct, handling blocks, literals, chains, and primaries."""
        if isinstance(construct.inner, Block):
            # Block: recurse into nested constructs
            for c in construct.inner.constructs:
                self._compile_construct(c)
            return

        if isinstance(construct.inner, Literal):
            # Bare literal: emit unsigned identity
            self._emit(construct.inner.id, None, "UNSIGNED")
            return

        # list[PrimaryConstruct] with optional chain
        primaries = construct.inner

        if construct.chain_op is None:
            # No chain - process primaries with inline ops
            self._process_primaries(primaries)
        else:
            # Chain op present
            self._process_chain(primaries, construct.chain_op, construct.chain_right)

    def _process_primaries(self, primaries: list[PrimaryConstruct]) -> None:
        """Process primary constructs with inline operators."""
        for pc in primaries:
            self._emit_primary(pc)

    def _emit_mcs(self, sig: str) -> bool:
        """Emit MCS entries for multi-character signatures.

        Emits:
          Unsigned -> {char: None} for each char in sig
          MCS CANONIZE -> {sig: [chars]} (list preserves order & duplicates)

        Returns True if MCS was emitted, False for single-char sigs.
        """
        if len(sig) <= 1:
            return False

        chars = list(sig)

        # Component identities: {char: None} for each (emitted first)
        for char in chars:
            self._emit(char, None, "UNSIGNED")

        # MCS canonization: {sig: [A, B, C, ...]} (list preserves order & duplicates)
        self._emit(sig, chars, "CANONIZE")

        return True

    def _emit_primary(self, pc: PrimaryConstruct) -> None:
        """Emit entries for a primary construct based on its inline op.

        Note: MCS emission for pc.sig should be done by caller.
        """
        sig = pc.sig.id
        self._emit_mcs(sig)

        if pc.op is None:
            # Unsigned: bare signature
            self._emit(sig, None, "UNSIGNED")
            return

        node = pc.node
        node_str = self._node_to_string(node)
        if node_str.isupper() and node_str.isalpha():
            self._emit_mcs(node_str)

        if pc.op == TokenType.COUNTERSIGN:
            # {sig: node}, {node: sig}
            self._emit(sig, node_str, "COUNTERSIGN")
            if self._is_signature(node):
                self._emit(node_str, sig, "COUNTERSIGN")

        elif pc.op == TokenType.UNDERSIGN:
            # {node: sig} — value (right side) becomes signature, query (left side) becomes node
            if sig == node_str:
                self._emit(sig, None, "IDENTITY")
            else:
                self._emit(node_str, sig, "UNDERSIGN")

        elif pc.op == TokenType.CONNOTATE:
            # {sig: node}
            self._emit(sig, node_str, "CONNOTATE")

    def _process_chain(
        self,
        left_primaries: list[PrimaryConstruct],
        chain_op: TokenType,
        right: Construct | None
    ) -> None:
        """Process a chain construct (=>).

        Left side is always primary_constructs (only they can have chain ops).
        Right side can be any construct: block (may contain mixed literals + primaries),
        a single literal, or primary_constructs.
        """
        # Process all left primaries with inline ops
        for pc in left_primaries:
            if pc.op is not None:
                self._emit_primary(pc)

        if right is None:
            return

        # Get right items (flatten blocks, handle literal/primaries)
        right_items = self._flatten_to_items(right)

        if chain_op == TokenType.CANONIZE:
            # CANONIZE: {owner: [all items]} — single entry with all right-hand items
            last = left_primaries[-1]
            owner = self._get_owner(last)  # node if present, else sig
            # Emit MCS for owner if it's a sig
            if last.node is None or isinstance(last.node, Signature):
                self._emit_mcs(owner)
            # Emit single CANONIZE entry with all right items as a list
            if right_items:
                item_ids = [self._item_id(item) for item in right_items]
                self._emit(owner, item_ids, "CANONIZE")

            # Recurse into right
            self._compile_construct(right)

    def _flatten_to_items(self, construct: Construct) -> list[ConstructItem]:
        """Extract all items from a construct.

        Handles:
        - Block: recursively flattens all nested constructs (may mix literals + primaries)
        - Literal: returns [literal] as a single item
        - list[PrimaryConstruct]: returns the primaries directly
        """
        if isinstance(construct.inner, Block):
            items: list[ConstructItem] = []
            for c in construct.inner.constructs:
                items.extend(self._flatten_to_items(c))
            return items
        if isinstance(construct.inner, Literal):
            return [construct.inner]
        return construct.inner

    def _flatten_to_primaries(self, construct: Construct) -> list[PrimaryConstruct]:
        """Extract only PrimaryConstruct items from a construct."""
        items = self._flatten_to_items(construct)
        return [item for item in items if isinstance(item, PrimaryConstruct)]

    def _item_id(self, item: ConstructItem) -> str:
        """Get the string ID from a construct item.

        PrimaryConstruct -> sig.id
        Literal -> literal id
        """
        if isinstance(item, PrimaryConstruct):
            return item.sig.id
        elif isinstance(item, Literal):
            return item.id
        return str(item)

    def _get_owner(self, pc: PrimaryConstruct) -> str:
        """Get owner for CANONIZE: node if present, else sig."""
        if pc.node is not None:
            return self._node_to_string(pc.node)
        return pc.sig.id

    def _node_to_string(self, node: Node | None) -> str:
        """Convert a Node to its string representation."""
        if isinstance(node, Signature):
            return node.id
        elif isinstance(node, Literal):
            return node.id
        return str(node)

    def _is_signature(self, node: Node| None) -> bool:
        """Check if a node is a signature (uppercase alpha)."""
        if isinstance(node, Signature):
            return True
        if isinstance(node, Literal):
            return False
        return False

    def _emit(self, sig: str, nodes: str | None | list[str], op: str) -> None:
        """Emit an entry with significance encoding.

        Singleton rule: if nodes is a list with length 1, unwrap to single value.
        """
        # Singleton check: unwrap single-element lists
        if isinstance(nodes, list) and len(nodes) == 1:
            nodes = nodes[0]
        # Encode signature as significant or literal
        sig_id = self._encode_node(sig)

        # Encode nodes
        encoded_nodes = self._encode_nodes(nodes)

        # Dedup: check if full entry already exists
        key = (sig_id, None if encoded_nodes is None else
               encoded_nodes if isinstance(encoded_nodes, int) else tuple(encoded_nodes))
        if key in self._seen:
            return
        self._seen.add(key)

        dbg = self._format_dbg(sig, nodes, op) if self.dev else ""
        self.entries.append(CompiledEntry(signature=sig_id, nodes=encoded_nodes, dbg_text=dbg))

    def _encode_sig(self, sig: str) -> int:
        """Encode a signature string to token ID (no significance bits)."""
        return self.tokenizer.encode(sig)[0]

    def _encode_node(self, node: str) -> int:
        """Encode a single node string to token ID (no significance bits).

        Signatures get packed, literals get character encoding.
        """
        if node.isupper() and node.isalpha():
            return self._encode_sig(node)
        else:
            # Literal: encode via tokenizer
            return self.tokenizer.encode(node[0] if len(node) > 1 else node)[0]

    def _encode_nodes(self, nodes: str | None | list[str]) -> None | int | list[int]:
        """Encode nodes to token IDs (no significance bits)."""
        if nodes is None:
            return None
        elif isinstance(nodes, str):
            # Single string node
            if nodes.isupper() and nodes.isalpha():
                # Signature: single packed token
                return self.tokenizer.encode(nodes)[0]
            elif len(nodes) == 1:
                # Single char literal
                return self.tokenizer.encode(nodes)[0]
            else:
                # Multi-char literal: list of encoded chars
                return self.tokenizer.encode(nodes)
        else:
            result: list[int] = []
            for n in nodes:
                if n.isupper() and n.isalpha():
                    result.append(self.tokenizer.encode(n)[0])
                else:
                    # Literal: encode each character
                    result.extend(self.tokenizer.encode(n))
            return result

    def _format_dbg(self, sig: str, nodes: str | None | list[str], op: str) -> str:
        """Format debug representation with significance level."""
        level = self._sig_levels.get(op, "S4")

        if nodes is None:
            return f"[{level}] {sig}: None"
        elif isinstance(nodes, str):
            return f"[{level}] {sig}: {nodes}"
        else:
            return f"[{level}] {sig}: {nodes}"


def compile_source(source: str, tokenizer: ModTokenizer | None = None, dev: bool = False) -> list[CompiledEntry]:
    """Compile KScript source string to entries."""
    from .lexer import Lexer

    tokens = Lexer(source).tokenize()
    kscript_file = Parser(tokens).parse()
    return Compiler(tokenizer, dev=dev).compile(kscript_file)
