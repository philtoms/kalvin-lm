"""KScript compiler with BWD construct binding semantics.

Op mappings (using abbreviated property chains):
  COUNTERSIGN   -> {sig: node}, {node: sig}
  UNDERSIGN     -> {sig: node}
  CANONIZE_FWD  -> {p[-1].(node or sig): [p.sig for p in r.primaries]}
  CONNOTATE_FWD -> {sig: [node]}
  CANONIZE_BWD  -> {r[0].sig: [p.sig for p in left_primaries]}
  CONNOTATE_BWD -> {r[0].sig: [left_primaries[-1].sig]}

Where:
  p = primary construct (left side)
  r = right (chain_right's primaries)

Significance encoding (bitwise OR on signature position):
  COUNTERSIGN   -> sig | S1
  UNDERSIGN     -> sig | S1
  CANONIZE_*    -> sig | S2
  CONNOTATE_*   -> sig | S3
  UNSIGNED      -> sig | S4 (S4=0, no bits)
"""

from __future__ import annotations

from typing import TypeAlias

from kalvin.abstract import KLine, KNodes, KSig, KSignificance
from kalvin.mod_tokenizer import Mod32Tokenizer, ModTokenizer, PACKED_BIT
from kalvin.significance import Int32Significance

from .ast import (
    Block,
    Construct,
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
        significance: KSignificance | None = None,
        dbg_text: str = ""
    ) -> "CompiledEntry":
        """Encode string signature/nodes to token IDs.

        Signatures (uppercase strings) are packed (PACKED_BIT set).
        Literals are unpacked: each char encoded as (ord(char) << 1).
        """
        sig_obj = significance or Int32Significance()
        sig_id = tokenizer.encode(sig, pack=True)[0]
        sig_id = cls._add_significance(sig_id, sig_level, sig_obj)

        if nodes is None:
            return cls(signature=sig_id, nodes=None, dbg_text=dbg_text)
        elif isinstance(nodes, str):
            if nodes.isupper() and nodes.isalpha():
                node_id = tokenizer.encode(nodes, pack=True)[0]
                return cls(signature=sig_id, nodes=node_id, dbg_text=dbg_text)
            else:
                node_ids = [ord(c) << 1 for c in nodes]
                return cls(signature=sig_id, nodes=node_ids, dbg_text=dbg_text)
        else:
            all_node_ids: list[int] = []
            for n in nodes:
                if n.isupper() and n.isalpha():
                    all_node_ids.append(tokenizer.encode(n, pack=True)[0])
                else:
                    all_node_ids.extend(ord(c) << 1 for c in n)
            return cls(signature=sig_id, nodes=all_node_ids, dbg_text=dbg_text)

    @staticmethod
    def _add_significance(token_id: int, sig_level: str, sig: KSignificance) -> int:
        """Add significance bits to token ID based on sig level."""
        if sig_level == "S1":
            return token_id | sig.S1
        elif sig_level == "S2":
            return token_id | sig.S2
        elif sig_level == "S3":
            return token_id | sig.S3
        else:  # S4 or unknown
            return token_id

    def decode(self, tokenizer: ModTokenizer) -> tuple[str, str | None | list[str]]:
        """Decode token IDs back to strings."""
        # Mask off significance bits to get raw token
        token_mask = ~(0xFFFFFFFF << 32)  # Lower 32 bits for token
        sig_token = self.signature & token_mask
        sig = tokenizer.decode([sig_token], pack=None)

        if self.nodes is None:
            return sig, None
        elif isinstance(self.nodes, int):
            # Check if packed (signature) or unpacked (literal char)
            if self.nodes & PACKED_BIT:
                decoded = tokenizer.decode([self.nodes], pack=None)
                return sig, decoded
            else:
                # Literal char: decode as chr(n >> 1)
                return sig, chr(self.nodes >> 1)
        else:
            all_unpacked = all((n & PACKED_BIT) == 0 for n in self.nodes)

            if all_unpacked:
                decoded = "".join(chr(n >> 1) for n in self.nodes)
                return sig, decoded
            else:
                decoded_nodes: list[str] = []
                literal_chars: list[int] = []

                for n in self.nodes:
                    if (n & PACKED_BIT) == 0:
                        literal_chars.append(n)
                    else:
                        if literal_chars:
                            decoded_nodes.append("".join(chr(c >> 1) for c in literal_chars))
                            literal_chars = []
                        decoded_nodes.append(tokenizer.decode([n], pack=None))

                if literal_chars:
                    decoded_nodes.append("".join(chr(c >> 1) for c in literal_chars))

                return sig, decoded_nodes


class Compiler:
    """Compiles KScript AST to CompiledEntry list.

    Traverses the AST and emits CompiledEntry based on operator type.
    """

    def __init__(self, tokenizer: ModTokenizer | None = None, dev: bool = False):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev
        self._sig = Int32Significance()
        self._sig_levels = {
            "COUNTERSIGN": self._sig.S1,
            "CANONIZE_FWD": self._sig.S2,
            "CANONIZE_BWD": self._sig.S2,
            "CONNOTATE_FWD": self._sig.S3,
            "CONNOTATE_BWD": self._sig.S3,
            "UNDERSIGN": self._sig.S1,
            "UNSIGNED": self._sig.S4,
            "MCS": self._sig.S2,       # MCS uses S2 like canonize
            "MCS_CHAR": self._sig.S1,  # Component chars are S1 (identity-like)
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
        """Compile a construct, handling blocks, chains, and primaries."""
        if isinstance(construct.inner, Block):
            # Block: recurse into nested constructs
            for c in construct.inner.constructs:
                self._compile_construct(c)
            return

        # list[PrimaryConstruct]
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
          Identity -> {char: char} for char in sig
          MCS -> {sig: [char for char in sig]} (second)

        Returns True if MCS was emitted, False for single-char sigs.
        """
        if len(sig) <= 1:
            return False

        chars = list(sig)

        # Component identities: {char: None} for each (emitted first)
        for char in chars:
            self._emit(char, char, "MCS_CHAR")

        # MCS canonization: {sig: [A, B, C, ...]} (emitted second)
        self._emit(sig, chars, "MCS")

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
            # {sig: node}
            if sig == node_str:
                self._emit(sig, None, "IDENTITY")
            else:
                self._emit(sig, node_str, "UNDERSIGN")

        elif pc.op == TokenType.CONNOTATE_FWD:
            # {sig: [node]}
            self._emit(sig, [node_str], "CONNOTATE_FWD")

    def _process_chain(
        self,
        left_primaries: list[PrimaryConstruct],
        chain_op: TokenType,
        right: Construct | None
    ) -> None:
        """Process a chain construct (=>, <=, <)."""
        # Process all left primaries (emits MCS + inline ops)
        for pc in left_primaries:
            if pc.op is not None:
                self._emit_primary(pc)

        if right is None:
            return

        # Get right primaries (flatten if right is also a chain)
        right_primaries = self._flatten_to_primaries(right)

        if chain_op == TokenType.CANONIZE_FWD:
            # CANONIZE_FWD: {p[-1].(node or sig): [p.sig for p in r.primaries]}
            last = left_primaries[-1]
            owner = self._get_owner(last)  # node if present, else sig
            # Emit MCS for owner if it's a sig
            if last.node is None or isinstance(last.node, Signature):
                self._emit_mcs(owner)
            nodes = [pc.sig.id for pc in right_primaries]
            self._emit(owner, nodes, "CANONIZE_FWD")

            # Recurse into right
            self._compile_construct(right)

        elif chain_op == TokenType.CANONIZE_BWD:
            # CANONIZE_BWD: {r[0].sig: [p.sig for p in left_primaries]}
            if right_primaries:
                owner = right_primaries[0].sig.id
                nodes = [pc.sig.id for pc in left_primaries]
                self._emit(owner, nodes, "CANONIZE_BWD")

            # Recurse into right
            self._compile_construct(right)

        elif chain_op == TokenType.CONNOTATE_BWD:
            # CONNOTATE_BWD: {r[0].sig: [left_primaries[-1].sig]}
            if right_primaries:
                owner = right_primaries[0].sig.id
                last_left = left_primaries[-1].sig.id
                self._emit(owner, [last_left], "CONNOTATE_BWD")

            # Recurse into right
            self._compile_construct(right)

    def _flatten_to_primaries(self, construct: Construct) -> list[PrimaryConstruct]:
        """Extract all primaries from a construct, handling nested chains."""
        if isinstance(construct.inner, Block):
            # Block - collect primaries from all nested constructs
            primaries = []
            for c in construct.inner.constructs:
                primaries.extend(self._flatten_to_primaries(c))
            return primaries
        return construct.inner

    def _get_owner(self, pc: PrimaryConstruct) -> str:
        """Get owner for CANONIZE_FWD: node if present, else sig."""
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
        """Emit an entry with significance encoding."""
        # Encode signature with significance bits
        sig_level = self._sig_levels.get(op, self._sig.S4)
        sig_id = self._encode_sig(sig) | sig_level

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
        return self.tokenizer.encode(sig, pack=True)[0]

    def _encode_node(self, node: str) -> int:
        """Encode a single node string to token ID (no significance bits).

        Signatures get packed, literals get character encoding.
        """
        if node.isupper() and node.isalpha():
            return self.tokenizer.encode(node, pack=True)[0]
        else:
            # Literal: encode each char as (ord(char) << 1)
            if len(node) == 1:
                return ord(node) << 1
            # Multi-char literal: return first char only for single node
            # (caller should use _encode_nodes for list handling)
            return ord(node[0]) << 1

    def _encode_nodes(self, nodes: str | None | list[str]) -> None | int | list[int]:
        """Encode nodes to token IDs (no significance bits)."""
        if nodes is None:
            return None
        elif isinstance(nodes, str):
            # Single string node
            if nodes.isupper() and nodes.isalpha():
                # Signature: single packed token
                return self.tokenizer.encode(nodes, pack=True)[0]
            elif len(nodes) == 1:
                # Single char literal
                return ord(nodes) << 1
            else:
                # Multi-char literal: list of encoded chars
                return [ord(c) << 1 for c in nodes]
        else:
            result: list[int] = []
            for n in nodes:
                if n.isupper() and n.isalpha():
                    result.append(self.tokenizer.encode(n, pack=True)[0])
                else:
                    # Literal: encode each character
                    result.extend(ord(c) << 1 for c in n)
            return result

    def _format_dbg(self, sig: str, nodes: str | None | list[str], op: str) -> str:
        """Format debug representation with significance level."""
        sig_val = self._sig_levels.get(op, self._sig.S4)
        if sig_val == self._sig.S1:
            level = "S1"
        elif sig_val == self._sig.S2:
            level = "S2"
        elif sig_val == self._sig.S3:
            level = "S3"
        else:
            level = "S4"

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
