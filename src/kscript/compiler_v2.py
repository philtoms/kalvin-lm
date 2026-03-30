"""Independent KScript compiler with op-mapped KLine emission.

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
"""

from dataclasses import dataclass
from typing import TypeAlias

from kalvin.abstract import KLine, KNodes, KSig
from kalvin.mod_tokenizer import Mod32Tokenizer, ModTokenizer, PACKED_BIT
from kalvin.significance import Int32Significance

from .token import TokenType
from .parser import (
    Block,
    Construct,
    KScriptFile,
    Literal,
    Node,
    Parser,
    PrimaryConstruct,
    Script,
    Signature,
)

_sig = Int32Significance()


@dataclass
class OpKLine:
    """A single op-mapped KLine with source context.

    Attributes:
        sig: The signature (owner) string
        nodes: None, str, or list of strings
        op: The operator that produced this entry
        dbg: Debug text representation
    """
    sig: str
    nodes: str | None | list[str]
    op: str
    dbg: str = ""


class CompilerV2:
    """Independent compiler emitting op-mapped KLines.

    Traverses the AST and emits OpKLines based on operator type.
    No significance encoding - just structural mapping.
    """

    def __init__(self, tokenizer: ModTokenizer | None = None, dev: bool = False):
        self.entries: list[OpKLine] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev
        self._emitted_sigs: set[str] = set()  # Track emitted MCS to avoid duplicates

    def compile(self, file: KScriptFile) -> list[OpKLine]:
        """Compile a KScriptFile to op-mapped entries."""
        for script in file.scripts:
            self._compile_script(script)
        return self.entries

    def _compile_script(self, script: Script) -> None:
        """Compile a script's constructs."""
        for construct in script.constructs:
            self._compile_construct(construct)

    def _emit_mcs(self, sig: str) -> bool:
        """Emit MCS entries for multi-character signatures.

        Emits:
          {sig: [char for char in sig]}
          {char: None} for each char

        Returns True if MCS was emitted, False for single-char sigs.
        """
        if len(sig) <= 1:
            return False
        if sig in self._emitted_sigs:
            return False

        self._emitted_sigs.add(sig)

        # MCS canonization: {sig: [A, B, C, ...]}
        chars = list(sig)
        self._emit(sig, chars, "MCS")

        # Component identities: {char: None} for each
        for char in chars:
            if char not in self._emitted_sigs:
                self._emit(char, None, "MCS_CHAR")
                self._emitted_sigs.add(char)

        return True

    def _compile_construct(self, construct: Construct) -> None:
        """Compile a construct, handling blocks, chains, and primaries."""
        if isinstance(construct.inner, Block):
            # Block: recurse into nested constructs
            for c in construct.inner.constructs:
                self._compile_construct(c)
            return

        # List[PrimaryConstruct]
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
            self._emit_mcs(pc.sig.id)
            self._emit_primary(pc)

    def _emit_primary(self, pc: PrimaryConstruct) -> None:
        """Emit entries for a primary construct based on its inline op.

        Note: MCS emission for pc.sig should be done by caller.
        """
        sig = pc.sig.id

        if pc.op is None:
            # Identity: bare signature
            self._emit(sig, None, "IDENTITY")
            return

        node = pc.node
        node_str = self._node_to_string(node)

        if pc.op == TokenType.COUNTERSIGN:
            # {sig: node}, {node: sig}
            self._emit(sig, node_str, "COUNTERSIGN")
            if self._is_signature(node):
                self._emit_mcs(node_str)
                self._emit(node_str, sig, "COUNTERSIGN")

        elif pc.op == TokenType.UNDERSIGN:
            # {sig: node}
            self._emit(sig, node_str, "UNDERSIGN")

        elif pc.op == TokenType.CONNOTATE_FWD:
            # {sig: [node]}
            self._emit(sig, [node_str], "CONNOTATE_FWD")

    def _process_chain(
        self,
        left_primaries: list[PrimaryConstruct],
        chain_op: TokenType,
        right: Construct
    ) -> None:
        """Process a chain construct (=>, <=, <)."""
        # Process all left primaries (emits MCS + inline ops)
        for pc in left_primaries:
            self._emit_mcs(pc.sig.id)
            if pc.op is not None:
                self._emit_primary(pc)

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
                self._emit_mcs(owner)
                nodes = [pc.sig.id for pc in left_primaries]
                self._emit(owner, nodes, "CANONIZE_BWD")

            # Recurse into right
            self._compile_construct(right)

        elif chain_op == TokenType.CONNOTATE_BWD:
            # CONNOTATE_BWD: {r[0].sig: [left_primaries[-1].sig]}
            if right_primaries:
                owner = right_primaries[0].sig.id
                self._emit_mcs(owner)
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

    def _node_to_string(self, node: Node) -> str:
        """Convert a Node to its string representation."""
        if isinstance(node, Signature):
            return node.id
        elif isinstance(node, Literal):
            return node.id
        return str(node)

    def _is_signature(self, node: Node) -> bool:
        """Check if a node is a signature (uppercase alpha)."""
        if isinstance(node, Signature):
            return True
        if isinstance(node, Literal):
            return False
        return False

    def _emit(self, sig: str, nodes: str | None | list[str], op: str) -> None:
        """Emit an op-mapped entry."""
        dbg = self._format_dbg(sig, nodes) if self.dev else ""
        self.entries.append(OpKLine(sig=sig, nodes=nodes, op=op, dbg=dbg))

    def _format_dbg(self, sig: str, nodes: str | None | list[str]) -> str:
        """Format debug representation."""
        if nodes is None:
            return f"{sig}: None"
        elif isinstance(nodes, str):
            return f"{sig}: {nodes}"
        else:
            return f"{sig}: {nodes}"


def compile_v2(source: str, dev: bool = False) -> list[OpKLine]:
    """Compile KScript source to op-mapped entries."""
    from .lexer import Lexer

    tokens = Lexer(source).tokenize()
    kfile = Parser(tokens).parse()
    return CompilerV2(dev=dev).compile(kfile)
