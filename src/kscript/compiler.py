"""Compiler for KScript AST to compiled entries."""

from kalvin.abstract import KLine, KNodes, KSig
from kalvin.mod_tokenizer import Mod32Tokenizer, ModTokenizer, PACKED_BIT
from kalvin.significance import Int32Significance

from .ast import (
    Construct,
    ConstructType,
    KScriptFile,
    Node,
    NumberLiteral,
    Script,
    Signature,
    StringLiteral,
)

# Significance instance for encoding
_sig = Int32Significance()


class CompiledEntry(KLine):
    """A single compiled KLine entry.

    Nodes semantics:
    - None: identity entry (sig exists with no children)
    - int: single token ID link (countersign, undersign)
    - list[int]: nodes list (connotate, canonize)
    """

    def __init__(self, signature: KSig, nodes: KNodes, dbg_text: str = ""):
        super().__init__(signature=signature, nodes=nodes, dbg_text=dbg_text)

    @classmethod
    def add_significance(cls, token_id: int, construct_type: ConstructType) -> int:
        """Add significance bits to token ID based on construct type.

        S1 (countersign): bit 63
        S2 (canonize): bit 55
        S3 (connotate): bit 32
        S4 (undersign): no bits (identity)
        """
        if construct_type == ConstructType.COUNTERSIGN:
            return token_id | _sig.S1
        elif construct_type in (ConstructType.CANONIZE_FWD, ConstructType.CANONIZE_BWD):
            return token_id | _sig.S2
        elif construct_type in (ConstructType.CONNOTATE_FWD, ConstructType.CONNOTATE_BWD):
            return token_id | _sig.S3
        else:  # UNDERSIGN or unknown
            return token_id  # S4: no significance bits

    @classmethod
    def encode(
        cls,
        sig: str,
        nodes: str | None | list[str],
        tokenizer: ModTokenizer,
        *,
        construct_type: ConstructType | None = None
    ) -> "CompiledEntry":
        """Encode string signature/nodes to token IDs.

        Signatures (uppercase strings) are packed (PACKED_BIT set) using tokenizer.
        Literals (non-uppercase strings) are unpacked (no PACKED_BIT).
        For literals, each character is encoded as (ord(char) << 1), leaving bit 0 clear.

        Args:
            sig: Signature string to encode
            nodes: Node(s) to encode - None, string, or list of strings
            tokenizer: Tokenizer for encoding
            construct_type: If provided, add significance bits based on construct type

        Returns:
            CompiledEntry with encoded signature and nodes
        """
        sig_id = tokenizer.encode(sig, pack=True)[0]

        # Add significance bits if construct type is specified
        if construct_type is not None:
            sig_id = cls.add_significance(sig_id, construct_type)

        if nodes is None:
            return cls(signature=sig_id, nodes=None)
        elif isinstance(nodes, str):
            if nodes.isupper():
                # Signature: single packed token
                node_id = tokenizer.encode(nodes, pack=True)[0]
                return cls(signature=sig_id, nodes=node_id)
            else:
                # Literal: encode each char as (ord << 1)
                node_ids = [ord(c) << 1 for c in nodes]
                return cls(signature=sig_id, nodes=node_ids)
        else:
            # For list of nodes: each node gets its own token(s)
            all_node_ids: list[int] = []
            for n in nodes:
                if n.isupper():
                    # Signature: single packed token
                    all_node_ids.append(tokenizer.encode(n, pack=True)[0])
                else:
                    # Literal: encode each char as (ord << 1)
                    all_node_ids.extend(ord(c) << 1 for c in n)
            return cls(signature=sig_id, nodes=all_node_ids)

    def decode(self, tokenizer: ModTokenizer) -> tuple[str, str | None | list[str]]:
        """Decode token IDs back to strings using auto-detection of packed/literal."""
        # Strip significance bits (keep only token bits 0-31)
        sig_token = self.signature & _sig.TOKEN_MASK
        sig = tokenizer.decode([sig_token], pack=None)

        # Use _nodes to get raw value (None, int, or list) not the normalized list
        if self._nodes is None:
            return sig, None
        elif isinstance(self._nodes, int):
            decoded = tokenizer.decode([self._nodes], pack=None)
            return sig, decoded
        else:
            # Check if all tokens are unpacked (literal) or mixed
            all_unpacked = all((n & PACKED_BIT) == 0 for n in self._nodes)

            if all_unpacked:
                # All literal: decode as single string (for countersign/undersign)
                decoded = "".join(chr(n >> 1) for n in self._nodes)
                return sig, decoded
            else:
                # Mixed literal/signature tokens
                # Literals: consecutive unpacked tokens form a string (each is ord << 1)
                # Signatures: single packed token decoded via tokenizer
                decoded_nodes: list[str] = []
                literal_chars: list[int] = []

                for n in self._nodes:
                    if (n & PACKED_BIT) == 0:
                        # Unpacked: part of a literal string
                        literal_chars.append(n)
                    else:
                        # Packed: signature token
                        # First, flush any accumulated literal chars
                        if literal_chars:
                            decoded_nodes.append("".join(chr(c >> 1) for c in literal_chars))
                            literal_chars = []
                        # Decode the signature
                        decoded_nodes.append(tokenizer.decode([n], pack=None))

                # Flush any remaining literal chars
                if literal_chars:
                    decoded_nodes.append("".join(chr(c >> 1) for c in literal_chars))

                return sig, decoded_nodes


class Compiler:
    """Compiles KScript AST to list of CompiledEntry objects."""

    def __init__(self, tokenizer: ModTokenizer | None = None):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()

    def _expand_mcs(self, sig: str) -> None:
        """Emit MCS canonization and identity entries for multi-character signatures.

        If sig has length > 1, emits:
        - {sig: [component chars]} with S2 significance (CANONIZE_FWD)
        - {component: null} for each char with S4 significance

        Single-character signatures are not expanded (atomic).
        """
        if len(sig) <= 1:
            return

        # Emit MCS canonization: {sig: [A, B, C, ...]} with S2 significance
        component_chars = list(sig)
        self.entries.append(
            CompiledEntry.encode(sig, component_chars, self.tokenizer, construct_type=ConstructType.CANONIZE_FWD)
        )

        # Emit component identities: {A: null}, {B: null}, ... with S4 significance
        for char in component_chars:
            self.entries.append(
                CompiledEntry.encode(char, None, self.tokenizer, construct_type=ConstructType.UNDERSIGN)
            )

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile to entries."""
        for script in file.scripts:
            self._compile_script(script)
        return self.entries

    def _compile_script(self, script: Script) -> None:
        """Compile a single script with immediate binding semantics."""
        sig = script.signature

        # Expand MCS for script signature before processing constructs
        self._expand_mcs(sig.id)

        # Get all nodes for constructs, using subscript signatures if needed
        constructs_with_nodes: list[tuple[Construct, list[Node]]] = []
        for i, construct in enumerate(script.constructs):
            is_last = i == len(script.constructs) - 1
            nodes = self._get_construct_nodes(construct, script, is_last)
            constructs_with_nodes.append((construct, nodes))

        # Check if any construct has nodes
        has_valid_constructs = any(nodes for _, nodes in constructs_with_nodes)

        if not has_valid_constructs:
            # Identity script: {sig: None}
            # Skip for multi-char signatures since MCS canonization already defines identity
            if len(sig.id) <= 1:
                self.entries.append(
                    CompiledEntry.encode(sig.id, None, self.tokenizer)
                )
        else:
            # Process constructs with immediate binding
            current_sig = sig
            previous_nodes: list[Node] = []

            for construct, nodes in constructs_with_nodes:
                if nodes:
                    self._compile_construct(current_sig, construct, nodes, previous_nodes)

                    # Update for next construct
                    previous_nodes = nodes
                    last_node = nodes[-1]
                    if isinstance(last_node, Signature):
                        current_sig = last_node

        # Compile subscripts
        for subscript in script.subscripts:
            self._compile_script(subscript)

    def _get_construct_nodes(
        self, construct: Construct, script: Script, is_last: bool
    ) -> list[Node]:
        """Get nodes for a construct, including subscript signatures if needed."""
        # Start with inline nodes
        nodes: list[Node] = construct.nodes

        # If no inline nodes and this is the last construct, use subscript signatures
        if not nodes and is_last and script.subscripts:
            nodes = [sub.signature for sub in script.subscripts]

        return nodes

    def _compile_construct(
        self, sig: Signature, construct: Construct, nodes: list[Node], previous_nodes: list[Node]
    ) -> None:
        """Compile a construct to entries based on its type."""
        construct_type = construct.type

        if construct_type == ConstructType.COUNTERSIGN:
            # Bidirectional: {sig: node} AND {node: sig}
            # But if node is a literal, recover undersign (no reverse)
            for node in nodes:
                self.entries.append(
                    CompiledEntry.encode(sig.id, node.id, self.tokenizer, construct_type=ConstructType.COUNTERSIGN)
                )
                # Only add reverse if node is signature-like (not a literal)
                if isinstance(node, Signature):
                    self.entries.append(
                        CompiledEntry.encode(node.id, sig.id, self.tokenizer, construct_type=ConstructType.COUNTERSIGN)
                    )

        elif construct_type in (ConstructType.CANONIZE_FWD, ConstructType.CANONIZE_BWD):
            # Canonize: {sig: [nodes...]}
            if construct_type == ConstructType.CANONIZE_BWD:
                # B <= A means {A: [B]} (current sig is child, node is parent)
                # X <= A B means {B: [A]} (nodes before last are children)
                # B C D <= A means {A: [B, C, D]} (script sig + leading nodes are children)
                # C => B D <= A means {A: [B, D]} (previous nodes are children)
                if len(nodes) >= 1:
                    parent = nodes[-1]
                    children = nodes[:-1]

                    if construct.has_leading_nodes:
                        # Include script signature with leading nodes
                        children = [sig] + children
                    elif not children:
                        # No nodes before parent, use previous nodes as children
                        children = previous_nodes if previous_nodes else [sig]

                    child_node_ids = [n.id for n in children]

                    self.entries.append(
                        CompiledEntry.encode(parent.id, child_node_ids, self.tokenizer, construct_type=ConstructType.CANONIZE_BWD)
                    )
            else:
                # Forward canonize: {sig: [nodes...]}
                node_ids = [n.id for n in nodes]
                self.entries.append(
                    CompiledEntry.encode(sig.id, node_ids, self.tokenizer, construct_type=ConstructType.CANONIZE_FWD)
                )

        elif construct_type == ConstructType.CONNOTATE_FWD:
            # Forward connotate: {sig: [node]} AND {node: None}
            for node in nodes:
                self.entries.append(
                    CompiledEntry.encode(sig.id, [node.id], self.tokenizer, construct_type=ConstructType.CONNOTATE_FWD)
                )
                self.entries.append(
                    CompiledEntry.encode(node.id, None, self.tokenizer, construct_type=ConstructType.CONNOTATE_FWD)
                )

        elif construct_type == ConstructType.CONNOTATE_BWD:
            # Backward connotate: {node: [sig]} AND {sig: None}
            for node in nodes:
                self.entries.append(
                    CompiledEntry.encode(node.id, [sig.id], self.tokenizer, construct_type=ConstructType.CONNOTATE_BWD)
                )
            self.entries.append(
                CompiledEntry.encode(sig.id, None, self.tokenizer, construct_type=ConstructType.CONNOTATE_BWD)
            )

        elif construct_type == ConstructType.UNDERSIGN:
            # Undersign: {sig: node} AND {node: None}
            for node in nodes:
                self.entries.append(
                    CompiledEntry.encode(sig.id, node.id, self.tokenizer, construct_type=ConstructType.UNDERSIGN)
                )
                self.entries.append(
                    CompiledEntry.encode(node.id, None, self.tokenizer, construct_type=ConstructType.UNDERSIGN)
                )
