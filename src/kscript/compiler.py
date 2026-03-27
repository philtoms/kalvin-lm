"""Compiler for KScript AST to compiled entries.

Key changes from v1:
- Each Construct has an explicit owner (Signature)
- BWD operators: RIGHT side signature owns the construct
- Entity emission for all signatures that don't continue with FWD
- MCS expansion for signatures in signature position only
"""

from kalvin.abstract import KLine, KNodes, KSig
from kalvin.mod_tokenizer import Mod32Tokenizer, ModTokenizer, PACKED_BIT
from kalvin.significance import Int32Significance

from .ast import (
    Construct,
    ConstructType,
    KScriptFile,
    Literal,
    Node,
    Script,
    Signature,
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
        S4 (undersign/identity): no bits
        """
        if construct_type == ConstructType.COUNTERSIGN:
            return token_id | _sig.S1
        elif construct_type in (ConstructType.CANONIZE_FWD, ConstructType.CANONIZE_BWD):
            return token_id | _sig.S2
        elif construct_type in (ConstructType.CONNOTATE_FWD, ConstructType.CONNOTATE_BWD):
            return token_id | _sig.S3
        else:  # UNDERSIGN, IDENTITY or unknown
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
            if nodes.isupper() and nodes.isalpha():
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
                if n.isupper() and n.isalpha():
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
    """Compiles KScript AST to list of CompiledEntry objects.

    Key semantics:
    - Every construct has an owner (Signature)
    - BWD operators: RIGHT side signature owns the construct
    - Entity emission for signatures that don't continue with FWD
    - MCS expansion for signatures in signature position only
    """

    def __init__(self, tokenizer: ModTokenizer | None = None):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        # Track which signatures have had MCS expanded (avoid duplicates)
        self._mcs_emitted: set[str] = set()
        # Track which entities have been emitted (avoid duplicates)
        self._entities_emitted: set[str] = set()

    def _emit_mcs(self, sig: str) -> None:
        """Emit MCS canonization and identity entries for multi-character signatures.

        MCS expansion is ONLY for signatures in SIGNATURE POSITION (construct owner).

        Emits:
        - {sig: [component chars]} with S2 significance (CANONIZE_FWD)
        - {component: null} for each char with S4 significance (UNDERSIGN)

        Single-character signatures are not expanded (atomic).
        """
        if len(sig) <= 1:
            return

        # Avoid duplicate MCS expansion
        if sig in self._mcs_emitted:
            return
        self._mcs_emitted.add(sig)

        # Emit MCS canonization: {sig: [A, B, C, ...]} with S2 significance
        component_chars = list(sig)
        self.entries.append(
            CompiledEntry.encode(sig, component_chars, self.tokenizer, construct_type=ConstructType.CANONIZE_FWD)
        )

        # Emit component identities: {A: null}, {B: null}, ... with S4 significance
        for char in component_chars:
            self._emit_entity(char)

    def _emit_entity(self, sig: str) -> None:
        """Emit entity entry for a signature.

        Entity entries ensure the signature exists in the graph.
        Uses S4 significance (UNDERSIGN with None nodes).
        """
        # Avoid duplicate entity emission
        if sig in self._entities_emitted:
            return
        self._entities_emitted.add(sig)

        self.entries.append(
            CompiledEntry.encode(sig, None, self.tokenizer, construct_type=ConstructType.UNDERSIGN)
        )

    def _node_to_string(self, node: Node) -> str:
        """Convert a Node to its string representation."""
        if isinstance(node, Signature):
            return node.id
        elif isinstance(node, Literal):
            return node.id
        else:
            return str(node)

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile to entries."""
        for script in file.scripts:
            self._compile_script(script)
        return self.entries

    def _compile_script(self, script: Script) -> None:
        """Compile a single script.

        The script's primary signature is always in signature position.
        """
        primary_sig = script.signature

        # MCS expansion for primary signature
        self._emit_mcs(primary_sig.id)

        if not script.constructs:
            # Identity script: emit entity only if MCS didn't handle it
            # (single-char signatures need explicit entity)
            if len(primary_sig.id) <= 1:
                self._emit_entity(primary_sig.id)
            return

        # Track which signatures need entity emission
        # (signatures that don't continue with FWD)
        signatures_needing_entity: set[str] = set()

        # Process all constructs
        for construct in script.constructs:
            self._compile_construct(construct, signatures_needing_entity)

        # Emit entities for signatures that need them
        for sig_id in signatures_needing_entity:
            self._emit_entity(sig_id)

    def _compile_construct(
        self,
        construct: Construct,
        entities_needed: set[str]
    ) -> None:
        """Compile a single construct.

        Args:
            construct: The construct to compile
            entities_needed: Set to track signatures needing entity entries
        """
        owner = construct.owner
        construct_type = construct.type
        nodes = construct.nodes

        # MCS expansion for owner (signature position)
        self._emit_mcs(owner.id)

        # Convert nodes to strings
        node_ids = [self._node_to_string(n) for n in nodes]

        # Track entities for node signatures (except countersign - reverse serves as entity)
        if construct_type != ConstructType.COUNTERSIGN:
            for node in nodes:
                if isinstance(node, Signature):
                    entities_needed.add(node.id)

        if construct_type == ConstructType.COUNTERSIGN:
            # Bidirectional: {owner: node} AND {node: owner}
            for node_id in node_ids:
                self.entries.append(
                    CompiledEntry.encode(owner.id, node_id, self.tokenizer, construct_type=ConstructType.COUNTERSIGN)
                )
                # Only add reverse if node is signature-like (not a literal)
                if node_id.isupper() and node_id.isalpha():
                    self.entries.append(
                        CompiledEntry.encode(node_id, owner.id, self.tokenizer, construct_type=ConstructType.COUNTERSIGN)
                    )

        elif construct_type == ConstructType.CANONIZE_FWD:
            # Forward canonize: {owner: [nodes...]}
            self.entries.append(
                CompiledEntry.encode(owner.id, node_ids, self.tokenizer, construct_type=ConstructType.CANONIZE_FWD)
            )
            # Nodes need entities (they don't continue with FWD from here)
            for node_id in node_ids:
                if node_id.isupper() and node_id.isalpha():
                    entities_needed.add(node_id)

        elif construct_type == ConstructType.CANONIZE_BWD:
            # Backward canonize: {owner: [nodes...]}
            # The owner (RIGHT side of BWD) points back to the nodes (LEFT side)
            self.entries.append(
                CompiledEntry.encode(owner.id, node_ids, self.tokenizer, construct_type=ConstructType.CANONIZE_BWD)
            )
            # Nodes need entities
            for node_id in node_ids:
                if node_id.isupper() and node_id.isalpha():
                    entities_needed.add(node_id)

        elif construct_type == ConstructType.CONNOTATE_FWD:
            # Forward connotate: {owner: [node]} AND {node: None}
            for node_id in node_ids:
                self.entries.append(
                    CompiledEntry.encode(owner.id, [node_id], self.tokenizer, construct_type=ConstructType.CONNOTATE_FWD)
                )
                # Entity will be emitted at end via entities_needed

        elif construct_type == ConstructType.CONNOTATE_BWD:
            # Backward connotate: {owner: [node]} AND entity for node
            self.entries.append(
                CompiledEntry.encode(owner.id, node_ids, self.tokenizer, construct_type=ConstructType.CONNOTATE_BWD)
            )
            # Entity will be emitted at end via entities_needed

        elif construct_type == ConstructType.UNDERSIGN:
            # Undersign: {owner: node} AND {node: None}
            for node_id in node_ids:
                self.entries.append(
                    CompiledEntry.encode(owner.id, node_id, self.tokenizer, construct_type=ConstructType.UNDERSIGN)
                )
                # Entity will be emitted at end via entities_needed
