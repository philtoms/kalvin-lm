"""Compiler for KScript AST to compiled entries."""

from kalvin.abstract import KLine, KNodes, KSig
from kalvin.mod_tokenizer import ModTokenizer, Mod32Tokenizer

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


class CompiledEntry(KLine):
    """A single compiled KLine entry.

    nodes semantics:
    - None: identity entry (sig exists with no children)
    - int: single token ID link (countersign, undersign)
    - list[int]: nodes list (connotate, canonize)
    """

    def __init__(self, signature: KSig, nodes: KNodes, dbg_text: str = ""):
        super().__init__(signature=signature, nodes=nodes, dbg_text=dbg_text)

    @classmethod
    def encode(
        cls, sig: str, nodes: str | None | list[str], tokenizer: ModTokenizer
    ) -> "CompiledEntry":
        """Encode string signature/nodes to token IDs using packed encoding."""
        sig_id = tokenizer.encode(sig, pack=True)[0]

        if nodes is None:
            return cls(signature=sig_id, nodes=None)
        elif isinstance(nodes, str):
            node_id = tokenizer.encode(nodes, pack=True)[0]
            return cls(signature=sig_id, nodes=node_id)
        else:
            # For list of nodes: each node gets its own packed token
            node_ids = [tokenizer.encode(n, pack=True)[0] for n in nodes]
            return cls(signature=sig_id, nodes=node_ids)

    def decode(self, tokenizer: ModTokenizer) -> tuple[str, str | None | list[str]]:
        """Decode token IDs back to strings using packed decoding."""
        sig = tokenizer.decode([self.signature], pack=True)
        if self.nodes is None:
            return sig, None
        elif isinstance(self.nodes, int):
            return sig, tokenizer.decode([self.nodes], pack=True)
        else:
            return sig, [tokenizer.decode([n], pack=True) for n in self.nodes]


class Compiler:
    """Compiles KScript AST to list of CompiledEntry objects."""

    def __init__(self, tokenizer: ModTokenizer | None = None):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile to entries."""
        for script in file.scripts:
            self._compile_script(script)
        return self.entries

    def _compile_script(self, script: Script) -> None:
        """Compile a single script with immediate binding semantics."""
        sig_name = script.signature.name

        # Get all nodes for constructs, using subscript signatures if needed
        constructs_with_nodes = []
        for i, construct in enumerate(script.constructs):
            is_last = i == len(script.constructs) - 1
            nodes = self._get_construct_nodes(construct, script, is_last)
            constructs_with_nodes.append((construct, nodes))

        # Check if any construct has nodes
        has_valid_constructs = any(nodes for _, nodes in constructs_with_nodes)

        if not has_valid_constructs:
            # Identity script: {sig: None}
            self.entries.append(
                CompiledEntry.encode(sig_name, None, self.tokenizer)
            )
        else:
            # Process constructs with immediate binding
            current_sig = sig_name
            previous_nodes: list[str] = []

            for construct, nodes in constructs_with_nodes:
                if nodes:
                    self._compile_construct(current_sig, construct, nodes, previous_nodes)

                    # Update for next construct
                    previous_nodes = nodes
                    last_node = nodes[-1]
                    if self._is_signature_like(last_node):
                        current_sig = last_node

        # Compile subscripts
        for subscript in script.subscripts:
            self._compile_script(subscript)

    def _get_construct_nodes(
        self, construct: Construct, script: Script, is_last: bool
    ) -> list[str]:
        """Get nodes for a construct, including subscript signatures if needed."""
        # Start with inline nodes
        nodes = [self._node_to_string(n) for n in construct.nodes]

        # If no inline nodes and this is the last construct, use subscript signatures
        if not nodes and is_last and script.subscripts:
            nodes = [sub.signature.name for sub in script.subscripts]

        return nodes

    def _compile_construct(
        self, sig: str, construct: Construct, nodes: list[str], previous_nodes: list[str]
    ) -> None:
        """Compile a construct to entries based on its type."""
        construct_type = construct.type
        if construct_type == ConstructType.COUNTERSIGN:
            # Bidirectional: {sig: node} AND {node: sig}
            for node in nodes:
                self.entries.append(
                    CompiledEntry.encode(sig, node, self.tokenizer)
                )
                self.entries.append(
                    CompiledEntry.encode(node, sig, self.tokenizer)
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

                    self.entries.append(
                        CompiledEntry.encode(parent, children, self.tokenizer)
                    )
            else:
                # Forward canonize: {sig: [nodes...]}
                self.entries.append(
                    CompiledEntry.encode(sig, nodes, self.tokenizer)
                )

        elif construct_type == ConstructType.CONNOTATE_FWD:
            # Forward connotate: {sig: [node]}
            for node in nodes:
                self.entries.append(
                    CompiledEntry.encode(sig, [node], self.tokenizer)
                )

        elif construct_type == ConstructType.CONNOTATE_BWD:
            # Backward connotate: {node: [sig]}
            for node in nodes:
                self.entries.append(
                    CompiledEntry.encode(node, [sig], self.tokenizer)
                )

        elif construct_type == ConstructType.UNDERSIGN:
            # Undersign: {sig: node}
            for node in nodes:
                self.entries.append(
                    CompiledEntry.encode(sig, node, self.tokenizer)
                )

    def _node_to_string(self, node: Node) -> str:
        """Convert a node to its string representation."""
        if isinstance(node, Signature):
            return node.name
        elif isinstance(node, StringLiteral):
            return node.value
        elif isinstance(node, NumberLiteral):
            return node.value
        return str(node)

    def _is_signature_like(self, value: str) -> bool:
        """Check if a value looks like a signature (all uppercase)."""
        return value.isupper() and value.isalpha()
