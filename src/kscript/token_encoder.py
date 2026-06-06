"""Token Encoder — converts symbolic KScript entries to tokenized CompiledEntry objects.

This module is responsible for string → token ID conversion using a tokenizer.
It takes symbolic entries from ASTEmitter and produces CompiledEntry objects
ready for the knowledge graph.
"""

from __future__ import annotations

from kalvin.kline import KLine, KNodes, KSig
from kalvin.mod_tokenizer import Mod32Tokenizer, ModTokenizer
from kalvin.signature import is_literal_node


class CompiledEntry(KLine):
    """A compiled KLine entry with decode support.

    Nodes semantics:
    - None: unsigned entry (sig exists with no children)
    - int: single token ID link (countersign, undersign)
    - list[int]: nodes list (connotate, canonize)
    """

    def __init__(self, signature: KSig, nodes: KNodes, dbg_text: str = "",
                 sig_level: str | None = None):
        super().__init__(signature=signature, nodes=nodes, dbg_text=dbg_text,
                         sig_level=sig_level)

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
        sig_id = tokenizer.encode(sig)[0]
        if nodes is None:
            return cls(signature=sig_id, nodes=None, dbg_text=dbg_text, sig_level=sig_level)
        elif isinstance(nodes, str):
            if nodes.isupper() and nodes.isalpha():
                node_id = tokenizer.encode(nodes)[0]
                return cls(signature=sig_id, nodes=node_id, dbg_text=dbg_text, sig_level=sig_level)
            else:
                node_ids = tokenizer.encode(nodes)
                return cls(signature=sig_id, nodes=node_ids, dbg_text=dbg_text, sig_level=sig_level)
        else:
            all_node_ids: list[int] = []
            for n in nodes:
                if n.isupper() and n.isalpha():
                    all_node_ids.append(tokenizer.encode(n)[0])
                else:
                    all_node_ids.extend(tokenizer.encode(n))
            return cls(signature=sig_id, nodes=all_node_ids, dbg_text=dbg_text, sig_level=sig_level)

    def decode(self, tokenizer: ModTokenizer) -> tuple[str, str | None | list[str]]:
        """Decode token IDs back to strings."""
        sig = tokenizer.decode([self.signature])

        if self.nodes is None:
            return sig, None
        elif isinstance(self.nodes, int):
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


class TokenEncoder:
    """Converts symbolic entries to tokenized CompiledEntry objects.

    Parameters
    ----------
    tokenizer:
        Tokenizer for encoding strings to token IDs.
    dev:
        If True, include debug text in entries.
    """

    def __init__(self, tokenizer: ModTokenizer | None = None, dev: bool = False):
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

    def encode_entries(self, symbolic_entries: list) -> list[CompiledEntry]:
        """Convert a list of SymbolicEntry to CompiledEntry objects.

        Args:
            symbolic_entries: List of SymbolicEntry(sig, nodes, op) named tuples.

        Returns:
            List of CompiledEntry objects with token IDs.
        """
        from .ast_emitter import SymbolicEntry
        results: list[CompiledEntry] = []
        for entry in symbolic_entries:
            results.append(self._encode_one(entry))
        return results

    def _encode_one(self, entry) -> CompiledEntry:
        """Encode a single SymbolicEntry to CompiledEntry."""
        sig_id = self._encode_node(entry.sig)
        encoded_nodes = self._encode_nodes(entry.nodes)

        level = self._sig_levels.get(entry.op, "S4")
        dbg = self._format_dbg(entry.sig, entry.nodes, entry.op) if self.dev else ""
        return CompiledEntry(signature=sig_id, nodes=encoded_nodes,
                              dbg_text=dbg, sig_level=level)

    def _encode_sig(self, sig: str) -> int:
        """Encode a signature string to token ID."""
        return self.tokenizer.encode(sig)[0]

    def _encode_node(self, node: str) -> int:
        """Encode a single node string to token ID."""
        if node.isupper() and node.isalpha():
            return self._encode_sig(node)
        else:
            return self.tokenizer.encode(node[0] if len(node) > 1 else node)[0]

    def _encode_nodes(self, nodes: str | None | list[str]) -> None | int | list[int]:
        """Encode nodes to token IDs."""
        if nodes is None:
            return None
        elif isinstance(nodes, str):
            if nodes.isupper() and nodes.isalpha():
                return self.tokenizer.encode(nodes)[0]
            elif len(nodes) == 1:
                return self.tokenizer.encode(nodes)[0]
            else:
                return self.tokenizer.encode(nodes)
        else:
            result: list[int] = []
            for n in nodes:
                if n.isupper() and n.isalpha():
                    result.append(self.tokenizer.encode(n)[0])
                else:
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
