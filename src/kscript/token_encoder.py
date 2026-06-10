"""Token Encoder — converts symbolic KScript entries to tokenized CompiledEntry objects.

This module is responsible for string → token ID conversion using a tokenizer.
It takes symbolic entries from ASTEmitter and produces CompiledEntry objects
ready for the knowledge graph.
"""

from __future__ import annotations

from kalvin.abstract import KTokenizer
from kalvin.kline import KLine, KNodes, KSig
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.signature import LITERAL_MASK, is_literal_node


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
        tokenizer: KTokenizer,
        *,
        sig_level: str = "S4",
        significance: object | None = None,
        dbg_text: str = ""
    ) -> CompiledEntry:
        """Encode string signature/nodes to token IDs.

        Signatures (uppercase strings) are packed automatically.
        Literals are encoded as literal nodes automatically.

        When tokenizer.supports_mcs is False (NLP-BPE), all BPE tokens
        from encode() are used for node positions instead of taking only
        the first token for uppercase identifiers.
        """
        sig_id = tokenizer.encode(sig)[0]
        if nodes is None:
            return cls(signature=sig_id, nodes=None, dbg_text=dbg_text, sig_level=sig_level)
        elif isinstance(nodes, str):
            if tokenizer.supports_mcs and nodes.isupper() and nodes.isalpha():
                # Mod32 mode: packed encoding, single token expected
                node_id = tokenizer.encode(nodes)[0]
                return cls(signature=sig_id, nodes=node_id, dbg_text=dbg_text, sig_level=sig_level)
            elif nodes.isupper() and nodes.isalpha():
                # NLP-BPE mode: use all BPE tokens
                node_ids = tokenizer.encode(nodes)
                return cls(signature=sig_id, nodes=node_ids[0] if len(node_ids) == 1 else node_ids,
                           dbg_text=dbg_text, sig_level=sig_level)
            else:
                # Literal: character-level encoding (all modes)
                node_ids = [(ord(c) << 32) | LITERAL_MASK for c in nodes]
                return cls(signature=sig_id, nodes=node_ids[0] if len(node_ids) == 1 else node_ids,
                           dbg_text=dbg_text, sig_level=sig_level)
        else:
            all_node_ids: list[int] = []
            for n in nodes:
                if n.isupper() and n.isalpha():
                    if tokenizer.supports_mcs:
                        # Mod32: single packed token
                        all_node_ids.append(tokenizer.encode(n)[0])
                    else:
                        # NLP-BPE: all BPE tokens
                        all_node_ids.extend(tokenizer.encode(n))
                else:
                    # Literal: character-level encoding
                    all_node_ids.extend((ord(c) << 32) | LITERAL_MASK for c in n)
            return cls(signature=sig_id, nodes=all_node_ids, dbg_text=dbg_text, sig_level=sig_level)

    def decode(self, tokenizer: KTokenizer) -> tuple[str, str | None | list[str]]:
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

    def __init__(self, tokenizer: KTokenizer | None = None, dev: bool = False):
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev
        self._decomposed_sigs: set[str] = set()
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
        results: list[CompiledEntry] = []
        for entry in symbolic_entries:
            # BPE decomposition for multi-token identifiers (NLP mode)
            if not self.tokenizer.supports_mcs:
                decomp = self._bpe_decompose(entry.sig)
                if decomp:
                    results.extend(decomp)
            results.append(self._encode_one(entry))
        return results

    def _bpe_decompose(self, sig_str: str) -> list[CompiledEntry] | None:
        """Emit BPE decomposition entries for multi-token identifiers.

        Per spec §5.3: when an identifier BPE-encodes to multiple tokens,
        emit unsigned entries for each component and a decomposition
        (canonize) entry mapping the first token to all tokens.

        Only active when tokenizer.supports_mcs is False (NLP-BPE mode).
        Only applies to uppercase identifiers — literals are always
        character-level and don't get BPE decomposition.
        Returns None for single-token identifiers or Mod32 mode.
        """
        if self.tokenizer.supports_mcs:
            return None
        # Only decompose uppercase alpha identifiers (signatures)
        if not sig_str.isupper() or not sig_str.isalpha():
            return None
        if sig_str in self._decomposed_sigs:
            return None
        tokens = self.tokenizer.encode(sig_str)
        if len(tokens) <= 1:
            return None

        self._decomposed_sigs.add(sig_str)
        entries: list[CompiledEntry] = []

        # Unsigned entry for each component token
        for tok in tokens:
            entries.append(CompiledEntry(
                signature=tok, nodes=None, sig_level="S4",
                dbg_text=f"[S4] {sig_str} component" if self.dev else ""
            ))

        # Decomposition entry: {first_token: [all_tokens]} (S2)
        entries.append(CompiledEntry(
            signature=tokens[0], nodes=tokens, sig_level="S2",
            dbg_text=f"[S2] {sig_str} decomposition" if self.dev else ""
        ))

        return entries

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
        """Encode a single node string to token ID (always returns int).

        Uppercase identifiers use tokenizer.encode()[0] (first BPE token
        for NLP-BPE, packed value for Mod32). Non-uppercase strings use
        literal character-level encoding (first character only).
        """
        if node.isupper() and node.isalpha():
            return self.tokenizer.encode(node)[0]
        else:
            # Literal: character-level encoding of first character
            return (ord(node[0]) << 32) | LITERAL_MASK

    def _encode_nodes(self, nodes: str | None | list[str]) -> None | int | list[int]:
        """Encode nodes to token IDs.

        When tokenizer.supports_mcs is True (Mod32):
          - Uppercase alpha → single packed int
          - Non-uppercase → literal character nodes (via tokenizer)

        When tokenizer.supports_mcs is False (NLP-BPE):
          - Uppercase alpha → all BPE tokens from encode()
          - Non-uppercase → literal character nodes (char-level)
        """
        if nodes is None:
            return None
        elif isinstance(nodes, str):
            if nodes.isupper() and nodes.isalpha():
                if self.tokenizer.supports_mcs:
                    # Mod32: packed encoding, single token
                    return self.tokenizer.encode(nodes)[0]
                else:
                    # NLP-BPE: all BPE tokens
                    encoded = self.tokenizer.encode(nodes)
                    return encoded[0] if len(encoded) == 1 else encoded
            else:
                # Literal: character-level encoding (all modes)
                encoded = [(ord(c) << 32) | LITERAL_MASK for c in nodes]
                return encoded[0] if len(encoded) == 1 else encoded
        else:
            result: list[int] = []
            for n in nodes:
                if n.isupper() and n.isalpha():
                    if self.tokenizer.supports_mcs:
                        # Mod32: single packed token
                        result.append(self.tokenizer.encode(n)[0])
                    else:
                        # NLP-BPE: all BPE tokens
                        result.extend(self.tokenizer.encode(n))
                else:
                    # Literal: character-level encoding
                    result.extend((ord(c) << 32) | LITERAL_MASK for c in n)
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
