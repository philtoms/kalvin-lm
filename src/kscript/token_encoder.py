"""Token Encoder — converts symbolic KScript entries to tokenized CompiledEntry objects.

This module is responsible for string → token ID conversion using a tokenizer.
It takes symbolic entries from ASTEmitter and produces CompiledEntry objects
ready for the knowledge graph.
"""

from __future__ import annotations

from kalvin.abstract import KTokenizer
from kalvin.kline import KLine, KNodes, KSig
from kalvin.mod_tokenizer import Mod32Tokenizer


class CompiledEntry(KLine):
    """A compiled KLine entry with decode support.

    Nodes semantics:
    - None: unsigned entry (sig exists with no children)
    - int: single token ID link (countersign, undersign)
    - list[int]: nodes list (connotate, canonize)

    When `_nodes_single` is True, the list of token IDs came from encoding
    a single string and should be batch-decoded. When False (or unset),
    each token ID is a separate node and decoded individually.
    """

    def __init__(self, signature: KSig, nodes: KNodes, dbg_text: str = "",
                 sig_level: str | None = None, _nodes_single: bool = False):
        super().__init__(signature=signature, nodes=nodes, dbg_text=dbg_text,
                         sig_level=sig_level)
        self._nodes_single = _nodes_single

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

        Signatures (uppercase strings) are encoded via the tokenizer.
        All strings go through the tokenizer — no branching.

        When tokenizer.supports_mcs is False (NLP-BPE), all BPE tokens
        from encode() are used for node positions instead of taking only
        the first token for uppercase identifiers.
        """
        sig_id = tokenizer.encode(sig)[0]
        if nodes is None:
            return cls(signature=sig_id, nodes=None, dbg_text=dbg_text, sig_level=sig_level)
        elif isinstance(nodes, str):
            is_sig = nodes.isupper() and nodes.isalpha()
            if tokenizer.supports_mcs and is_sig:
                # Mod32 mode: packed encoding, single token expected
                node_id = tokenizer.encode(nodes)[0]
                return cls(signature=sig_id, nodes=node_id, dbg_text=dbg_text, sig_level=sig_level)
            elif is_sig:
                # NLP-BPE mode: use all BPE tokens
                node_ids = tokenizer.encode(nodes)
                return cls(signature=sig_id, nodes=node_ids[0] if len(node_ids) == 1 else node_ids,
                           dbg_text=dbg_text, sig_level=sig_level, _nodes_single=False)
            else:
                # Non-signature string: encode via tokenizer, batch decode
                node_ids = tokenizer.encode(nodes)
                return cls(signature=sig_id, nodes=node_ids[0] if len(node_ids) == 1 else node_ids,
                           dbg_text=dbg_text, sig_level=sig_level, _nodes_single=True)
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
                    # Non-signature string: encode via tokenizer
                    all_node_ids.extend(tokenizer.encode(n))
            return cls(signature=sig_id, nodes=all_node_ids, dbg_text=dbg_text, sig_level=sig_level)

    def decode(self, tokenizer: KTokenizer) -> tuple[str, str | None | list[str]]:
        """Decode token IDs back to strings."""
        sig = tokenizer.decode([self.signature])

        if not self.nodes:
            return sig, None
        elif isinstance(self.nodes, int):
            decoded = tokenizer.decode([self.nodes])
            return sig, decoded
        elif self._nodes_single:
            # Nodes came from encoding a single string — batch decode
            decoded = tokenizer.decode(list(self.nodes))
            return sig, decoded
        else:
            # Nodes came from encoding multiple strings — individual decode
            decoded_nodes: list[str] = []
            for n in self.nodes:
                decoded = tokenizer.decode([n])
                decoded_nodes.append(decoded)

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
        emit unsigned entries for each component token and a decomposition
        (canonize) entry mapping the first token to all tokens.

        Only active when tokenizer.supports_mcs is False (NLP-BPE mode).
        Only applies to uppercase identifiers.
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
        encoded_nodes, nodes_single = self._encode_nodes(entry.nodes)

        level = self._sig_levels.get(entry.op, "S4")
        dbg = self._format_dbg(entry.sig, entry.nodes, entry.op) if self.dev else ""
        return CompiledEntry(signature=sig_id, nodes=encoded_nodes,
                              dbg_text=dbg, sig_level=level,
                              _nodes_single=nodes_single)

    def _encode_sig(self, sig: str) -> int:
        """Encode a signature string to token ID."""
        return self.tokenizer.encode(sig)[0]

    def _encode_node(self, node: str) -> int:
        """Encode a single node string to token ID (always returns int).

        Uppercase identifiers use tokenizer.encode()[0] (first BPE token
        for NLP-BPE, packed value for Mod32). Non-uppercase strings use
        the tokenizer directly.
        """
        if node.isupper() and node.isalpha():
            return self.tokenizer.encode(node)[0]
        else:
            # Encode via tokenizer
            encoded = self.tokenizer.encode(node)
            return encoded[0]

    def _encode_nodes(self, nodes: str | None | list[str]) -> tuple[None | int | list[int], bool]:
        """Encode nodes to token IDs.

        Returns (encoded_nodes, is_single) where is_single indicates whether
        the nodes came from encoding a single string (used by decode for
        batch vs individual decoding).

        When tokenizer.supports_mcs is True (Mod32):
          - Uppercase alpha → single packed int
          - Non-uppercase → encode via tokenizer

        When tokenizer.supports_mcs is False (NLP-BPE):
          - Uppercase alpha → all BPE tokens from encode()
          - Non-uppercase → encode via tokenizer
        """
        if nodes is None:
            return None, False
        elif isinstance(nodes, str):
            is_sig = nodes.isupper() and nodes.isalpha()
            if is_sig:
                if self.tokenizer.supports_mcs:
                    # Mod32: packed encoding, single token
                    return self.tokenizer.encode(nodes)[0], False
                else:
                    # NLP-BPE: all BPE tokens, individual decode
                    encoded = self.tokenizer.encode(nodes)
                    return (encoded[0] if len(encoded) == 1 else encoded), False
            else:
                # Non-signature string: encode via tokenizer, batch decode
                encoded = self.tokenizer.encode(nodes)
                return (encoded[0] if len(encoded) == 1 else encoded), True
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
                    # Non-signature string: encode via tokenizer
                    result.extend(self.tokenizer.encode(n))
            return result, False

    def _format_dbg(self, sig: str, nodes: str | None | list[str], op: str) -> str:
        """Format debug representation with significance level."""
        level = self._sig_levels.get(op, "S4")
        if nodes is None:
            return f"[{level}] {sig}: None"
        elif isinstance(nodes, str):
            return f"[{level}] {sig}: {nodes}"
        else:
            return f"[{level}] {sig}: {nodes}"
