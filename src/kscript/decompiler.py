"""Decompiler for KLines back to KScript source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kalvin.mod_tokenizer import ModTokenizer, Mod32Tokenizer, PACKED_BIT
from kalvin.significance import Int32Significance
from kalvin.kline import KLine


@dataclass
class DecompiledEntry:
    """A single decompiled KLine."""
    level: str  # S1, S2, S3, S4
    sig: str  # Decoded signature name
    nodes: list[str] | str | None  # Decoded node names

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "level": self.level,
            "sig": self.sig,
            "nodes": self.nodes,
        }

    def to_kscript(self) -> str:
        """Format as KScript source."""
        op_map = {
            "S1": "==",
            "S2": "=>",
            "S3": ">",
            "S4": "=",
        }

        op = op_map.get(self.level, "=")

        if self.nodes is None:
            return self.sig
        else:
            nodes_str = self._format_nodes(self.nodes)
            return f"{self.sig} {op} {nodes_str}"

    def _format_nodes(self, nodes: list[str] | str | None) -> str:
        """Format nodes for display."""
        if nodes is None:
            return ""
        if isinstance(nodes, str):
            return nodes
        return " ".join(nodes)


class Decompiler:
    """Decompiles KLines to KScript source.

    KLines are processed in sequence order. MCS entries provide
    name mappings for multi-character signatures but are not output.

    Significance bits determine construct types:
        - S1 (bit 63): countersign (==)
        - S1 (bit 63): undersign (=)
        - S2 (bits 55-62): canonize (=>)
        - S3 (bits 32-54): connotate (>)
        - S4 (no bits): unsigned 
    """

    def __init__(self, tokenizer: ModTokenizer | None = None):
        self.tokenizer = tokenizer if tokenizer else Mod32Tokenizer()
        self._sig = Int32Significance()
        self._mcs_names: dict[int, str] = {}

    def decompile(self, klines: list[KLine]) -> list[DecompiledEntry]:
        """Convert KLines to decompiled entries in sequence order.

        Args:
            klines: Ordered list of KLines

        Returns:
            List of DecompiledEntry objects
        """
        if not klines:
            return []

        # First pass: build MCS name mapping
        self._build_mcs_names(klines)

        # Second pass: decompile all entries in order
        result = []
        for kline in klines:
            entry = self._decompile_kline(kline)
            if entry is not None:
                result.append(entry)

        return result

    def clear(self) -> None:
        """Clear state."""
        self._mcs_names = {}

    def _build_mcs_names(self, klines: list[KLine]) -> None:
        """Build mapping from packed tokens to original MCS names.

        An MCS entry has:
        - S2 significance (canonize level)
        - Multiple nodes that are all packed single-char tokens
        - The signature token equals the OR of all node tokens
        """
        for kline in klines:
            level = self._sig.get_level(kline.signature)
            if level != "S2":
                continue

            nodes = kline.as_node_list()
            if not nodes or len(nodes) < 2:
                continue

            # Check if all nodes are packed single-char tokens
            chars = []
            for node in nodes:
                char = self._try_decode_packed_single_char(node)
                if char is None:
                    break
                chars.append(char)

            if len(chars) != len(nodes):
                continue

            # Key check: MCS signature token == OR of all node tokens
            base_token = self._sig.strip(kline.signature)
            nodes_or = 0
            for node in nodes:
                nodes_or |= node

            if base_token == nodes_or:
                original_name = "".join(chars)
                self._mcs_names[base_token] = original_name

    def _try_decode_packed_single_char(self, node: int) -> str | None:
        """Try to decode a node as a packed single-char token."""
        if (node & PACKED_BIT) == 0:
            return None

        decoded = self.tokenizer.decode([node], pack=None)
        if decoded and len(decoded) == 1 and decoded.isupper():
            return decoded
        return None

    def _is_mcs_entry(self, kline: KLine) -> bool:
        """Check if this KLine is an MCS entry (sig token == OR of node tokens)."""
        nodes = kline.as_node_list()
        if not nodes or len(nodes) < 2:
            return False

        base_token = self._sig.strip(kline.signature)
        nodes_or = 0
        for node in nodes:
            nodes_or |= node

        return base_token == nodes_or

    def _decompile_kline(self, kline: KLine) -> DecompiledEntry | None:
        """Decompile a single KLine to entry. Returns None for MCS entries."""
        sig = kline.signature
        level = self._sig.get_level(sig)
        sig_str = self._decode_sig(sig)
        nodes = kline.as_node_list()

        # MCS entries get special significance
        if level == "S2" and self._is_mcs_entry(kline):
            node_strs = self._decode_nodes(nodes)
            return DecompiledEntry(
                level="MCS",
                sig=sig_str,
                nodes=node_strs,
            )

        if not nodes:
            # Unsigned
            return DecompiledEntry(
                level=level,
                sig=sig_str,
                nodes=None,
            )

        # Construct with nodes
        node_strs = self._decode_nodes(nodes)
        # Single result -> string, multiple -> list
        node_output: list[str] | str = node_strs[0] if len(node_strs) == 1 else node_strs
        return DecompiledEntry(
            level=level,
            sig=sig_str,
            nodes=node_output,
        )

    def _decode_sig(self, sig: int) -> str:
        """Decode signature to string, using MCS name if available."""
        base_token = self._sig.strip(sig)

        if base_token in self._mcs_names:
            return self._mcs_names[base_token]

        if (base_token & PACKED_BIT) == 0:
            return self._decode_node(base_token)

        result = self.tokenizer.decode([base_token], pack=None)
        return result if result else f"<{base_token}>"

    def _decode_node(self, node: int) -> str:
        """Decode a node value to string."""
        if (node & PACKED_BIT) == 0:
            return chr(node >> 1)

        if node in self._mcs_names:
            return self._mcs_names[node]

        result = self.tokenizer.decode([node], pack=None)
        return result if result else f"<{node}>"

    def _decode_nodes(self, nodes: list[int]) -> list[str]:
        """Decode a list of nodes to strings, grouping consecutive literal chars."""
        result: list[str] = []
        literal_chars: list[str] = []

        for node in nodes:
            if (node & PACKED_BIT) == 0:
                literal_chars.append(chr(node >> 1))
            else:
                if literal_chars:
                    result.append("".join(literal_chars))
                    literal_chars = []
                if node in self._mcs_names:
                    result.append(self._mcs_names[node])
                else:
                    decoded = self.tokenizer.decode([node], pack=None)
                    result.append(decoded if decoded else f"<{node}>")

        if literal_chars:
            result.append("".join(literal_chars))

        return result
