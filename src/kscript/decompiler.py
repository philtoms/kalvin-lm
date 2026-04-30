"""Decompiler for KLines back to KScript source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kalvin.mod_tokenizer import ModTokenizer, Mod32Tokenizer
from kalvin.signature import make_signature
from kalvin.agent import D_MAX
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

        Detects two MCS patterns:
        1. Legacy: single entry with multiple packed single-char nodes, sig == OR of nodes
        2. Per-char: consecutive entries with same sig, each having a single
           packed single-char node, where OR of all char tokens == sig
        """
        # Pattern 1: Legacy multi-node MCS entry
        for kline in klines:
            nodes = kline.as_node_list()
            if not nodes or len(nodes) < 2:
                continue

            chars = []
            for node in nodes:
                char = self._try_decode_packed_single_char(node)
                if char is None:
                    break
                chars.append(char)

            if len(chars) != len(nodes):
                continue

            base_token = kline.signature
            nodes_or = 0
            for node in nodes:
                nodes_or |= node

            if base_token == nodes_or:
                original_name = "".join(chars)
                self._mcs_names[base_token] = original_name

        # Pattern 2: Per-char MCS entries (consecutive same-sig single-char nodes)
        i = 0
        while i < len(klines):
            kline = klines[i]
            sig = kline.signature
            if sig in self._mcs_names:
                i += 1
                continue

            nodes = kline.as_node_list()
            if len(nodes) != 1:
                i += 1
                continue

            char = self._try_decode_packed_single_char(nodes[0])
            if char is None:
                i += 1
                continue

            # Collect consecutive same-sig, single-char packed entries
            chars: list[str] = [char]
            nodes_or = nodes[0]
            j = i + 1
            while j < len(klines):
                next_kline = klines[j]
                if next_kline.signature != sig:
                    break
                next_nodes = next_kline.as_node_list()
                if len(next_nodes) != 1:
                    break
                next_char = self._try_decode_packed_single_char(next_nodes[0])
                if next_char is None:
                    break
                new_or = nodes_or | next_nodes[0]
                if new_or != (new_or & sig):
                    # New bits not in sig - stop collecting
                    break
                chars.append(next_char)
                nodes_or = new_or
                j += 1

            if len(chars) >= 2 and nodes_or == sig:
                self._mcs_names[sig] = "".join(chars)
                i = j
                continue

            i += 1

    def _try_decode_packed_single_char(self, node: int) -> str | None:
        """Try to decode a node as a packed single-char token."""
        if self.tokenizer.is_literal(node):
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

        nodes_sig = make_signature(nodes, self.tokenizer.is_literal)
        return kline.signature == nodes_sig

    def _decompile_kline(self, kline: KLine) -> DecompiledEntry | None:
        """Decompile a single KLine to entry. Returns None for MCS entries."""
        sig = kline.signature
        sig_str = self._decode_sig(sig)
        nodes = kline.as_node_list()

        # MCS entries get special significance
        if self._is_mcs_entry(kline):
            node_strs = self._decode_nodes(nodes)
            return DecompiledEntry(
                level="S2",
                sig=sig_str,
                nodes=node_strs,
            )

        # Determine level from node structure
        level = self._infer_level(kline)

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

    def _infer_level(self, kline: KLine) -> str:
        """Infer significance level from node structure.

        S1: countersign/undersign (single non-overlapping node, heuristic)
        S2: (signature & node values) != 0 (canonize)
        S3: (signature & node values) == 0 (connotate)
        S4: no nodes (unsigned)

        Note: With singleton unwrapping, int nodes can be any op type.
        We use bit overlap as the primary heuristic for int nodes.
        """
        nodes = kline.nodes
        if nodes is None:
            return "S4"
        if isinstance(nodes, int):
            # Single node: use bit overlap to distinguish canonize (S2) from others
            combined = kline.signature & nodes
            return "S2" if combined != 0 else "S3"

        nodes_sig = make_signature(nodes, self.tokenizer.is_literal)
        combined = kline.signature & nodes_sig
        return "S2" if combined != 0 else "S3"

    def _decode_sig(self, sig: int) -> str:
        """Decode signature to string, using MCS name if available."""
        if sig in self._mcs_names:
            return self._mcs_names[sig]

        if self.tokenizer.is_literal(sig):
            return self._decode_node(sig)

        result = self.tokenizer.decode([sig], pack=None)
        return result if result else f"<{sig}>"

    def _decode_node(self, node: int) -> str:
        """Decode a node value to string."""
        if self.tokenizer.is_literal(node):
            return self.tokenizer.decode([node])

        if node in self._mcs_names:
            return self._mcs_names[node]

        result = self.tokenizer.decode([node], pack=None)
        return result if result else f"<{node}>"

    def _decode_nodes(self, nodes: list[int]) -> list[str]:
        """Decode a list of nodes to strings, grouping consecutive literal chars."""
        result: list[str] = []
        literal_ids: list[int] = []

        for node in nodes:
            if self.tokenizer.is_literal(node):
                literal_ids.append(node)
            else:
                if literal_ids:
                    result.append(self.tokenizer.decode(literal_ids))
                    literal_ids = []
                if node in self._mcs_names:
                    result.append(self._mcs_names[node])
                else:
                    decoded = self.tokenizer.decode([node], pack=None)
                    result.append(decoded if decoded else f"<{node}>")

        if literal_ids:
            result.append(self.tokenizer.decode(literal_ids))

        return result
