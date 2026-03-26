"""Decompiler for KLines back to KScript source."""

from kalvin.abstract import KLine
from kalvin.mod_tokenizer import ModTokenizer
from kalvin.significance import Int32Significance


class DecompilerError(Exception):
    """Exception raised when decompilation encounters an error."""
    pass


class Decompiler:
    """Decompiles ordered KLines with significance bits to KScript source.

    The first KLine is the primary statement. All subsequent KLines
    exist to satisfy it. Significance bits determine construct types:
        - S1 (bit 63): countersign (==)
        - S2 (bits 55-62): canonize (=>)
        - S3 (bits 32-54): connotate (>)
        - S4 (no bits): undersign (=)
    """

    def __init__(self, tokenizer: ModTokenizer):
        self.tokenizer = tokenizer
        self._sig = Int32Significance()
        self._by_sig: dict[int, KLine] = {}  # Full sig -> KLine
        self._by_token: dict[int, list[int]] = {}  # Base token -> list of full sigs
        self._processed: set[int] = set()
        self._lines: list[str] = []
        self._all_sigs: set[int] = set()  # All signatures in input

    def decompile(self, klines: list[KLine]) -> str:
        """Convert ordered KLines to KScript source.

        Args:
            klines: Ordered list of KLines (first is primary statement)

        Returns:
            KScript source code as string
        """
        if not klines:
            return ""

        # Reset state
        self._by_sig = {}
        self._by_token = {}
        self._processed = set()
        self._lines = []
        self._all_sigs = set()

        # Build lookup and track all signatures
        for kline in klines:
            self._by_sig[kline.signature] = kline
            self._all_sigs.add(kline.signature)
            # Index by base token - track ALL signatures for this token
            base_token = self._sig.strip_significance(kline.signature)
            if base_token not in self._by_token:
                self._by_token[base_token] = []
            self._by_token[base_token].append(kline.signature)

        # Process primary statement
        primary = klines[0]
        self._emit(primary.signature, indent=0)

        # Check for orphans (unreachable from primary)
        for sig in self._all_sigs:
            if sig not in self._processed:
                sig_str = self._decode_sig(sig)
                self._lines.append(f"!!! ORPHAN: {sig_str}")

        return "\n".join(self._lines)

    def _get_significance_level(self, signature: int) -> str:
        """Detect significance level from signature bits.

        Returns: "S1", "S2", "S3", or "S4"
        """
        return self._sig.get_level(signature)

    def _get_construct_op(self, level: str) -> str:
        """Map significance level to construct operator."""
        return self._sig.get_construct_op(level)

    def _decode_sig(self, sig: int) -> str:
        """Decode signature to string, stripping significance bits."""
        try:
            token = self._sig.strip_significance(sig)
            result = self.tokenizer.decode([token], pack=None)
            if not result:
                return f"!!! BROKEN: expected {sig}"
            return result
        except Exception:
            return f"!!! BROKEN: expected {sig}"

    def _decode_node(self, node: int) -> str:
        """Decode node to string (no significance bits to strip)."""
        try:
            result = self.tokenizer.decode([node], pack=None)
            if not result:
                return f"!!! BROKEN: expected {node}"
            return result
        except Exception:
            return f"!!! BROKEN: expected {node}"

    def _get_all_sigs_for_token(self, token: int) -> list[int]:
        """Get all signatures for a base token."""
        return self._by_token.get(token, [])

    def _emit(self, sig: int, indent: int) -> None:
        """Recursively emit script for signature."""
        if sig in self._processed:
            return
        self._processed.add(sig)

        entry = self._by_sig.get(sig)
        sig_str = self._decode_sig(sig)
        prefix = "  " * indent

        if entry is None:
            # Missing entry = identity
            self._lines.append(f"{prefix}{sig_str}")
            return

        level = self._get_significance_level(sig)
        op = self._get_construct_op(level)
        nodes = entry.as_node_list()

        if level == "S4":
            # Undersign - emit with = operator
            if nodes:
                node_str = self._decode_node(nodes[0])
                self._lines.append(f"{prefix}{sig_str} {op} {node_str}")
            else:
                # Identity (no nodes)
                self._lines.append(f"{prefix}{sig_str}")
        elif level == "S1":
            # Countersign - emit once, then process partner's OTHER constructs
            if nodes:
                node = nodes[0]
                node_str = self._decode_node(node)
                self._lines.append(f"{prefix}{sig_str} {op} {node_str}")
                # Mark partner's base token as processed (for countersign lookup)
                self._processed.add(node)
                # Find partner's countersign signature and mark it
                for partner_sig in self._get_all_sigs_for_token(node):
                    if self._get_significance_level(partner_sig) == "S1":
                        self._processed.add(partner_sig)
                        break
                # Now emit ALL OTHER constructs for partner (non-countersign)
                for partner_sig in self._get_all_sigs_for_token(node):
                    if partner_sig not in self._processed:
                        self._emit(partner_sig, indent)
        elif level == "S2":
            # Canonize - subscripts for multi-node
            if len(nodes) > 1:
                self._lines.append(f"{prefix}{sig_str} {op}")
                for node in nodes:
                    # Process all signatures for this node
                    self._emit_all_for_token(node, indent + 1)
            elif nodes:
                # Single node - inline
                node_str = self._decode_node(nodes[0])
                self._lines.append(f"{prefix}{sig_str} {op} {node_str}")
        elif level == "S3":
            # Connotate - inline
            if nodes:
                node_str = self._decode_node(nodes[0])
                self._lines.append(f"{prefix}{sig_str} {op} {node_str}")

    def _emit_all_for_token(self, base_token: int, indent: int) -> None:
        """Emit all signatures for a base token."""
        sigs = self._get_all_sigs_for_token(base_token)
        if not sigs:
            # No signatures - emit as identity
            token_str = self._decode_node(base_token)
            prefix = "  " * indent
            self._lines.append(f"{prefix}{token_str}")
            return

        for sig in sigs:
            self._emit(sig, indent)
