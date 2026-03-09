"""Significance calculation for KLine matching."""

from __future__ import annotations

from kalvin.abstract import KLine, KModel, KSignificance, KSig, KNone


class Int64Significance(KSignificance):
    """64-bit integer-based significance implementation.

    Layout: S1(bit56) | S1%(bits57-63) | S2(bits40-55) | S3(bits16-39) | Reserved(bits0-15)
    Higher bits = more significant: S1 > S2 > S3 > S4
    """

    # S1: single bit (bit 56) - prefix match indicator
    _S1 = 1 << 56

    # S1%: 7 bits (bits 57-63) for degree/percentage
    _S1_PCT_SHIFT = 57
    _S1_PCT_MASK = 0x7F << _S1_PCT_SHIFT

    # S2: single bit (bit 40)
    _S2 = 1 << 40

    # S2: 16 bits (bits 40-55)
    _S2_SHIFT = 40
    _S2_MASK = 0xFFFF << _S2_SHIFT
    _S2_S1_PCT_SHIFT = 40   # S1 percentage within S2
    _S2_S2_PCT_SHIFT = 48   # S2 percentage within S2

    # S3: single bit (bit 16)
    _S3 = 1 << 16

    # S3: 24 bits (bits 16-39)
    _S3_SHIFT = 16
    _S3_MASK = 0xFFFFFF << _S3_SHIFT
    _S3_S1_PCT_SHIFT = 16   # S1% for unordered matches (bits 16-23)
    _S3_S2_PCT_SHIFT = 24   # S2% for unordered matches (bits 24-31)
    _S3_GEN_PCT_SHIFT = 32  # Generational S1% (bits 32-39)

    # S4: no significance
    _S4 = 0

    # === Significance level constants ===

    @property
    def S1(self) -> KSig:
        """S1 significance level (highest - prefix match)."""
        return self._S1

    @property
    def S2(self) -> KSig:
        """S2 significance level (partial positional match)."""
        return self._S2

    @property
    def S3(self) -> KSig:
        """S3 significance level (unordered/generational match)."""
        return self._S3

    @property
    def S4(self) -> KSig:
        """S4 significance level (no match)."""
        return self._S4

    # === S1 operations ===

    def has_s1(self, sig: KSig) -> bool:
        """Check if S1 bit is set (prefix match)."""
        return bool(sig & self._S1)

    def get_s1_percentage(self, sig: KSig) -> int:
        """Extract S1 percentage (0-127)."""
        return (sig >> self._S1_PCT_SHIFT) & 0x7F

    def build_s1(self, percentage: int = 100) -> KSig:
        """Build S1 significance with optional percentage."""
        scaled = max(0, min(100, percentage)) * 127 // 100  # Scale to 7 bits
        return self._S1 | (scaled << self._S1_PCT_SHIFT)

    # === S2 operations ===

    def get_s2(self, sig: KSig) -> KSig:
        """Extract full S2 value (0-65535)."""
        return (sig >> self._S2_SHIFT) & 0xFFFF

    def get_s2_s1_percentage(self, sig: KSig) -> int:
        """Extract S2's S1 percentage (0-255)."""
        return (sig >> self._S2_S1_PCT_SHIFT) & 0xFF

    def get_s2_s2_percentage(self, sig: KSig) -> int:
        """Extract S2's S2 percentage (0-255)."""
        return (sig >> self._S2_S2_PCT_SHIFT) & 0xFF

    def build_s2(self, s1_pct: int, s2_pct: int) -> KSig:
        """Build S2 significance."""
        s1_scaled = max(0, min(100, s1_pct)) * 255 // 100
        s2_scaled = max(0, min(100, s2_pct)) * 255 // 100
        return (s1_scaled << self._S2_S1_PCT_SHIFT) | (s2_scaled << self._S2_S2_PCT_SHIFT)

    # === S3 operations ===

    def get_s3(self, sig: KSig) -> KSig:
        """Extract full S3 value (0-16777215)."""
        return (sig >> self._S3_SHIFT) & 0xFFFFFF

    def get_s3_s1_percentage(self, sig: KSig) -> int:
        """Extract S3's S1 percentage for unordered matches (0-255)."""
        return (sig >> self._S3_S1_PCT_SHIFT) & 0xFF

    def get_s3_s2_percentage(self, sig: KSig) -> int:
        """Extract S3's S2 percentage for unordered matches (0-255)."""
        return (sig >> self._S3_S2_PCT_SHIFT) & 0xFF

    def get_s3_gen_percentage(self, sig: KSig) -> int:
        """Extract S3's generational S1 percentage (0-255)."""
        return (sig >> self._S3_GEN_PCT_SHIFT) & 0xFF

    def build_s3(self, s1_pct: int, s2_pct: int, gen_pct: int) -> KSig:
        """Build S3 significance."""
        s1_scaled = max(0, min(100, s1_pct)) * 255 // 100
        s2_scaled = max(0, min(100, s2_pct)) * 255 // 100
        gen_scaled = max(0, min(100, gen_pct)) * 255 // 100
        return (
            (s1_scaled << self._S3_S1_PCT_SHIFT)
            | (s2_scaled << self._S3_S2_PCT_SHIFT)
            | (gen_scaled << self._S3_GEN_PCT_SHIFT)
        )

    # === Significance calculation ===

    def calculate(self, model: "KModel", query: "KLine", target: "KLine") -> KSig:
        """Calculate significance between query and target KLines.

        Significance is comparable as integers - higher = more significant.
        S1 > S2 > S3 > S4.

        Args:
            model: The Model containing the KLines (for descendant lookup)
            query: The query KLine
            target: The target KLine to compare against

        Returns:
            64-bit significance value
        """
        # Handle empty node lists
        if not query.nodes and not target.nodes:
            return self._S1  # Perfect match
        if not query.nodes or not target.nodes:
            return self._S4

        min_len = min(len(query.nodes), len(target.nodes))

        # Count S1 matches: positional equality (up to min length)
        s1_match_positions = set(
            i for i in range(min_len) if query.nodes[i] == target.nodes[i]
        )
        s1_matches = len(s1_match_positions)

        # S1: All prefix nodes match
        if s1_matches == min_len:
            return self._S1  # All matched

        # S1: countersigned
        for kline in model.find_signed_klines(target.signature):
            if kline.signature == query.signature:
                return self._S1  # All matched

        # S2: Partial match (some positional matches exist)
        if s1_matches > 0:
            s1_pct = (s1_matches * 100) // min_len

            # S2 matches: nodes at different positions
            target_set = set(target.nodes)
            s2_matches = 0
            for i, node in enumerate(query.nodes):
                if i in s1_match_positions:
                    continue  # Already counted as S1
                if node in target_set:
                    s2_matches += 1

            s2_pct = (s2_matches * 100) // len(query.nodes) if query.nodes else 0
            return self.build_s2(s1_pct, s2_pct)

        # S3: No positional matches, check unordered and generational
        target_set = set(target.nodes)
        query_set = set(query.nodes)

        # S3-Unordered S1: query nodes that exist in target (any position)
        unordered_s1_matches = query_set & target_set
        s3_s1_pct = (
            (len(unordered_s1_matches) * 100) // len(query_set) if query_set else 0
        )

        # S3-Unordered S2: query nodes whose children match target nodes
        s3_s2_matches = 0
        for node in query.nodes:
            if node in target_set:
                continue  # Already S1 match
            # Check if node's children intersect with target
            node_kline = model.find_kline(node)
            if node_kline is not KNone:
                node_children = set(node_kline.nodes)
                if node_children & target_set:
                    s3_s2_matches += 1

        s3_s2_pct = (
            (s3_s2_matches * 100) // len(query.nodes) if query.nodes else 0
        )

        # S3-Generational: query nodes whose descendants (at any depth) match target nodes
        gen_matches = 0
        for node in query.nodes:
            if node in target_set:
                continue  # Already S1 match
            # Collect all descendants of this node
            descendants = model.get_all_descendants(node)
            if descendants & target_set:
                gen_matches += 1

        gen_pct = (gen_matches * 100) // len(query.nodes) if query.nodes else 0

        if s3_s1_pct > 0 or s3_s2_pct > 0 or gen_pct > 0:
            return self.build_s3(s3_s1_pct, s3_s2_pct, gen_pct)

        # S4: No match
        return self._S4


# Module-level constants (class attributes for convenience)
S1 = KSig(Int64Significance._S1)
S2 = KSig(Int64Significance._S2)
S3 = KSig(Int64Significance._S3)
S4 = KSig(Int64Significance._S4)
