"""Significance calculation for KLine matching."""

from __future__ import annotations

from abc import abstractmethod

from kalvin.abstract import KLine, KFrame, KSignificance, KSig, KNone, KSignificance


class IntSignificance(KSignificance):
    """Abstract base for integer-based significance implementations.

    Provides the common S1-S4 level interface and calculate() method shared
    by all integer significance strategies. Higher = more significant: S1 > S2 > S3 > S4.
    """
    _S1: int
    _S1_PCT_SHIFT = 57
    _S1_PCT_MASK = 0x7F

    _S2: int
    _S2_MASK: int
    _S2_SHIFT: int
    _S2_MASK: int
    _S2_PCT_MASK: int
    _S2_S1_PCT_SHIFT: int
    _S2_S2_PCT_SHIFT: int
    _S2_RANGE: int

    _S3: int
    _S3_SHIFT: int
    _S3_MASK: int
    _S3_PCT_MASK: int
    _S3_S1_PCT_SHIFT: int
    _S3_S2_PCT_SHIFT: int
    _S3_GEN_PCT_SHIFT: int
    _S3_GEN_PCT_MASK: int
    _S3_RANGE: int

    _S4: int = 0

    _TOKEN_MASK: int

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
        return 0

    # === Level detection ===

    def has_s1(self, sig: KSig) -> bool:
        """Check if S1 bit is set."""
        return bool(sig & self._S1)

    def has_s2(self, sig: KSig) -> bool:
        """Check if S2 bit is set."""
        return bool(sig & self.S2)

    def has_s3(self, sig: KSig) -> bool:
        """Check if S3 bit is set."""
        return bool(sig & self.S3)

    def has_s4(self, sig: KSig) -> bool:
        """Check all bits clear."""
        return sig == self.S4

    # === Level builders (subclass overrides for richer encoding) ===

    @abstractmethod
    def build_s2(self, s1_pct: int, s2_pct: int) -> KSig:
        """Build S2 significance from match percentages."""
        ...

    @abstractmethod
    def build_s3(self, s1_pct: int, s2_pct: int, gen_pct: int) -> KSig:
        """Build S3 significance from match percentages."""
        ...

    # === S1 operations ===

    def get_s1_percentage(self, sig: KSig) -> int:
        """Extract S1 percentage."""
        return (sig >> self._S1_PCT_SHIFT) & self._S1_PCT_MASK

    # === S2 operations ===

    def get_s2(self, sig: KSig) -> KSig:
        """Extract full S2 value."""
        return (sig >> self._S2_SHIFT) & self._S2_MASK

    def get_s2_s1_percentage(self, sig: KSig) -> int:
        """Extract S2's S1 percentage."""
        return (sig >> self._S2_S1_PCT_SHIFT) & self._S2_PCT_MASK

    def get_s2_s2_percentage(self, sig: KSig) -> int:
        """Extract S2's S2 percentage."""
        return (sig >> self._S2_S2_PCT_SHIFT) & self._S2_PCT_MASK

    # === S3 operations ===

    def get_s3(self, sig: KSig) -> KSig:
        """Extract full S3 value."""
        return (sig >> self._S3_SHIFT) & self._S3_MASK

    def get_s3_s1_percentage(self, sig: KSig) -> int:
        """Extract S3's S1 percentage for unordered matches."""
        return (sig >> self._S3_S1_PCT_SHIFT) & self._S3_PCT_MASK

    def get_s3_s2_percentage(self, sig: KSig) -> int:
        """Extract S3's S2 percentage for unordered matches."""
        return (sig >> self._S3_S2_PCT_SHIFT) & self._S3_PCT_MASK

    def get_s3_gen_percentage(self, sig: KSig) -> int:
        """Extract S3's generational S1 percentage."""
        return (sig >> self._S3_GEN_PCT_SHIFT) & self._S3_GEN_PCT_MASK

    # === S4 operations ===

    def get_s4(self, sig: KSig) -> KSig:
        """Extract full S4 value."""
        return sig

    # === Significance calculation ===

    def is_significant(self, sig: KSig) -> bool:
        """Is S1 or S4 significant"""
        return self.has_s1(sig) or self.has_s4(sig)

    def is_rational(self, sig: KSig) -> bool:
        """Is S2 or S3 significant"""
        return not self.is_significant(sig)

    def calculate(self, frame: "KFrame", query: "KLine", target: "KLine") -> KSig:
        """Calculate significance between query and target KLines.

        Significance is comparable as integers - higher = more significant.
        S1 > S2 > S3 > S4.

        Args:
            frame: The Frame containing the KLines (for descendant lookup)
            query: The query KLine
            target: The target KLine to compare against

        Returns:
            Significance value
        """
        # Get nodes as lists for comparison
        query_nodes = query.as_node_list()
        target_nodes = target.as_node_list()

        # Handle empty node lists
        if not query_nodes or not target_nodes:
            return self.S4

        min_len = min(len(query_nodes), len(target_nodes))

        # Count S1 matches: positional equality (up to min length)
        s1_match_positions = set(
            i for i in range(min_len) if query_nodes[i] == target_nodes[i]
        )
        s1_matches = len(s1_match_positions)

        # S1: All prefix nodes match
        if s1_matches == min_len:
            return self.S1  # All matched

        # S1: countersigned
        for kline in frame.find_signed_klines(target.signature):
            if kline.signature == query.signature:
                return self.S1  # All matched

        # S2: Partial match (some positional matches exist)
        if s1_matches > 0:
            s1_pct = (s1_matches * 100) // min_len

            # S2 matches: nodes at different positions
            target_set = set(target_nodes)
            s2_matches = 0
            for i, node in enumerate(query_nodes):
                if i in s1_match_positions:
                    continue  # Already counted as S1
                if node in target_set:
                    s2_matches += 1

            s2_pct = (s2_matches * 100) // len(query_nodes) if query_nodes else 0
            return self.build_s2(s1_pct, s2_pct)

        # S3: No positional matches, check unordered and generational
        target_set = set(target_nodes)
        query_set = set(query_nodes)

        # S3-Unordered S1: query nodes that exist in target (any position)
        unordered_s1_matches = query_set & target_set
        s3_s1_pct = (
            (len(unordered_s1_matches) * 100) // len(query_set) if query_set else 0
        )

        # S3-Unordered S2: query nodes whose children match target nodes
        s3_s2_matches = 0
        for node in query_nodes:
            if node in target_set:
                continue  # Already S1 match
            # Check if node's children intersect with target
            node_kline = frame.find_kline(node)
            if node_kline is not KNone:
                node_children = set(node_kline.as_node_list())
                if node_children & target_set:
                    s3_s2_matches += 1

        s3_s2_pct = (
            (s3_s2_matches * 100) // len(query_nodes) if query_nodes else 0
        )

        # S3-Generational: query nodes whose descendants (at any depth) match target nodes
        gen_matches = 0
        for node in query_nodes:
            if node in target_set:
                continue  # Already S1 match
            # Collect all descendants of this node
            descendants = frame.get_all_descendants(node)
            if descendants & target_set:
                gen_matches += 1

        gen_pct = (gen_matches * 100) // len(query_nodes) if query_nodes else 0

        if s3_s1_pct > 0 or s3_s2_pct > 0 or gen_pct > 0:
            return self.build_s3(s3_s1_pct, s3_s2_pct, gen_pct)

        # S4: No match
        return self.S4

    # === helper functions ===

    def get_level(self, sig: int) -> str:
        """Detect significance level from signature bits.

        Hierarchical detection: S1 > S2 > S3 > S4
        """
        if sig & self._S1:           # bit 63 set
            return "S1"
        elif sig & self._S2_RANGE:   # any bit 55-62 set
            return "S2"
        elif sig & self._S3_RANGE:   # any bit 32-54 set
            return "S3"
        else:
            return "S4"             # all significance bits clear

    def strip_significance(self, sig: int) -> int:
        """Strip significance bits, returning only token bits."""
        return sig & self._TOKEN_MASK


class Int64Significance(IntSignificance):
    """64-bit integer-based significance implementation.

    Layout: S1(bit56) | S1%(bits57-63) | S2(bits40-55) | S3(bits16-39) | Reserved(bits0-15)
    Higher bits = more significant: S1 > S2 > S3 > S4
    """

    # S1: single bit (bit 56) - prefix match indicator
    _S1 = 1 << 56

    # S1%: 7 bits (bits 57-63) for degree/percentage
    _S1_PCT_SHIFT = 57
    _S1_PCT_MASK = 0x7F

    # S2: single bit (bit 40)
    _S2 = 1 << 40

    # S2: 16 bits (bits 40-55)
    _S2_SHIFT = 40
    _S2_MASK = 0xFFFF
    _S2_PCT_MASK = 0xFF
    _S2_S1_PCT_SHIFT = 40   # S1 percentage within S2
    _S2_S2_PCT_SHIFT = 48   # S2 percentage within S2
    _S2_RANGE = 0xFFFF << 40 

    # S3: single bit (bit 16)
    _S3 = 1 << 16

    # S3: 24 bits (bits 16-39)
    _S3_SHIFT = 16
    _S3_MASK = 0xFFFFFF
    _S3_PCT_MASK = 0xFF
    _S3_S1_PCT_SHIFT = 16   # S1% for unordered matches (bits 16-23)
    _S3_S2_PCT_SHIFT = 24   # S2% for unordered matches (bits 24-31)
    _S3_GEN_PCT_SHIFT = 32  # Generational S1% (bits 32-39)
    _S3_GEN_PCT_MASK = 0xFF
    _S3_RANGE = 0xFFFFFF << 16 

    _TOKEN_MASK = (1 << 64) - 1

    # === build operations ===

    def build_s1(self, percentage: int = 100) -> KSig:
        """Build S1 significance with optional percentage."""
        scaled = max(0, min(100, percentage)) * 127 // 100  # Scale to 7 bits
        return self._S1 | (scaled << self._S1_PCT_SHIFT)

    def build_s2(self, s1_pct: int, s2_pct: int) -> KSig:
        """Build S2 significance with percentage encoding."""
        s1_scaled = max(0, min(100, s1_pct)) * 255 // 100
        s2_scaled = max(0, min(100, s2_pct)) * 255 // 100
        return (s1_scaled << self._S2_S1_PCT_SHIFT) | (s2_scaled << self._S2_S2_PCT_SHIFT)

    def build_s3(self, s1_pct: int, s2_pct: int, gen_pct: int) -> KSig:
        """Build S3 significance with percentage encoding."""
        s1_scaled = max(0, min(100, s1_pct)) * 255 // 100
        s2_scaled = max(0, min(100, s2_pct)) * 255 // 100
        gen_scaled = max(0, min(100, gen_pct)) * 255 // 100
        return (
            (s1_scaled << self._S3_S1_PCT_SHIFT)
            | (s2_scaled << self._S3_S2_PCT_SHIFT)
            | (gen_scaled << self._S3_GEN_PCT_SHIFT)
        )


class Int32Significance(IntSignificance):
    """32-bit significance implementation for KScript constructs.

    Layout (32-bit, before shift):
    - Bit 31: S1 indicator (countersign)
    - Bits 23-30: S2 range (8 bits, canonize)
    - Bits 0-22: S3 range (23 bits, connotate)
    - All clear: S4 (identity)

    Stored in bits 32-63 of 64-bit signature (shifted left 32).
    Token space remains in bits 0-31 (unchanged).
    """

    # S1: single bit (bit 63 after shift) - countersign indicator
    _S1 = 1 << 63
    _S1_PCT_MASK = 1
    _S1_PCT_SHIFT = 63

    # S2: 8 bits (bits 55-62 after shift) - canonize
    _S2 = 1 << 55
    _S2_SHIFT = 55
    _S2_MASK = 0xFF
    _S2_PCT_MASK = 0xF
    _S2_S1_PCT_SHIFT = 55   # S1 percentage within S2 (bits 55-62)
    _S2_S2_PCT_SHIFT = 59   # S2 percentage within S2 (bits 59-62) - 4 bits each
    _S2_RANGE = 0xFF << 55  # bits 55-62 (8 bits for S2)


    # S3: 23 bits (bits 32-54 after shift) - connotate
    _S3 = 1 << 32
    _S3_SHIFT = 32
    _S3_MASK = 0x7F_FFFF
    _S3_PCT_MASK = 0xF
    _S3_S1_PCT_SHIFT = 32   # S1% for unordered matches (bits 32-39, 8 bits)
    _S3_S2_PCT_SHIFT = 40   # S2% for unordered matches (bits 40-47, 8 bits)
    _S3_GEN_PCT_SHIFT = 48  # Generational S1% (bits 48-54, 7 bits)
    _S3_GEN_PCT_MASK = 0x7F
    _S3_RANGE = 0x7F_FFFF << 32  # bits 32-54 (23 bits for S3)

    _TOKEN_MASK = (1 << 32) - 1

    # === build operations ===

    def build_s1(self, s1_pct: int, s2_pct: int) -> KSig:
        """Build S1 significance (no percentage encoding)."""
        return self._S1

    def build_s2(self, s1_pct: int, s2_pct: int) -> KSig:
        """Build S2 significance with percentage encoding."""
        s1_scaled = max(0, min(100, s1_pct)) * 15 // 100  # 4 bits (0-15)
        s2_scaled = max(0, min(100, s2_pct)) * 15 // 100  # 4 bits (0-15)
        return (s1_scaled << self._S2_S1_PCT_SHIFT) | (s2_scaled << self._S2_S2_PCT_SHIFT)

    def build_s3(self, s1_pct: int, s2_pct: int, gen_pct: int) -> KSig:
        """Build S3 significance with percentage encoding."""
        s1_scaled = max(0, min(100, s1_pct)) * 255 // 100  # 8 bits (0-255)
        s2_scaled = max(0, min(100, s2_pct)) * 255 // 100  # 8 bits (0-255)
        gen_scaled = max(0, min(100, gen_pct)) * 127 // 100  # 7 bits (0-127)
        return (
            (s1_scaled << self._S3_S1_PCT_SHIFT)
            | (s2_scaled << self._S3_S2_PCT_SHIFT)
            | (gen_scaled << self._S3_GEN_PCT_SHIFT)
        )
