"""Significance calculation for KLine matching.

Subtractive model: significance starts from an ideal (all bits set for the level's
domain) and is reduced by clearing bits based on match imperfection.
Higher value = more significant.  S1 > S2 > S3 > S4.

32-bit significance (shifted to top 32 bits of 64-bit int):
  S1 range: bits 63-61 (3 bits)   → ideal: all 32 sig bits set
  S2 range: bits 60-52 (9 bits)   → ideal: bits 60-32 set
  S3 range: bits 51-32 (20 bits)  → ideal: bits 51-32 set
  S4: 0

64-bit significance (proportionately scaled to fill 64 bits):
  S1 range: bits 63-58 (6 bits)   → ideal: all 64 bits set
  S2 range: bits 57-40 (18 bits)  → ideal: bits 57-0 set
  S3 range: bits 39-0  (40 bits)  → ideal: bits 39-0 set
  S4: 0
"""

from __future__ import annotations

from abc import abstractmethod

from kalvin.abstract import KSignificance, KSig, KNodes


class IntSignificance(KSignificance):
    """Abstract base for integer-based significance implementations.

    Provides the common S1-S4 level interface and calculate() method shared
    by all integer significance strategies.

    Subtractive model: each level has an *ideal* value (all bits in its domain
    set).  Build functions start from that ideal and clear bits proportionally
    to (100 − percentage), so higher match quality → more bits remain → higher
    numeric value.  The ordering S1 > S2 > S3 > S4 is preserved naturally.
    """

    # --- Range masks (set by concrete classes) ---
    _S1_RANGE: int
    _S2_RANGE: int
    _S3_RANGE: int

    # --- Ideal values (set by concrete classes) ---
    _IDEAL_S1: int
    _IDEAL_S2: int
    _IDEAL_S3: int

    # --- Token mask ---
    _TOKEN_MASK: int

    # --- S1 percentage sub-field ---
    _S1_PCT_SHIFT: int
    _S1_PCT_WIDTH: int   # number of quality bits for S1

    # --- S2 sub-fields ---
    _S2_INDICATOR: int   # single bit, always set at S2 level
    _S2_SHIFT: int
    _S2_MASK: int
    _S2_S1_PCT_SHIFT: int
    _S2_S1_PCT_WIDTH: int
    _S2_S2_PCT_SHIFT: int
    _S2_S2_PCT_WIDTH: int

    # --- S3 sub-fields ---
    _S3_INDICATOR: int   # single bit, always set at S3 level
    _S3_SHIFT: int
    _S3_MASK: int
    _S3_S1_PCT_SHIFT: int
    _S3_S1_PCT_WIDTH: int
    _S3_S2_PCT_SHIFT: int
    _S3_S2_PCT_WIDTH: int
    _S3_GEN_PCT_SHIFT: int
    _S3_GEN_PCT_WIDTH: int

    # ==================================================================
    # Properties – return ideal values
    # ==================================================================

    @property
    def S1(self) -> KSig:
        """S1 significance level (highest – ideal: all significance bits set)."""
        return self._IDEAL_S1

    @property
    def S2(self) -> KSig:
        """S2 significance level (ideal: S2 + S3 ranges all set)."""
        return self._IDEAL_S2

    @property
    def S3(self) -> KSig:
        """S3 significance level (ideal: S3 range all set)."""
        return self._IDEAL_S3

    @property
    def S4(self) -> KSig:
        """S4 significance level (no match – all bits clear)."""
        return 0

    # ==================================================================
    # Level detection
    # ==================================================================

    def has_s1(self, sig: KSig) -> bool:
        """Check if any S1-range bit is set."""
        return bool(sig & self._S1_RANGE)

    def has_s2(self, sig: KSig) -> bool:
        """Check if any S2-range bit is set."""
        return bool(sig & self._S2_RANGE)

    def has_s3(self, sig: KSig) -> bool:
        """Check if any S3-range bit is set."""
        return bool(sig & self._S3_RANGE)

    def has_s4(self, sig: KSig) -> bool:
        """Check all significance bits are clear."""
        return sig == self.S4

    # ==================================================================
    # Level builders (concrete classes override with richer encoding)
    # ==================================================================

    @abstractmethod
    def build_s2(self, s1_pct: int, s2_pct: int) -> KSig:
        """Build S2 significance by subtracting from ideal S2.

        Higher percentages → less subtraction → more bits set.
        """
        ...

    @abstractmethod
    def build_s3(self, s1_pct: int, s2_pct: int, gen_pct: int) -> KSig:
        """Build S3 significance by subtracting from ideal S3."""
        ...

    # ==================================================================
    # S1 operations
    # ==================================================================

    def get_s1_percentage(self, sig: KSig) -> int:
        """Extract S1 percentage (0–100)."""
        width = self._S1_PCT_WIDTH
        raw = (sig >> self._S1_PCT_SHIFT) & ((1 << width) - 1)
        return self._raw_to_pct(raw, width)

    # ==================================================================
    # S2 operations
    # ==================================================================

    def get_s2(self, sig: KSig) -> KSig:
        """Extract full S2-level value (S2 + S3 ranges)."""
        return (sig >> self._S2_SHIFT) & self._S2_MASK

    def get_s2_s1_percentage(self, sig: KSig) -> int:
        """Extract S2's S1 positional-match percentage (0–100)."""
        width = self._S2_S1_PCT_WIDTH
        raw = (sig >> self._S2_S1_PCT_SHIFT) & ((1 << width) - 1)
        return self._raw_to_pct(raw, width)

    def get_s2_s2_percentage(self, sig: KSig) -> int:
        """Extract S2's non-positional-match percentage (0–100)."""
        width = self._S2_S2_PCT_WIDTH
        raw = (sig >> self._S2_S2_PCT_SHIFT) & ((1 << width) - 1)
        return self._raw_to_pct(raw, width)

    # ==================================================================
    # S3 operations
    # ==================================================================

    def get_s3(self, sig: KSig) -> KSig:
        """Extract full S3-level value (S3 range)."""
        return (sig >> self._S3_SHIFT) & self._S3_MASK

    def get_s3_s1_percentage(self, sig: KSig) -> int:
        """Extract S3's unordered S1 percentage (0–100)."""
        width = self._S3_S1_PCT_WIDTH
        raw = (sig >> self._S3_S1_PCT_SHIFT) & ((1 << width) - 1)
        return self._raw_to_pct(raw, width)

    def get_s3_s2_percentage(self, sig: KSig) -> int:
        """Extract S3's unordered S2 percentage (0–100)."""
        width = self._S3_S2_PCT_WIDTH
        raw = (sig >> self._S3_S2_PCT_SHIFT) & ((1 << width) - 1)
        return self._raw_to_pct(raw, width)

    def get_s3_gen_percentage(self, sig: KSig) -> int:
        """Extract S3's generational percentage (0–100)."""
        width = self._S3_GEN_PCT_WIDTH
        raw = (sig >> self._S3_GEN_PCT_SHIFT) & ((1 << width) - 1)
        return self._raw_to_pct(raw, width)

    # ==================================================================
    # S4 operations
    # ==================================================================

    def get_s4(self, sig: KSig) -> KSig:
        """Extract full S4 value."""
        return sig

    # ==================================================================
    # Helper functions
    # ==================================================================

    @staticmethod
    def _pct_to_raw(pct: int, width: int) -> int:
        """Scale percentage (0–100) to a raw value in [0, 2^width − 1].

        At 100 % the maximum value (all bits set) is returned, representing
        the ideal – nothing subtracted.  Lower percentages clear more bits.
        """
        pct = max(0, min(100, pct))
        max_val = (1 << width) - 1
        return pct * max_val // 100

    @staticmethod
    def _raw_to_pct(raw: int, width: int) -> int:
        """Convert a raw sub-field value back to percentage (0–100)."""
        max_val = (1 << width) - 1
        return round(raw * 100 / max_val) if max_val > 0 else 0

    def get_level(self, sig: KSig) -> str:
        """Detect significance level from signature bits.

        Hierarchical detection: S1 > S2 > S3 > S4.
        """
        if sig & self._S1_RANGE:
            return "S1"
        elif sig & self._S2_RANGE:
            return "S2"
        elif sig & self._S3_RANGE:
            return "S3"
        else:
            return "S4"

    def set_level(self, sig: KSig, level: int) -> KSig:
        """Set significance level (OR ideal level bits into *sig*)."""
        return sig | level

    def strip(self, sig: KSig) -> KSig:
        """Strip significance bits, returning only token bits."""
        return sig & self._TOKEN_MASK

    def is_identity(self, sig1: KSig | KNodes, sig2: KSig | KNodes) -> bool:
        """Test if two signatures have equal token bits."""
        if not isinstance(sig1, KSig) or not isinstance(sig2, KSig):
            return False
        return self.strip(sig1) == self.strip(sig2)

    def is_signed(self, sig) -> bool:
        """Test signature is KSig (int)."""
        return isinstance(sig, KSig)

    def is_unsigned(self, sig) -> bool:
        """Test signature is unsigned (None)."""
        return sig is None


# ======================================================================
# Int64Significance
# ======================================================================

class Int64Significance(IntSignificance):
    """64-bit integer-based significance implementation.

    Bit layout (all 64 bits are significance):
        S1 range: bits 63–58  (6 bits)
        S2 range: bits 57–40  (18 bits)
        S3 range: bits 39–0   (40 bits)

    Ideals:
        S1 = 0xFFFF_FFFF_FFFF_FFFF  (all 64 bits)
        S2 = 0x03FF_FFFF_FFFF_FFFF  (bits 57–0)
        S3 = 0x0000_FFFF_FFFF_FFFF  (bits 39–0)
        S4 = 0x0000_0000_0000_0000
    """

    # --- Ranges ---
    _S1_RANGE = ((1 << 6) - 1) << 58          # bits 63–58
    _S2_RANGE = ((1 << 18) - 1) << 40         # bits 57–40
    _S3_RANGE = (1 << 40) - 1                 # bits 39–0

    # --- Ideals ---
    _IDEAL_S1 = (1 << 64) - 1                 # all bits
    _IDEAL_S2 = (1 << 58) - 1                 # bits 57–0
    _IDEAL_S3 = (1 << 40) - 1                 # bits 39–0

    # --- Token mask (64-bit: no stripping) ---
    _TOKEN_MASK = (1 << 64) - 1

    # --- S1 percentage ---
    _S1_PCT_SHIFT = 0
    _S1_PCT_WIDTH = 58                        # bits 57–0

    # --- S2 indicator ---
    _S2_INDICATOR = 1 << 57                   # always set at S2 level

    # --- S2 extraction ---
    _S2_SHIFT = 0
    _S2_MASK = (1 << 58) - 1                  # bits 57–0

    # --- S2 s1-pct sub-field (positional match quality) ---
    _S2_S1_PCT_SHIFT = 40
    _S2_S1_PCT_WIDTH = 17                     # bits 56–40

    # --- S2 s2-pct sub-field (non-positional match quality) ---
    _S2_S2_PCT_SHIFT = 0
    _S2_S2_PCT_WIDTH = 40                     # bits 39–0

    # --- S3 indicator ---
    _S3_INDICATOR = 1 << 39                   # always set at S3 level

    # --- S3 extraction ---
    _S3_SHIFT = 0
    _S3_MASK = (1 << 40) - 1                  # bits 39–0

    # --- S3 s1-pct sub-field ---
    _S3_S1_PCT_SHIFT = 26
    _S3_S1_PCT_WIDTH = 13                     # bits 38–26

    # --- S3 s2-pct sub-field ---
    _S3_S2_PCT_SHIFT = 13
    _S3_S2_PCT_WIDTH = 13                     # bits 25–13

    # --- S3 gen-pct sub-field ---
    _S3_GEN_PCT_SHIFT = 0
    _S3_GEN_PCT_WIDTH = 13                     # bits 12–0

    # ==================================================================
    # Build operations (subtractive from ideal)
    # ==================================================================

    def build_s1(self, percentage: int = 100) -> KSig:
        """Build S1 significance by subtracting from ideal S1.

        S1 range bits (63–58) are always set.  The quality sub-field
        (bits 57–0, 58 bits) is scaled from the ideal (all bits set)
        proportionally to *percentage*.
        """
        pct = max(0, min(100, percentage))
        raw = self._pct_to_raw(pct, self._S1_PCT_WIDTH)
        return self._S1_RANGE | (raw << self._S1_PCT_SHIFT)

    def build_s2(self, s1_pct: int, s2_pct: int) -> KSig:
        """Build S2 significance by subtracting from ideal S2.

        Bit 57 (S2 indicator) is always set.
        s1_pct is encoded in bits 56–40 (17 bits).
        s2_pct is encoded in bits 39–0  (40 bits).
        """
        s1_raw = self._pct_to_raw(s1_pct, self._S2_S1_PCT_WIDTH)
        s2_raw = self._pct_to_raw(s2_pct, self._S2_S2_PCT_WIDTH)
        return (
            self._S2_INDICATOR
            | (s1_raw << self._S2_S1_PCT_SHIFT)
            | (s2_raw << self._S2_S2_PCT_SHIFT)
        )

    def build_s3(self, s1_pct: int, s2_pct: int, gen_pct: int) -> KSig:
        """Build S3 significance by subtracting from ideal S3.

        Bit 39 (S3 indicator) is always set.
        s1_pct is encoded in bits 38–26 (13 bits).
        s2_pct is encoded in bits 25–13 (13 bits).
        gen_pct is encoded in bits 12–0  (13 bits).
        """
        s1_raw = self._pct_to_raw(s1_pct, self._S3_S1_PCT_WIDTH)
        s2_raw = self._pct_to_raw(s2_pct, self._S3_S2_PCT_WIDTH)
        gen_raw = self._pct_to_raw(gen_pct, self._S3_GEN_PCT_WIDTH)
        return (
            self._S3_INDICATOR
            | (s1_raw << self._S3_S1_PCT_SHIFT)
            | (s2_raw << self._S3_S2_PCT_SHIFT)
            | (gen_raw << self._S3_GEN_PCT_SHIFT)
        )


# ======================================================================
# Int32Significance
# ======================================================================

class Int32Significance(IntSignificance):
    """32-bit significance implementation for KScript constructs.

    Significance occupies bits 32–63 of the 64-bit signature.
    Token space remains in bits 0–31 (unchanged).

    Bit layout (significance in bits 32–63):
        S1 range: bits 63–61  (3 bits)
        S2 range: bits 60–52  (9 bits)
        S3 range: bits 51–32  (20 bits)

    Ideals (in 64-bit context):
        S1 = 0xFFFF_FFFF_0000_0000  (all 32 sig bits)
        S2 = 0x1FFF_FFFF_F000_0000  (bits 60–32)
        S3 = 0x000F_FFFF_F000_0000  (bits 51–32)
        S4 = 0x0000_0000_0000_0000
    """

    # --- Ranges (in 64-bit context, significance shifted to top 32 bits) ---
    _S1_RANGE = ((1 << 3) - 1) << 61          # bits 63–61
    _S2_RANGE = ((1 << 9) - 1) << 52          # bits 60–52
    _S3_RANGE = ((1 << 20) - 1) << 32         # bits 51–32

    # --- Ideals ---
    _IDEAL_S1 = ((1 << 32) - 1) << 32         # bits 63–32 all set
    _IDEAL_S2 = ((1 << 29) - 1) << 32         # bits 60–32 all set
    _IDEAL_S3 = ((1 << 20) - 1) << 32         # bits 51–32 all set

    # --- Token mask ---
    _TOKEN_MASK = (1 << 32) - 1

    # --- S1 percentage (quality in bits 60–32, 29 bits) ---
    _S1_PCT_SHIFT = 32
    _S1_PCT_WIDTH = 29

    # --- S2 indicator ---
    _S2_INDICATOR = 1 << 60                   # always set at S2 level

    # --- S2 extraction ---
    _S2_SHIFT = 32
    _S2_MASK = (1 << 29) - 1                  # bits 60–32

    # --- S2 s1-pct sub-field ---
    _S2_S1_PCT_SHIFT = 52
    _S2_S1_PCT_WIDTH = 8                      # bits 59–52

    # --- S2 s2-pct sub-field ---
    _S2_S2_PCT_SHIFT = 32
    _S2_S2_PCT_WIDTH = 20                     # bits 51–32

    # --- S3 indicator ---
    _S3_INDICATOR = 1 << 51                   # always set at S3 level

    # --- S3 extraction ---
    _S3_SHIFT = 32
    _S3_MASK = (1 << 20) - 1                  # bits 51–32

    # --- S3 s1-pct sub-field ---
    _S3_S1_PCT_SHIFT = 45
    _S3_S1_PCT_WIDTH = 6                      # bits 50–45

    # --- S3 s2-pct sub-field ---
    _S3_S2_PCT_SHIFT = 39
    _S3_S2_PCT_WIDTH = 6                      # bits 44–39

    # --- S3 gen-pct sub-field ---
    _S3_GEN_PCT_SHIFT = 32
    _S3_GEN_PCT_WIDTH = 7                      # bits 38–32

    # ==================================================================
    # Build operations (subtractive from ideal)
    # ==================================================================

    def build_s1(self, percentage: int = 100) -> KSig:
        """Build S1 significance by subtracting from ideal S1.

        S1 range bits (63–61) are always set.  The quality sub-field
        (bits 60–32, 29 bits) is scaled from the ideal proportionally
        to *percentage*.
        """
        pct = max(0, min(100, percentage))
        raw = self._pct_to_raw(pct, self._S1_PCT_WIDTH)
        return self._S1_RANGE | (raw << self._S1_PCT_SHIFT)

    def build_s2(self, s1_pct: int, s2_pct: int) -> KSig:
        """Build S2 significance by subtracting from ideal S2.

        Bit 60 (S2 indicator) is always set.
        s1_pct is encoded in bits 59–52 (8 bits).
        s2_pct is encoded in bits 51–32 (20 bits).
        """
        s1_raw = self._pct_to_raw(s1_pct, self._S2_S1_PCT_WIDTH)
        s2_raw = self._pct_to_raw(s2_pct, self._S2_S2_PCT_WIDTH)
        return (
            self._S2_INDICATOR
            | (s1_raw << self._S2_S1_PCT_SHIFT)
            | (s2_raw << self._S2_S2_PCT_SHIFT)
        )

    def build_s3(self, s1_pct: int, s2_pct: int, gen_pct: int) -> KSig:
        """Build S3 significance by subtracting from ideal S3.

        Bit 51 (S3 indicator) is always set.
        s1_pct is encoded in bits 50–45 (6 bits).
        s2_pct is encoded in bits 44–39 (6 bits).
        gen_pct is encoded in bits 38–32 (7 bits).
        """
        s1_raw = self._pct_to_raw(s1_pct, self._S3_S1_PCT_WIDTH)
        s2_raw = self._pct_to_raw(s2_pct, self._S3_S2_PCT_WIDTH)
        gen_raw = self._pct_to_raw(gen_pct, self._S3_GEN_PCT_WIDTH)
        return (
            self._S3_INDICATOR
            | (s1_raw << self._S3_S1_PCT_SHIFT)
            | (s2_raw << self._S3_S2_PCT_SHIFT)
            | (gen_raw << self._S3_GEN_PCT_SHIFT)
        )
