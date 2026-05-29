"""Significance level classification for KLine signatures."""

from __future__ import annotations


class Int32Significance:
    """Classify signatures into significance levels (S1–S4)."""

    def get_level(self, signature: int) -> str:
        """Return the significance level for a signature value.

        Uses bit-count heuristic: more set bits → higher significance.
        """
        if signature == 0:
            return "S4"
        bit_count = bin(signature).count("1")
        if bit_count >= 32:
            return "S1"
        elif bit_count >= 16:
            return "S2"
        elif bit_count >= 8:
            return "S3"
        else:
            return "S4"
