"""sig8.byte — the 8-bit inverted-distance representation.

Implements the locked decisions Q1, Q3, Q4, Q5, Q9.

Significance occupies the **low 8 bits** of an int. Higher bits are reserved
(carry context in the future). Access is always via masking: ``sig & 0xFF``.

The byte is a **global linear inverted distance** within ``(0x00, 0xFF)``:
higher byte = closer match. There is no reshape at the S2|S3 boundary (Q3) —
S2 and S3 are contiguous regions of one linear axis.
"""

from __future__ import annotations

# Q1: 8-bit significance lives in the low byte; mask isolates it.
SIG_MASK: int = 0xFF

# Q9: saturation guards on the two limits only.
#   0xFF is reachable ONLY by exact match (distance 0).
#   0x00 is reachable ONLY by a structural unresolvable.
# The interior (0x01..0xFE) is the open band of graded distance.
SIG_MAX: int = 0xFF  # the S1 sentinel / exact-match byte
SIG_MIN: int = 0x00  # the S4 sentinel / structural-unresolvable byte

# Q4/Q5: S2_S3_BOUNDARY is the lowest S2 byte (inclusive) and the ONLY
# configurable knob. Defaults chosen so the band split is legible.
DEFAULT_S2_S3_BOUNDARY: int = 0x80  # S2 = [0x80, 0xFE]; S3 = [0x01, 0x7F]


class BandLayout:
    """The four bands over the linear byte, derived from one boundary.

    Q5: only ``S2_S3_BOUNDARY`` is configurable. S1|S2 and S3|S4 are fixed
    sentinels exposed as named constants.

        S1 = [0xFF]                    (only exact match — distance 0)
        S2 = [S2_S3_BOUNDARY, 0xFE]    (close, direct)
        S3 = [0x01, S2_S3_BOUNDARY - 1] (indirect, decayed)
        S4 = [0x00]                    (only structural unresolvable)

    Band-representative values (the canonical bytes a producer stamps when it
    asserts a band rather than computes a distance):

        SIG_S1 = 0xFF
        SIG_S2 = 0xFE                 (the top of S2)
        SIG_S3 = S2_S3_BOUNDARY - 1    (the top of S3)
        SIG_S4 = 0x00
    """

    __slots__ = ("s2_s3_boundary",)

    def __init__(self, s2_s3_boundary: int = DEFAULT_S2_S3_BOUNDARY) -> None:
        # Interior guard: boundary must split the open band cleanly.
        if not (0x02 <= s2_s3_boundary <= 0xFE):
            raise ValueError(
                f"S2_S3_BOUNDARY must be in [0x02, 0xFE] to leave room for "
                f"non-empty S2 and S3 bands; got {s2_s3_boundary:#04x}"
            )
        self.s2_s3_boundary = s2_s3_boundary

    # Fixed sentinels (Q5).
    @property
    def SIG_S1(self) -> int:
        return 0xFF

    @property
    def SIG_S4(self) -> int:
        return 0x00

    # Boundary-derived representatives (Q5).
    @property
    def SIG_S2(self) -> int:
        return 0xFE  # top of S2

    @property
    def SIG_S3(self) -> int:
        return self.s2_s3_boundary - 1  # top of S3

    # Band classification (the routing *use*; Q7 — one quantity, two uses).
    def classify(self, sig: int) -> str:
        """Classify a raw byte into S1/S2/S3/S4. Operates on the low byte only."""
        b = sig & SIG_MASK
        if b == self.SIG_S1:
            return "S1"
        if b >= self.s2_s3_boundary:
            return "S2"
        if b >= 0x01:
            return "S3"
        return "S4"


# Distance ↔ byte conversion.
#
# Q3: global LINEAR inverted distance. distance 0 -> 0xFF; the open interior
# maps linearly; we never emit 0xFF for distance > 0 (Q9 saturation guard) nor
# 0x00 for a resolvable distance (only structural unresolvable hits 0x00).
_MAX_INTERIOR_DISTANCE: int = 0xFE  # distance at which the byte floors at 0x01


def distance_to_byte(distance: int) -> int:
    """Linear inverted distance -> byte in [0x01, 0xFF].

    - distance 0           -> 0xFF  (exact match — the only path to 0xFF)
    - distance 1..0xFE     -> 0xFE..0x01 (linear)
    - distance >= 0xFE     -> 0x01 (floors; never reaches 0x00 — Q9 guard)

    Caller is responsible for the 0x00 sentinel: this function never emits it,
    because a *computed* distance is by definition resolvable. Only a structural
    unresolvable yields SIG_S4 (0x00), and that path does not go through here.
    """
    if distance < 0:
        raise ValueError(f"distance must be non-negative; got {distance}")
    if distance == 0:
        return SIG_MAX
    if distance >= _MAX_INTERIOR_DISTANCE:
        return 0x01
    # Linear: distance 1 -> 0xFE, ..., distance 0xFD -> 0x02, distance 0xFE -> 0x01
    return SIG_MAX - distance  # in [0x01, 0xFE] for distance in [1, 0xFE]


def byte_to_distance(sig: int) -> float:
    """Inverse of distance_to_byte (for inspection/logging only).

    Returns the distance that would have produced this byte. 0x00 has no
    finite distance (it is the unresolvable sentinel) and returns +inf.
    """
    b = sig & SIG_MASK
    if b == SIG_MIN:
        return float("inf")
    return float(SIG_MAX - b)


__all__ = [
    "SIG_MASK",
    "SIG_MAX",
    "SIG_MIN",
    "DEFAULT_S2_S3_BOUNDARY",
    "BandLayout",
    "distance_to_byte",
    "byte_to_distance",
]
