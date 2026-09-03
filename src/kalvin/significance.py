"""Significance — the 8-bit compositional distance algebra and band layout.

This module owns the *topology* of significance: how a structural distance
between two klines is encoded as a single byte, how that byte is partitioned
into the S1/S2/S3/S4 bands, and how per-node accountedness composes into a
terminal grade. It is pure byte/structure algebra — no graph traversal, no
expansion proposals. The expansion pipeline (``kalvin.expand``) builds on top
of this layer; the dependency is strictly one-way.

Two clusters live here:

  1. **Byte algebra & band layout** — ``distance_to_byte`` (linear inverted
     distance → byte), ``BandLayout`` (the four bands over the byte, with
     only ``S2_S3_BOUNDARY`` configurable), the band-representative
     sentinels (``SIG_S1..SIG_S4``), and ``band_significance`` (compile-time
     structural relationship → band-representative byte).
  2. **Compositional seams & aggregation** — the ``DecayFunction`` /
     ``ComposeFunction`` protocols with default implementations, and the
     ``Aggregator`` that bundles layout + the two seams and composes a
     terminal byte via ``compose_terminal``.
  3. **Structural grounding** — structural band derivation lives in
     ``kline.sig_level``; model-state grounding queries (``is_countersigned``)
     live on :class:`kalvin.model.Model`.

Significance occupies the LOW 8 BITS of an int; access is always via
masking: ``sig & SIG_MASK``. It is a global linear inverted distance in
``(0x00, 0xFF)``: higher byte = closer match, no reshape at the S2|S3
boundary. Saturation guards: ``0xFF`` is reachable ONLY by exact match
(distance 0); ``0x00`` is reachable ONLY by a structural unresolvable;
the interior ``(0x01..0xFE)`` is the open band of graded distance.

Only ``S2_S3_BOUNDARY`` is configurable; S1|S2 and S3|S4 are fixed
sentinels.

Module-level constants and types:
  SIG_MASK, SIG8_MAX, SIG8_MIN, DEFAULT_S2_S3_BOUNDARY,
  SIG_S1, SIG_S2, SIG_S3, SIG_S4 (band-representative sentinels),
  BandLayout, distance_to_byte, Aggregator, DEFAULT_AGGREGATOR

Producer significance:
  band_significance — op → band-representative integer
  LEVEL_TO_SIG — map sig_level string to band-representative byte
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from kalvin.kline import KLine

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier

# ─────────────────────────────────────────────────────────────────────
# 8-bit compositional significance.
#
# (See module docstring for the saturation/band invariants.)
# ─────────────────────────────────────────────────────────────────────

#: Low-byte mask isolating the 8-bit significance.
SIG_MASK: int = 0xFF

#: The S1 sentinel / exact-match byte. Reachable only at distance 0.
SIG8_MAX: int = 0xFF

#: The S4 sentinel / structural-unresolvable byte. A computed (resolvable)
#: distance never yields this — only a structural unresolvable path does,
#: which does not go through ``distance_to_byte``.
SIG8_MIN: int = 0x00

#: Default for the one configurable boundary. S2 = [0x80, 0xFE];
#: S3 = [0x01, 0x7F]. Must be in [0x02, 0xFE] so both interior bands are
#: non-empty (see ``BandLayout.validate``).
DEFAULT_S2_S3_BOUNDARY: int = 0x80

#: Distance at which ``distance_to_byte`` floors at 0x01. Distances >= this
#: saturate to 0x01, never 0x00.
_MAX_INTERIOR_DISTANCE: int = 0xFE

# Band-representative significance values — the canonical bytes a producer
# stamps when asserting a band rather than computing a grade (the compiler,
# the countersign reciprocal, band_significance). Computed values from
# expand() may be any byte within a band, not only the representative.
#
# Fixed (not derived from a BandLayout): structural significance marks *which
# band a structure claims*, independent of where the configurable
# S2_S3_BOUNDARY is drawn for computed grades. BandLayout exposes matching
# layout-derived representatives for classification.
SIG_S1 = 0xFF  # exact match  (== SIG8_MAX; the S1|S2 boundary)
SIG_S2 = 0xFE  # top of S2
SIG_S3 = 0x7F  # top of S3 at the default boundary (DEFAULT_S2_S3_BOUNDARY - 1)
SIG_S4 = 0x00  # the S4 sentinel  (== SIG8_MIN; structural unresolvable)

#: Map ``kline.sig_level`` strings to band-representative bytes.
#: Inverse of ``BandLayout.classify``.
LEVEL_TO_SIG: dict[str, int] = {
    "S1": SIG_S1,
    "S2": SIG_S2,
    "S3": SIG_S3,
    "S4": SIG_S4,
}

# Compile-time production op → band-representative significance. Producers
# that assert a band rather than compute a distance (the compiler) look up
# here. The band is the Target Significance — the answer key a trainee must
# derive, not a structural measurement. CONNOTES and DENOTES both map to
# SIG_S3; IDENTITY (self-referential, word-bound or self-denote) maps to
# SIG_S1; UNKNOWN (empty, orphan) maps to SIG_S4; unknown ops default to
# SIG_S4.
_OP_TO_SIG: dict[str, int] = {
    "COUNTERSIGNS": SIG_S2,
    "CANONIZES": SIG_S2,
    "CONNOTES": SIG_S3,
    "DENOTES": SIG_S2,
    "IDENTITY": SIG_S1,
    "UNKNOWN": SIG_S4,
    "ASK": SIG_S4,
    "MTS": SIG_S1,
}


def band_significance(op: str) -> int:
    """Compile-time production op → band-representative Target Significance.

    Maps the closed set of production ops (plus IDENTITY for self-referential
    emissions) to the maximal significance of each band. The result is the
    **Target Significance** — the answer key a trainee must learn to derive,
    not a structural measurement of any one kline. Used by producers that
    assert a band rather than compute a distance (the compiler).
    Unknown ops default to ``SIG_S4``.
    """
    return _OP_TO_SIG.get(op, SIG_S4)


# ── Byte conversion (linear inverted distance) ──────────────────────


def distance_to_byte(distance: int) -> int:
    """Linear inverted ``distance`` → byte in ``[0x01, 0xFF]``.

    - ``distance == 0``        → ``0xFF`` (exact match — the only path to 0xFF)
    - ``1 <= distance <= 0xFE`` → ``0xFE..0x01`` (linear, monotone decreasing)
    - ``distance >= 0xFE``      → ``0x01`` (floors; never reaches 0x00)

    This function never emits ``0x00``: a *computed* distance is by definition
    resolvable, and only a structural unresolvable yields the ``SIG8_MIN``
    sentinel — that path does not go through here.
    """
    if distance < 0:
        raise ValueError(f"distance must be non-negative; got {distance}")
    if distance == 0:
        return SIG8_MAX
    if distance >= _MAX_INTERIOR_DISTANCE:
        return 0x01
    # Linear: distance 1 → 0xFE, ..., distance 0xFD → 0x02, distance 0xFE → 0x01.
    return SIG8_MAX - distance  # in [0x01, 0xFE] for distance in [1, 0xFE]


class BandLayout:
    """The four bands over the linear byte, derived from one boundary.

    Only ``s2_s3_boundary`` is configurable; S1|S2 and S3|S4 are fixed
    sentinels exposed as named constants.

        S1 = [0xFF]                       (only exact match — distance 0)
        S2 = [s2_s3_boundary, 0xFE]       (close, direct)
        S3 = [0x01, s2_s3_boundary - 1]    (indirect, decayed)
        S4 = [0x00]                       (only structural unresolvable)

    Band-representative values (the canonical bytes a producer stamps when
    it asserts a band rather than computes a distance)::

        sig_s1 = 0xFF
        sig_s2 = 0xFE                  (the top of S2)
        sig_s3 = s2_s3_boundary - 1     (the top of S3)
        sig_s4 = 0x00
    """

    __slots__ = ("s2_s3_boundary",)

    def __init__(self, s2_s3_boundary: int = DEFAULT_S2_S3_BOUNDARY) -> None:
        self.validate_boundary(s2_s3_boundary)
        self.s2_s3_boundary = s2_s3_boundary

    @staticmethod
    def validate_boundary(s2_s3_boundary: int) -> None:
        """Interior guard: the boundary must split the open band cleanly.

        ``[0x02, 0xFE]`` leaves room for non-empty S2 ``[b, 0xFE]`` and
        non-empty S3 ``[0x01, b-1]``.
        """
        if not isinstance(s2_s3_boundary, int) or not (
            0x02 <= s2_s3_boundary <= 0xFE
        ):
            raise ValueError(
                f"S2_S3_BOUNDARY must be an int in [0x02, 0xFE] to leave room "
                f"for non-empty S2 and S3 bands; got {s2_s3_boundary!r}"
            )

    # Fixed sentinels, as lowercase properties (ruff N802). The module-level
    # SIG_S1/SIG_S4 constants are uppercase (constants, not functions); these
    # are the layout-internal accessors.
    @property
    def sig_s1(self) -> int:
        return 0xFF

    @property
    def sig_s4(self) -> int:
        return 0x00

    # Boundary-derived representatives.
    @property
    def sig_s2(self) -> int:
        return 0xFE  # top of S2

    @property
    def sig_s3(self) -> int:
        return self.s2_s3_boundary - 1  # top of S3

    def classify(self, sig: int) -> str:
        """Classify a raw byte into S1/S2/S3/S4. Operates on the low byte only."""
        b = sig & SIG_MASK
        if b == self.sig_s1:
            return "S1"
        if b >= self.s2_s3_boundary:
            return "S2"
        if b >= 0x01:
            return "S3"
        return "S4"


# ── Pluggable seams for compositional significance ───────────────────
#
# A compositional significance has two functions:
#   1. DecayFunction  — leaf decay: reentrant hop count -> accountedness.
#   2. ComposeFunction — how per-node accountedness values combine.
# Both are Protocols with default implementations; expand() is parameterised
# over them via Aggregator.


@runtime_checkable
class DecayFunction(Protocol):
    """Leaf decay: reentrant hop count -> accountedness contribution in [0, 1].

    The caller (expand) supplies the boundary cases directly:
      - matched AND grounded     -> 1.0  (decay is never called)
      - matched but ungrounded   -> decay(1)  (one hop of doubt)
      - resolvable in h hops      -> decay(h)
      - unresolvable              -> 0.0  (decay is never called)
    """

    def __call__(self, hops: int) -> float: ...


@runtime_checkable
class ComposeFunction(Protocol):
    """Composition: per-slot accountedness values -> aggregate in [0, 1].

    The result is the accounted fraction consumed by the byte encoder.
    Implementations should be count-invariant (scaling the same accountedness
    distribution leaves the byte unchanged) unless they deliberately trade
    that property away.
    """

    def __call__(self, slot_values: Sequence[float]) -> float: ...


# ── Default decay functions ──────────────────────────────────────────


def asymptotic_decay(hops: int, k: int = 50) -> float:
    """The default decay curve: ``k / (k + hops)``.

    hops 0 -> 1.0, monotone decreasing, asymptotes to 0 as hops -> inf.
    Larger ``k`` decays more slowly (more tolerance for deep resolution).
    """
    if hops < 0:
        raise ValueError(f"hops must be non-negative; got {hops}")
    return k / (k + hops)


def harmonic_decay(hops: int) -> float:
    """``1 / (1 + hops)``. A standard harmonic decay; slower than small-k asymptote."""
    if hops < 0:
        raise ValueError(f"hops must be non-negative; got {hops}")
    return 1.0 / (1.0 + hops)


def linear_decay(hops: int, reach: int = 100) -> float:
    """Linear decay to 0 at ``hops == reach``; clamps at 0 beyond.

    ``reach`` is the hop count at which a node is considered fully unaccounted.
    """
    if hops < 0:
        raise ValueError(f"hops must be non-negative; got {hops}")
    if reach <= 0:
        raise ValueError(f"reach must be positive; got {reach}")
    return max(0.0, 1.0 - hops / reach)


def make_asymptotic_decay(k: int = 50) -> DecayFunction:
    """Curry ``k`` into an asymptotic_decay callable."""

    def _decay(hops: int) -> float:
        return asymptotic_decay(hops, k=k)

    return _decay


# ── Default compose functions ────────────────────────────────────────


def mean_compose(slot_values: Sequence[float]) -> float:
    """Default composition: arithmetic mean of per-slot accountedness.

    Count-invariant by construction: scaling the same accountedness
    distribution (repeating it) leaves the mean — and thus the byte —
    unchanged.
    """
    n = len(slot_values)
    if n == 0:
        # No slots — vacuously unaccounted. Callers only invoke compose when
        # there is at least one slot; this guard is defensive.
        return 0.0
    return sum(slot_values) / n


@dataclass(frozen=True)
class Aggregator:
    """The compose-on-return policy, bundling layout + the two seams.

    ``expand()`` is parameterised over a single ``Aggregator`` (keyword-only,
    defaulting to :data:`DEFAULT_AGGREGATOR`) so the call site stays readable
    while the recursion threads one object. Freezing makes it safe to share
    across recursive calls and threads.
    """

    layout: BandLayout = field(default_factory=BandLayout)
    decay: DecayFunction = field(default_factory=make_asymptotic_decay)
    compose: ComposeFunction = field(default=mean_compose)

    def compose_terminal(self, slot_values: list[float]) -> int:
        """Compose per-slot accountedness into the terminal significance byte.

        Maps the compose() of per-slot accountedness through the linear
        inverted-distance byte, with the two saturation guards:

          accounted_fraction == 1.0 -> 0xFF  (exact match only)
          accounted_fraction == 0.0 -> 0x00  (only total non-account)
          otherwise                 -> byte in (0x00, 0xFF)
        """
        frac = max(0.0, min(1.0, self.compose(slot_values)))  # clamp float noise
        if frac >= 1.0:
            return SIG8_MAX
        if frac <= 0.0:
            return SIG8_MIN
        # Map the unaccounted fraction (1 - frac) to an interior distance in
        # [1, 0xFE] so every resolvable pair lands strictly inside (0x00, 0xFF).
        interior_distance = 1 + round((1.0 - frac) * (_MAX_INTERIOR_DISTANCE - 1))
        return distance_to_byte(interior_distance)


#: Module-level default aggregator: default layout, asymptotic decay (k=50),
#: mean compose. cogitator uses this unless constructed otherwise.
DEFAULT_AGGREGATOR = Aggregator()


@dataclass(frozen=True)
class ProposalAggregator(Aggregator):
    """Proposal grading: the S2|S3 boundary is the zero point of judgement.

    A proposal is a conjunction of claims. Its weakest claim decides the
    sign; the rest is context — how well K understands what it is saying.

    - Every slot at least partially accounted: positive. The byte rises
      from the boundary with the weakest claim's accountedness.
    - Any slot unaccounted (a wild guess — no connotational path): negative.
      The byte falls from the boundary with the mean accountedness over
      *all* slots — quality of the understood part times coverage. More
      wild guesses means less of the proposal understood, so a smaller
      negative distance from noise; fewer guesses stays closer to the
      boundary (a more confident long-worded "no").

    0xFF stays reserved for the exact case (all slots at 1.0 — the
    grounded shortcut in the grader handles ratified shapes first).
    """

    def compose_terminal(self, slot_values: list[float]) -> int:
        if not slot_values:
            return SIG8_MIN
        boundary = self.layout.s2_s3_boundary
        weakest = min(slot_values)
        if weakest > 0.0:
            frac = min(1.0, weakest)
            if frac >= 1.0:
                return SIG8_MAX
            return boundary + round(frac * (SIG8_MAX - 1 - boundary))
        context = max(0.0, min(1.0, sum(slot_values) / len(slot_values)))
        return round(context * (boundary - 1))


#: Proposal grading aggregator (see :class:`ProposalAggregator`).
PROPOSAL_AGGREGATOR = ProposalAggregator()


# Structural Grounding


def structural_sig(level: str) -> int:
    """Map a ``sig_level`` string ("S1"–"S4") to its band-representative byte.

    Inverse of ``BandLayout.classify``; used by callers that need the int byte
    (e.g. ``KValue`` construction) from a ``sig_level`` result.
    """
    return LEVEL_TO_SIG[level]
