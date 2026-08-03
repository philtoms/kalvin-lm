"""sig8.aggregate — the hybrid compose-on-return aggregation (Q10–Q16).

This is the prototype's reason for existing. It re-implements the aggregation
model from ``src/kalvin/expand.py`` with the new scheme:

- topology is captured on descent (which mismatched node resolves to which
  edge, with the **raw reentrant hop count** retained per node — Q13),
- decay and composition are applied **on the return phase** (the Q16
  "hybrid" decision), replacing the production sum-and-invert pattern.

The evaluation direction is exactly what ``expand()`` already does: child
``QueryCandidate`` items are yielded from recursive ``expand`` calls on the
way down, and the terminal candidate is yielded last. Here, the terminal
yield carries the composed accountedness, mapped through ``distance_to_byte``.

Per-node accountedness (Q11/12):
    matched / grounded                 -> 1.0
    resolvable in ``h`` reentrant hops -> decay(h)
    unresolvable                       -> 0.0

Aggregate (Q10): accounted fraction = compose(per_slot_values), then mapped
through the linear inverted-distance byte (Q3) with saturation guards (Q9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .byte import SIG_MAX, SIG_MIN, BandLayout, distance_to_byte
from .functions import (
    ComposeFunction,
    DecayFunction,
    make_asymptotic_decay,
    mean_compose,
)


@dataclass
class ResolvedNode:
    """One query/candidate node slot's resolution outcome.

    Captured during descent, consumed during compose-on-return. The hop count
    is the **raw reentrant hop count** (Q13) — not a band-coded constant.
    """

    # The accountedness contribution, in [0.0, 1.0]. Caller sets this directly
    # (Q11/12 + Q17a):
    #   1.0       -> matched AND grounded
    #   decay(1)  -> matched but ungrounded (Q17a: "+1 = one hop of doubt")
    #   decay(h)  -> resolvable in h reentrant hops
    #   0.0       -> unresolvable
    # Side-channel candidates (signifies / connotation) do NOT live here —
    # they are a separate yield, orthogonal to the slot's accountedness (Q17b).
    accountedness: float


@dataclass
class ExpansionResult:
    """The composed outcome of expanding one (query, candidate) pair.

    Mirrors the production ``QueryCandidate`` (query, candidate, significance),
    but carries the *intermediate* quantities so the prototype can show its
    working. Only ``significance`` survives into production; the rest are
    diagnostic.
    """

    slot_values: list[float]
    accounted_fraction: float
    significance: int  # the low-byte int (caller masks with & 0xFF)

    def __repr__(self) -> str:
        return (
            f"ExpansionResult(slots={self.slot_values}, "
            f"frac={self.accounted_fraction:.4f}, sig={self.significance:#04x})"
        )


@dataclass
class Aggregator:
    """The compose-on-return aggregator, parameterised by the two seams.

    ``layout`` defines the band geometry (only ``s2_s3_boundary`` is
    configurable — Q5). ``decay`` and ``compose`` are the two pluggable
    functions (Q12, Q16).
    """

    layout: BandLayout = field(default_factory=BandLayout)
    decay: DecayFunction = field(default_factory=make_asymptotic_decay)
    compose: ComposeFunction = field(default=mean_compose)

    # ── The compose-on-return kernel ───────────────────────────────────

    def compose_terminal(self, slot_values: list[float]) -> ExpansionResult:
        """Compose the terminal significance for one (query, candidate) pair.

        This is the replacement for the production pattern::

            significance = (~min(total_distance, D_MAX - 1)) & MASK64

        The fraction is the compose() of per-slot accountedness; the byte is
        the linear inverted distance of the *unaccounted* fraction, with the
        two saturation guards (Q9) enforced by ``distance_to_byte``.

        accounted_fraction = 1.0  -> byte 0xFF  (exact match only)
        accounted_fraction = 0.0  -> byte 0x00  (only if every slot was 0.0)
        otherwise              -> byte in (0x00, 0xFF)
        """
        frac = self.compose(slot_values)
        # Clamp for float noise.
        frac = max(0.0, min(1.0, frac))

        # Q9 saturation guards:
        if frac >= 1.0:
            byte = SIG_MAX  # 0xFF — only full account reaches here
        elif frac <= 0.0:
            byte = SIG_MIN  # 0x00 — only total non-account reaches here
        else:
            # Map the accounted fraction to a byte.
            #
            # The open interior (0x01..0xFE) is a linear inverted distance.
            # We treat the *unaccounted* fraction (1 - frac) as the "distance":
            #   frac ~ 1.0 -> distance ~ 0   -> byte ~ 0xFF
            #   frac ~ 0.0 -> distance ~ 1.0 -> byte ~ 0x01  (then 0x00 by guard)
            #
            # distance_to_byte takes an integer distance; scale the fraction
            # into the interior range [1, 0xFE] so every resolvable pair lands
            # strictly inside (0x00, 0xFF).
            distance = 1 + round((1.0 - frac) * (0xFE - 1))
            byte = distance_to_byte(distance)

        return ExpansionResult(
            slot_values=list(slot_values),
            accounted_fraction=frac,
            significance=byte,
        )

    # ── Convenience builders for the three per-slot cases (Q11/12) ──────

    def matched_slot(self) -> ResolvedNode:
        """Matched AND grounded node -> accountedness 1.0."""
        return ResolvedNode(accountedness=1.0)

    def matched_ungrounded_slot(self) -> ResolvedNode:
        """Matched but ungrounded node -> decay(1) (Q17a).

        Preserves production's "+1 penalty = one hop of doubt" semantics,
        routed through the decay seam so the pluggable curve controls it.
        """
        return ResolvedNode(accountedness=self.decay(1))

    def resolvable_slot(self, hops: int) -> ResolvedNode:
        """Resolvable in ``hops`` reentrant hops -> decay(hops)."""
        return ResolvedNode(accountedness=self.decay(hops))

    def unresolvable_slot(self) -> ResolvedNode:
        """Unresolvable node -> accountedness 0.0."""
        return ResolvedNode(accountedness=0.0)


__all__ = ["ResolvedNode", "ExpansionResult", "Aggregator"]
