"""sig8.functions — the two pluggable seams (Q12, Q16).

The grill surfaced that a *compositional* significance has **two** functions,
not one:

1. **DecayFunction** — leaf decay: ``decay(hops) -> float in [0.0, 1.0]``.
   Applied per-node when a mismatched node resolves in ``h`` hops. Already
   locked in Q11/Q12; retained unchanged here.

2. **ComposeFunction** — composition: ``compose(slot_values) -> float in [0,1]``.
   How per-node accountedness values combine into the aggregate accountedness
   for a kline pair. Newly surfaced in Q16.

Both are ``Protocol``s with default implementations. Defaults:

- ``asymptotic_decay`` (k) — the Q12 default.
- ``mean_compose`` — preserves Q10's count-invariance: a 3-node and a 30-node
  S2 kline with equal accountedness land at the same byte.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class DecayFunction(Protocol):
    """Leaf decay: reentrant hop count -> accountedness contribution in [0,1].

    - matched / grounded                 -> 1.0  (caller supplies this directly;
                                                  decay is never called)
    - resolvable in ``h`` reentrant hops -> decay(h)
    - unresolvable                       -> 0.0  (caller supplies this directly)
    """

    def __call__(self, hops: int) -> float: ...


@runtime_checkable
class ComposeFunction(Protocol):
    """Composition: per-slot accountedness values -> aggregate in [0,1].

    The result is the *accounted fraction* (Q10) consumed by the byte encoder.
    Implementations must be count-invariant (Q10's intent) if they are to keep
    the "3-node and 30-node with equal accountedness -> same byte" property.
    """

    def __call__(self, slot_values: Sequence[float]) -> float: ...


# ── Default decay functions (Q12) ──────────────────────────────────────


def asymptotic_decay(hops: int, k: int = 50) -> float:
    """The default decay curve. ``k / (k + hops)``.

    hops 0 -> 1.0, monotone decreasing, asymptotes to 0 as hops -> inf.
    Larger ``k`` decays more slowly (more tolerance for deep resolution).
    """
    if hops < 0:
        raise ValueError(f"hops must be non-negative; got {hops}")
    return k / (k + hops)


def harmonic_decay(hops: int) -> float:
    """1 / (1 + hops). A standard harmonic decay; slower than small-k asymptote."""
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


# ── Default compose functions ──────────────────────────────────────────


def mean_compose(slot_values: Sequence[float]) -> float:
    """Default composition: arithmetic mean of per-slot accountedness.

    Count-invariant by construction: doubling the node count at equal
    accountedness leaves the mean unchanged. This is the Q10 property
    ("3-node and 30-node S2 kline with equal accountedness -> same byte").
    """
    n = len(slot_values)
    if n == 0:
        # No slots — vacuously unaccounted. The caller is expected to only
        # invoke compose when there is at least one slot.
        return 0.0
    return sum(slot_values) / n


def min_compose(slot_values: Sequence[float]) -> float:
    """Weakest-link composition: the least-accounted slot dominates.

    NOT count-invariant in the Q10 sense (adding a fully-unaccounted slot
    floors the result). Useful as a contrast to mean_compose in experiments.
    """
    if not slot_values:
        return 0.0
    return min(slot_values)


__all__ = [
    "DecayFunction",
    "ComposeFunction",
    "asymptotic_decay",
    "harmonic_decay",
    "linear_decay",
    "make_asymptotic_decay",
    "mean_compose",
    "min_compose",
]
