"""The shared algebra — klines, the node operations, the band and the
grading measures (docs/rationalisation.md §1, §4). The kline itself is
kalvin's KLine: signature and nodes, the nodes a list — a construction
can cite the same signature twice."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from fractions import Fraction

from kalvin.kline import KLine, KNode, mts_initials


# --- §1 — Node algebra (word-bit realisation; ⊥ = 0) ---


def _provenance(x: KNode, y: KNode, op: str) -> str:
    """The label of a value derived from two operands."""
    a, b = getattr(x, "label", ""), getattr(y, "label", "")
    return f"{a}{op}{b}" if a and b else ""


def _derived(v: int, x: KNode, y: KNode, op: str) -> str:
    """The label of a derived value: its MTS word decomposition when the
    word registry covers it (WDMH ∖ MHALL → WDH), else the two-operand
    provenance."""
    if not v:
        return "⊥"
    return mts_initials(v) or _provenance(x, y, op)


def underfit(x: KNode, y: KNode) -> KNode:
    """v ∖ w — residue, x beyond y."""
    v = x & ~y
    return KNode(v, _derived(v, x, y, "∖"))

def overfit(x: Iterable[KNode], y: KNode ) -> KNode:
    """v ∖ w — residue, x beyond y."""
    _x = can(x)
    v = _x & ~y
    return KNode(v, _derived(v, _x, y, "∖"))


def fit(x: KNode, y: KNode) -> KNode:
    """v ∧ w — meet."""
    v = x & y
    return KNode(v, _derived(v, x, y, "∧"))


def contained(x: KNode, y: KNode) -> bool:
    """x ⪯ y — containment."""
    return x & y == x


def mu(v: int) -> int:
    """μ — the measure: mass of a node (word-bit: popcount)."""
    return v.bit_count()


# --- §1 Definition 2 — Kline, and the composition fold ---


def can(n: Iterable[KNode]) -> KNode:
    """can(N) = ⋁ N; can(∅) = ⊥ — labels compose through the fold. The
    fold is blind to order and repetition: a node cited twice contributes
    once."""
    out = KNode(0)
    for node in n:
        out = out.merge(node)
    return out


# --- §4 — the band, decomposition, and the measures ---


class Band(Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"


def band(k: KLine) -> Band:
    """V(C) — ordered cases (§4 Definition 11)."""
    if not k.nodes:
        return Band.S4
    a_hat = can(k.nodes)
    if k.signature == a_hat:
        return Band.S1
    if k.signature & a_hat:
        return Band.S2
    return Band.S3


def significance(k: KLine) -> Fraction:
    """g(C) = μ(F)/μ(σ ∨ â) — the entropy complement."""
    a_hat = can(k.nodes)
    return Fraction(mu(k.signature & a_hat), mu(k.signature | a_hat))


def entropy(k: KLine) -> Fraction:
    """H(C) = 1 − g(C) — the unexplained fraction of the pairing."""
    return 1 - significance(k)


def fingerprint(k: KLine) -> int:
    return k.signature | can(k.nodes)

BETA = Fraction(1, 2)


def effort(k: KLine, beta: Fraction = BETA) -> Fraction:
    """E(C) = (μ(U) + β·μ(O))/μ(σ ∨ â) — entropy biased by repairability."""
    a_hat = can(k.nodes)
    total = mu(k.signature | a_hat)
    return Fraction(
        mu(underfit(k.signature, a_hat)) + beta * mu(underfit(a_hat, k.signature)), total
    )
