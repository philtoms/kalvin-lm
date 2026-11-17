"""The shared algebra — klines, the node operations, the band and the
grading measures (docs/rationalisation.md §1, §4). The kline itself is
kalvin's KLine: signature and nodes, the nodes a list — a construction
can cite the same signature twice."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from kalvin.kline import KLine, KNode


# --- §1 — Node algebra (word-bit realisation; ⊥ = 0) ---


def _provenance(x: KNode, y: KNode, op: str) -> str:
    """The label of a value derived from two operands."""
    a, b = getattr(x, "label", ""), getattr(y, "label", "")
    return f"{a}{op}{b}" if a and b else ""


def _tokens(label: str) -> "list[str] | None":
    """The word tokens of a pure composition label ('A|B'), else None."""
    if not label:
        return None
    parts = label.split("|")
    if any(not p or ("∖" in p) or ("∧" in p) for p in parts):
        return None
    return parts


def underfit(x: KNode, y: KNode) -> KNode:
    """v ∖ w — residue, x beyond y."""
    label = _provenance(x, y, "∖")
    tx, ty = _tokens(getattr(x, "label", "")), _tokens(getattr(y, "label", ""))
    if tx and ty:
        keep = [p for p in tx if p not in ty]
        label = "|".join(keep) if keep else "⊥"
    return KNode(x & ~y, label)

def overfit(x: Iterable[KNode], y: KNode ) -> KNode:
    """v ∖ w — residue, x beyond y."""
    _x = can(x)
    label = _provenance(_x, y, "∖")
    tx, ty = _tokens(getattr(_x, "label", "")), _tokens(getattr(y, "label", ""))
    if tx and ty:
        keep = [p for p in tx if p not in ty]
        label = "|".join(keep) if keep else "⊥"
    return KNode(_x & ~y, label)


def fit(x: KNode, y: KNode) -> KNode:
    """v ∧ w — meet; shared tokens label it."""
    label = _provenance(x, y, "∧")
    tx, ty = _tokens(getattr(x, "label", "")), _tokens(getattr(y, "label", ""))
    if tx and ty:
        shared = [p for p in tx if p in ty]
        if shared:
            label = "|".join(shared)
    return KNode(x & y, label)


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


@dataclass(frozen=True)
class Decomposition:
    """F, U, O — fit, underfit (signature-side), overfit (adjacency-side)."""

    F: KNode
    U: KNode
    O: KNode


def decompose(k: KLine) -> Decomposition:
    a_hat = can(k.nodes)
    f = fit(k.signature, a_hat)
    return Decomposition(F=f, U=underfit(k.signature, a_hat), O=underfit(a_hat, k.signature))


def significance(k: KLine) -> Fraction:
    """g(C) = μ(F)/μ(σ ∨ â) — the entropy complement."""
    a_hat = can(k.nodes)
    return Fraction(mu(k.signature & a_hat), mu(k.signature | a_hat))


def entropy(k: KLine) -> Fraction:
    """H(C) = 1 − g(C) — the unexplained fraction of the pairing."""
    return 1 - significance(k)


BETA = Fraction(1, 2)


def effort(k: KLine, beta: Fraction = BETA) -> Fraction:
    """E(C) = (μ(U) + β·μ(O))/μ(σ ∨ â) — entropy biased by repairability."""
    t = decompose(k)
    total = mu(k.signature | can(k.nodes))
    return Fraction(mu(t.U) + beta * mu(t.O), total)
