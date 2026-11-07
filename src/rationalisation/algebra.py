"""The shared algebra — klines, the node operations, the band and the
grading measures (docs/rationalisation.md §1, §4). Shared by the
Cogitator and the ReferenceCogitator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from kalvin.kline import KNode, KNone


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


def residue(x: KNode, y: KNode) -> KNode:
    """v ∖ w — residue, x beyond y."""
    label = _provenance(x, y, "∖")
    tx, ty = _tokens(getattr(x, "label", "")), _tokens(getattr(y, "label", ""))
    if tx and ty:
        keep = [p for p in tx if p not in ty]
        label = "|".join(keep) if keep else "⊥"
    return KNode(x & ~y, label)


def meet(x: KNode, y: KNode) -> KNode:
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


@dataclass(frozen=True)
class Kline:
    """⟨σ, N⟩ — signature and adjacency set."""

    sig: KNode
    adj: frozenset[KNode]

    def __post_init__(self) -> None:
        if self.sig == 0:
            raise ValueError("signature must be non-empty (Definition 2)")

    def fingerprint(self) -> int:
        return self.sig | can(self.adj)


def can(n: frozenset[KNode]) -> KNode:
    """can(N) = ⋁ N; can(∅) = ⊥ — labels compose through the fold."""
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


def band(k: Kline) -> Band:
    """V(C) — ordered cases (§4 Definition 11)."""
    if not k.adj:
        return Band.S4
    a_hat = can(k.adj)
    if k.sig == a_hat:
        return Band.S1
    if k.sig & a_hat:
        return Band.S2
    return Band.S3


@dataclass(frozen=True)
class Decomposition:
    """F, U, O — fit, underfit (signature-side), overfit (adjacency-side)."""

    F: KNode
    U: KNode
    O: KNode


def decompose(k: Kline) -> Decomposition:
    a_hat = can(k.adj)
    f = meet(k.sig, a_hat)
    return Decomposition(F=f, U=residue(k.sig, a_hat), O=residue(a_hat, k.sig))


def significance(k: Kline) -> Fraction:
    """g(C) = μ(F)/μ(σ ∨ â) — the entropy complement."""
    a_hat = can(k.adj)
    return Fraction(mu(k.sig & a_hat), mu(k.sig | a_hat))


def entropy(k: Kline) -> Fraction:
    """H(C) = 1 − g(C) — the unexplained fraction of the pairing."""
    return 1 - significance(k)


BETA = Fraction(1, 2)


def effort(k: Kline, beta: Fraction = BETA) -> Fraction:
    """E(C) = (μ(U) + β·μ(O))/μ(σ ∨ â) — entropy biased by repairability."""
    t = decompose(k)
    total = mu(k.sig | can(k.adj))
    return Fraction(mu(t.U) + beta * mu(t.O), total)
