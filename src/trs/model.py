"""Significance-routed model — reference implementation of docs/trs-significance.md."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections import deque
from fractions import Fraction


# --- §1 Definition 1 — Node algebra (word-bit realisation; ⊥ = 0) ---


def residue(x: int, y: int) -> int:
    """v ∖ w — residue, x beyond y."""
    return x & ~y


def contained(x: int, y: int) -> bool:
    """x ⪯ y — containment."""
    return x & y == x


def mu(v: int) -> int:
    """μ — the measure: mass of a node (word-bit: popcount)."""
    return v.bit_count()


# --- §1 Definition 2 — Kline ---


@dataclass(frozen=True)
class Kline:
    """⟨σ, N⟩ — signature and adjacency set (§1 Definition 2)."""

    sig: int
    adj: frozenset[int]

    def __post_init__(self) -> None:
        if self.sig == 0:
            raise ValueError("signature must be non-empty (Definition 2)")


# --- §2 Definition 4 — can: the composition fold ---


def can(n: frozenset[int]) -> int:
    """can(N) = ⋁ N; can(∅) = ⊥."""
    out = 0
    for node in n:
        out |= node
    return out


# --- §4 Definition 11 — the band ---


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


# --- §1 Definition 3 + I-1/I-2 — Domain ---


class Domain:
    """D ⊆_fin Kline with the canonical index (I-2) and lookups (Definition 7)."""

    def __init__(self, klines: "list[Kline] | tuple" = ()) -> None:
        self._klines: set[Kline] = set()
        self._canonical: dict[int, frozenset[int]] = {}
        for k in klines:
            self.add(k)

    def add(self, k: Kline) -> bool:
        """R1 store — idempotent; returns True when new."""
        if k in self._klines:
            return False
        self._klines.add(k)
        if self.is_canonical(k):
            self._canonical.setdefault(k.sig, k.adj)
        return True

    def is_canonical(self, k: Kline) -> bool:
        """§2 Definition 5."""
        return k.sig == can(k.adj)

    def __contains__(self, k: Kline) -> bool:
        return k in self._klines

    def __iter__(self):
        return iter(self._klines)

    def __len__(self) -> int:
        return len(self._klines)

    # §2 Definition 6 — witness (total)
    def witness(self, s: int) -> frozenset[int]:
        return self._canonical.get(s, frozenset())

    # §2 Definition 7 — three lookups
    def exact(self, s: int) -> "list[Kline]":
        return [k for k in self._klines if k.sig == s]

    def overlap(self, s: int) -> "list[Kline]":
        return [k for k in self._klines if k.sig & s]

    def covered(self, c: int) -> "list[Kline]":
        return [k for k in self._klines if contained(k.sig, c)]

    # I-1 — referential closure check
    def i1_closed(self) -> bool:
        sigs = {k.sig for k in self._klines}
        return all(n in sigs for k in self._klines for n in k.adj)


# --- §4 Definition 12 — decomposition ---


@dataclass(frozen=True)
class Decomposition:
    """F, U, O — fit, underfit (signature-side), overfit (adjacency-side)."""

    F: int
    U: int
    O: int


def decompose(k: Kline) -> Decomposition:
    a_hat = can(k.adj)
    f = k.sig & a_hat
    return Decomposition(F=f, U=residue(k.sig, a_hat), O=residue(a_hat, k.sig))


# --- §4 Definition 13 — entropy and graded significance ---


def significance(k: Kline) -> Fraction:
    """g(C) = μ(F)/μ(σ ∨ â) — the entropy complement (§4 Definition 13)."""
    a_hat = can(k.adj)
    return Fraction(mu(k.sig & a_hat), mu(k.sig | a_hat))


def entropy(k: Kline) -> Fraction:
    """H(C) = 1 − g(C) — surprise: the unexplained fraction of the pairing."""
    return 1 - significance(k)


BETA = Fraction(1, 2)


def effort(k: Kline, beta: Fraction = BETA) -> Fraction:
    """E(C) = (μ(U) + β·μ(O))/μ(σ ∨ â) — entropy biased by repairability (§4 Definition 14)."""
    t = decompose(k)
    total = mu(k.sig | can(k.adj))
    return Fraction(mu(t.U) + beta * mu(t.O), total)


# --- §5 — the rewrite rules ---


def r2_products(d: Domain, k: Kline) -> list[Kline]:
    """R2 — three klines for the non-empty heads, cross-wise pairing."""
    t = decompose(k)
    return [
        Kline(head, carried)
        for head, carried in (
            (t.F, d.witness(t.F)),
            (t.U, d.witness(t.O)),
            (t.O, d.witness(t.U)),
        )
        if head
    ]


def r3_substitute(d: Domain, k: Kline) -> frozenset[int]:
    """R3 — single pass: replace occurring covered signatures by their adjacencies."""
    c_n = can(k.adj)
    n_prime = set(k.adj)
    for nb in d.covered(c_n):
        if nb.sig in k.adj:  # the licence is the occurrence
            n_prime.discard(nb.sig)
            n_prime |= nb.adj
    return frozenset(n_prime)


# --- §5 — reentry: the rationalisation runner (strategy: FIFO + dedupe + bound) ---


@dataclass(frozen=True)
class End:
    """A terminal arrival — with the halt measure (Definition 14) where a kline halted."""

    reason: str
    kline: Kline | None
    effort: Fraction | None = None


@dataclass
class Rationalisation:
    stores: list[Kline]
    ends: list[End]
    rounds: int


def rationalise(d: Domain, entry: Kline, max_rounds: int = 64) -> Rationalisation:
    """Run the entry through the pipeline and the rules until S1/S4 arrival (§5, §7)."""
    # queue holds (entry, origin): the first-round candidate that opened the trace (Definition 9)
    queue: deque[tuple[Kline, Kline | None]] = deque([(entry, None)])
    seen = {entry}
    stores: list[Kline] = []
    ends: list[End] = []
    rounds = 0

    while queue and rounds < max_rounds:
        e, origin = queue.popleft()
        rounds += 1
        if not e.adj:  # R4 arrival — an entry kline with no adjacency
            ends.append(End("R4: entry kline with no adjacency", e, effort(e)))
            continue
        neighbours = d.overlap(e.sig)  # Definition 9
        if not neighbours:
            ends.append(End("no candidates — the entry terminates", e, effort(e)))
            continue
        c_e = can(e.adj)
        for nb in sorted(neighbours, key=lambda k: k.sig):
            cand = Kline(c_e, nb.adj)
            opener = origin if origin is not None else cand  # first-round candidates open their traces
            b = band(cand)
            if b is Band.S1:  # R1 — proof of full understanding: store the opening candidate
                if d.add(opener):
                    stores.append(opener)
            elif b is Band.S2:  # R2 — three products re-enter
                for p in r2_products(d, cand):
                    if p not in seen:
                        seen.add(p)
                        queue.append((p, opener))
            elif b is Band.S3:  # R3 — substitute, canonicalise, re-enter
                n_prime = r3_substitute(d, cand)
                if n_prime:
                    p = Kline(can(n_prime), n_prime)
                    if p not in seen:
                        seen.add(p)
                        queue.append((p, opener))
                else:
                    ends.append(End("R4: expansion emptied the adjacency", None))
            else:  # S4 — candidate whose neighbour supplied no adjacency
                ends.append(End("R4: candidate with no adjacency", cand, effort(cand)))

    if queue:
        ends.append(End(f"bound: reentry depth {max_rounds} reached", None))
    return Rationalisation(stores=stores, ends=ends, rounds=rounds)
