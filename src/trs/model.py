"""Significance-routed model — reference implementation of docs/trs-significance.md."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from kalvin.kline import KNode


# --- §1 Definition 1 — Node algebra (word-bit realisation; ⊥ = 0) ---


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
    """v ∖ w — residue, x beyond y. Where both labels are pure
    compositions and y's tokens are contained in x's, the label is the
    remaining tokens — the content, not the expression."""
    label = _provenance(x, y, "∖")
    tx, ty = _tokens(getattr(x, "label", "")), _tokens(getattr(y, "label", ""))
    if tx and ty and set(ty) <= set(tx):
        keep = [p for p in tx if p not in ty]
        label = "|".join(keep) if keep else "⊥"
    return KNode(x & ~y, label)


def meet(x: KNode, y: KNode) -> KNode:
    """v ∧ w — meet, carried as a labelled node; shared tokens label it."""
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


# --- §1 Definition 2 — Kline ---


@dataclass(frozen=True)
class Kline:
    """⟨σ, N⟩ — signature and adjacency set (§1 Definition 2)."""

    sig: KNode
    adj: frozenset[KNode]

    def __post_init__(self) -> None:
        if self.sig == 0:
            raise ValueError("signature must be non-empty (Definition 2)")


# --- §2 Definition 4 — can: the composition fold ---


def can(n: frozenset[KNode]) -> KNode:
    """can(N) = ⋁ N; can(∅) = ⊥ — labels compose through the fold."""
    out = KNode(0)
    for node in n:
        out = out.merge(node)
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


# --- §1 Definition 3 — Domain ---


class Domain:
    """D ⊆_fin Kline with the first-stored witness index (Definition 6) and lookups (Definition 7)."""

    def __init__(self, klines: "list[Kline] | tuple" = ()) -> None:
        self._klines: set[Kline] = set()
        self._order: list[Kline] = []  # insertion order — first-stored is observable
        self._canonical: dict[KNode, frozenset[KNode]] = {}
        for k in klines:
            self.add(k)

    def add(self, k: Kline) -> bool:
        """R1 store — idempotent; returns True when new."""
        if k in self._klines:
            return False
        self._klines.add(k)
        self._order.append(k)
        if self.is_canonical(k):
            self._canonical.setdefault(k.sig, k.adj)  # first-stored wins (Definition 6)
        return True
    def is_canonical(self, k: Kline) -> bool:
        """§2 Definition 5."""
        return k.sig == can(k.adj)

    def __contains__(self, k: Kline) -> bool:
        return k in self._klines

    def __iter__(self):
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._klines)

    # §2 Definition 6 — witness (total)
    def witness(self, s: KNode) -> frozenset[KNode]:
        return self._canonical.get(s, frozenset())

    # §2 Definition 7 — three lookups
    def exact(self, s: KNode) -> "list[Kline]":
        return [k for k in self._klines if k.sig == s]

    def overlap(self, s: KNode) -> "list[Kline]":
        return [k for k in self._klines if k.sig & s]

    def covered(self, c: KNode) -> "list[Kline]":
        return [k for k in self._klines if contained(k.sig, c)]


# --- §4 Definition 12 — decomposition ---


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


def r3_substitute(d: Domain, k: Kline) -> frozenset[KNode]:
    """R3 — single pass: replace occurring covered signatures by their adjacencies.

    Several covered klines may share a signature; the fold order is §7
    strategy, and the reference takes them sorted by (sig, adjacency)."""
    c_n = can(k.adj)
    n_prime = set(k.adj)
    for nb in sorted(d.covered(c_n), key=lambda x: (x.sig, tuple(sorted(x.adj)))):
        if nb.sig in k.adj:  # the licence is the occurrence
            n_prime.discard(nb.sig)
            n_prime |= nb.adj
    return frozenset(n_prime)


# --- §5 — reentry: the rationalisation runner (strategy: LIFO stack, duplicates allowed, bounded) ---


@dataclass(frozen=True)
class End:
    """A terminal arrival — with the halt measure (Definition 14) where a kline halted."""

    reason: str
    kline: Kline | None
    effort: Fraction | None = None


@dataclass(frozen=True)
class Proposal:
    """A halted kline read as a request (§5 Definition 15)."""

    kind: str  # "acquire" | "connect" | "continue"
    kline: Kline
    effort: Fraction


@dataclass
class Rationalisation:
    stores: list[Kline]
    ends: list[End]
    rounds: int


def rationalise(d: Domain, entry: Kline, max_rounds: int = 64) -> Rationalisation:
    """Run the entry through the pipeline and the rules until S1/S4 arrival (§5, §7).

    The work queue is an ordinary stack: duplicates are allowed and the
    next work entry is the latest queued. The opener is cleared on pop and
    set to the entry's first candidate — it does not change for that work
    entry (Definition 9)."""
    queue: list[Kline] = [entry]
    stores: list[Kline] = []
    ends: list[End] = []
    rounds = 0

    while queue and rounds < max_rounds:
        e = queue.pop()  # the latest queued item
        rounds += 1
        if not e.adj:  # R4 arrival — an entry kline with no adjacency
            ends.append(End("R4: entry kline with no adjacency", e, effort(e)))
            continue
        neighbours = d.overlap(e.sig)  # Definition 9
        if not neighbours:
            ends.append(End("no candidates — the entry terminates", e, effort(e)))
            continue
        c_e = can(e.adj)
        opener: Kline | None = None  # cleared on pop; set by the first candidate
        # NOTE (reentry redesign): sig-ascending candidate order is
        # incidental, not algebraic — revisit.
        for nb in sorted(neighbours, key=lambda k: k.sig):
            cand = Kline(c_e, nb.adj)
            if opener is None:
                opener = cand  # opens this work entry's trace — unchanging for it
            b = band(cand)
            if b is Band.S1:  # R1 — proof of full understanding: store the opening candidate
                if d.add(opener):
                    stores.append(opener)
            elif b is Band.S2:  # R2 — three products re-enter
                for p in r2_products(d, cand):
                    queue.append(p)
            elif b is Band.S3:  # R3 — substitute, canonicalise, re-enter
                n_prime = r3_substitute(d, cand)
                if n_prime:
                    queue.append(Kline(can(n_prime), n_prime))
                else:
                    ends.append(End("R4: expansion emptied the adjacency", cand, effort(cand)))
            else:  # S4 — candidate whose neighbour supplied no adjacency
                ends.append(End("R4: candidate with no adjacency", cand, effort(cand)))

    if queue:
        pending = min(queue, key=effort)  # the frontier's cheapest continuation
        ends.append(End(f"bound: reentry depth {max_rounds} reached", pending, effort(pending)))
    return Rationalisation(stores=stores, ends=ends, rounds=rounds)


def rationalise2(d: Domain, entry: Kline, max_depth: int = 64) -> Rationalisation:
    """The step-1.2 redesign prototype.

    A work item's value is a head; its candidates are its neighbours'
    pairings, tried in order until one resolves at S1. R1 stores the work
    item's FIRST candidate — a later candidate resolving at S1 still
    stores the first. The S2 products are consumed as one response: F
    accumulates the witness, U is recursively applied as its own work
    item, and the enriched F is pushed and iterated — a new work item
    whose opener is itself.
    """
    stores: list[Kline] = []
    ends: list[End] = []
    rounds = 0

    def bare(u: KNode) -> Kline:
        return Kline(u, frozenset())

    def candidates(head: KNode) -> list[Kline]:
        nbs = d.overlap(head)
        return [Kline(head, nb.adj) for nb in sorted(nbs, key=lambda k: k.sig)]  # ordering: not algebraic

    def consume(u: KNode, depth: int) -> bool:
        """A work item: try its candidates in order."""
        nonlocal rounds
        rounds += 1
        if depth >= max_depth:
            ends.append(End("bound: recursion depth reached", bare(u), effort(bare(u))))
            return False
        cands = candidates(u)
        if not cands:
            ends.append(End("no candidates — the work item terminates", bare(u), effort(bare(u))))
            return False
        for cand in cands:
            if resolve(cand, cand, depth):
                return True
        ends.append(End("no candidate resolved", bare(u), effort(bare(u))))
        return False

    def resolve(cand: Kline, opener: Kline, depth: int) -> bool:
        """One candidate's chain; True when it reaches S1."""
        nonlocal rounds
        rounds += 1
        if depth >= max_depth:
            ends.append(End("bound: recursion depth reached", cand, effort(cand)))
            return False
        mark = len(stores)
        b = band(cand)
        if b is Band.S1:
            d.add(opener)  # R1 — the sole store point; stores, always
            stores.append(opener)
            return True
        if b is Band.S2:
            t = decompose(cand)
            held: set = set(cand.adj)  # F holds the witness that produced the fit
            if t.U:
                consume(t.U, depth + 1)  # U is recursively applied
            for s in stores[mark:]:  # stored signatures in sigma: add to F
                if s.sig & cand.sig == s.sig:
                    held |= s.adj
            if held == set(cand.adj):
                return False  # the pass grew nothing — this candidate has stalled
            enriched = Kline(cand.sig, frozenset(held))
            return resolve(enriched, opener, depth + 1)  # push and iterate — a new work item
        if b is Band.S3:
            n_prime = r3_substitute(d, cand)
            if n_prime:
                return resolve(Kline(cand.sig, n_prime), opener, depth + 1)
            ends.append(End("R4: expansion emptied the adjacency", cand, effort(cand)))
            return False
        ends.append(End("R4: candidate with no adjacency", cand, effort(cand)))
        return False

    if not entry.adj:
        ends.append(End("R4: entry kline with no adjacency", entry, effort(entry)))
    else:
        consume(can(entry.adj), 0)
    return Rationalisation(stores=stores, ends=ends, rounds=rounds)


def proposals(r: Rationalisation) -> list[Proposal]:
    """Every halt generates one proposal (§5 Definition 15), ordered by graded
    significance — ascending effort, ties in generation order."""
    out = []
    for e in r.ends:
        if e.kline is None:
            continue
        if e.reason.startswith("no candidates"):
            kind = "connect"
        elif e.reason.startswith("bound"):
            kind = "continue"
        else:
            kind = "acquire"
        out.append(Proposal(kind, e.kline, e.effort if e.effort is not None else effort(e.kline)))
    return sorted(out, key=lambda p: p.effort)  # least entropy first — most significant next step
