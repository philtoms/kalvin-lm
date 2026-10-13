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
    if tx and ty: #and set(ty) <= set(tx):
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

    def merge(self, m: Kline | None = None) -> Kline:
        return Kline(self.sig.merge(m.sig), self.adj | m.adj) if m else self

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


def r2_products(c: Kline, k: Kline) -> tuple[list[Kline], Kline]:
    """R2 (§5 Rule R2) — same-side products for the non-empty residual
    heads; the fit kline is earned, not re-entered."""
    t = decompose(c)
    n_f = frozenset(n for n in c.adj if contained(n, t.F))
    n_o = frozenset(n for n in c.adj if contained(n, t.O))
    n_u = frozenset(n for n in k.adj if contained(n, t.U))
    products = []
    k_f = Kline(t.F, n_f)
    if t.U:
        products.append(Kline(t.U, n_u))  # ⟨U, N_U⟩ — the entry's U-side
    if t.O:
        products.append(Kline(t.O, n_o))  # ⟨O, N_O⟩ — the candidate's O-side
    return products, k_f


def r3_substitute(d: Domain, k: Kline) -> frozenset[KNode]:
    """R3 — single pass: replace occurring covered signatures by their adjacencies.

    Several covered klines may share a signature; the fold order is §7
    strategy, and the reference takes them sorted by (sig, adjacency)."""
    c_n = can(k.adj)
    n_prime = set(k.adj)
    for nb in sorted(d.covered(c_n), key=lambda x: (x.sig, tuple(sorted(x.adj)))):
        if nb.sig in k.adj and nb.adj != frozenset([nb.sig]):  # like for like is no substitution
            n_prime.discard(nb.sig)
            n_prime |= nb.adj
    return frozenset(n_prime)


# --- §5 — reentry: the rationalisation runner ---


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


def rationalise(
    d: Domain, entry: Kline, depth: int = 0, seen: set[KNode] | None = None, earned: Kline | None = None
) -> Rationalisation:
    """§5 as one recursive function — the standardised reentry.

    Every kline runs the Definition 9 pipeline as an entry (presented or
    re-entry, Definition 8); its candidates route by band. R1 stores the
    candidate merged with the earned composition — an S1 arrival
    directly, or an S3 arrival whose substituted re-entry returns with
    stores. The recursion returns at S1 — the entry stored — or S4.
    Bounds (§7 strategy): re-entry depth 10 and a per-run memo of
    processed entry canons.
    """

    stores: list[Kline] = []
    ends: list[End] = []
    if seen is None:
        seen = set()  # the run's canon memo — §7 strategy
    if not entry.adj:  # R4 arrival — a presented entry with no adjacency
        ends.append(End("R4: entry kline with no adjacency", entry, effort(entry)))
        return Rationalisation(stores, ends, depth)
    c_e = can(entry.adj)
    if c_e in seen:
        ends.append(End(f"bound: recursion seen {c_e}", entry, effort(entry)))
        return Rationalisation(stores, ends, depth)
    seen.add(c_e)
    if depth >= 10:
        ends.append(End("bound: recursion depth reached", entry, effort(entry)))
        return Rationalisation(stores, ends, depth)
    nbs = d.overlap(entry.sig)
    if not nbs:
        ends.append(End("no candidates — the entry terminates", entry, effort(entry)))
        return Rationalisation(stores, ends, depth)
    cands = [Kline(c_e, nb.adj) for nb in nbs]  # neighbour order is §7 strategy
    for c in cands:
        ok = False
        # later candidates inherit the earned adjacencies (§5)
        b = band(c)
        if b is Band.S1:
            ok = True

        elif b is Band.S2:
            # the fit kline is earned — merged into the composition that
            # rides with the split; the residual products re-enter in
            # order, each carrying it, and ground for their own sake — no
            # product return stores the candidate. A product with no
            # adjacency routes to R4.
            products, nf = r2_products(c, entry)
            for prod in products:
                r = rationalise(d, prod, depth + 1, seen, nf.merge(earned))
                ends.extend(r.ends)
                stores.extend(r.stores)
                
        elif b is Band.S3:
            n_prime = r3_substitute(d, c)  # R3 — substitute the candidate
            if not n_prime:
                ends.append(End("R4: expansion emptied the adjacency", c, effort(c)))
            elif n_prime == c.adj:
                ends.append(End("R4: substitution like for like", c, effort(c)))
            else:
                r_prod = rationalise(d, Kline(c.sig, n_prime), depth + 1, seen)  # the claim is kept
                ends.extend(r_prod.ends)  # the product's halts are the tree's halts
                if r_prod.stores:
                    ok = True

        elif b is Band.S4:
            ends.append(End("R4: candidate with no adjacency", c, effort(c)))

        if ok:
            s = c.merge(earned)
            d.add(s)
            if not c in stores:
                stores.append(s)
        
    return Rationalisation(stores, ends, depth)


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
