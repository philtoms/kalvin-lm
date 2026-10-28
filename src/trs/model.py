"""Significance-routed model — reference implementation of docs/trs-significance.md."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from kalvin.kline import KNode, KNone


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

    def fingerprint(self) -> int:
        return self.sig | can(self.adj)
    
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

    def __init__(self, klines: list[Kline] | tuple = ()) -> None:
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
    def exact(self, s: KNode) -> list[Kline]:
        return [k for k in self._klines if k.sig == s]

    def overlap(self, s: KNode) -> list[Kline]:
        return [k for k in self._klines if k.sig & s]

    def covered(self, c: KNode) -> list[Kline]:
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
    heads; the fit kline is returned separately and stores in its own
    right."""
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

@dataclass
class Candidate:
    """The open-format candidate — the unit of work on the reentry stack."""

    sig: KNode  # candidate signature
    F: KNode
    U: KNode
    O: KNode
    n_f: frozenset[KNode]
    n_u: frozenset[KNode]
    n_o: frozenset[KNode]
    C: Kline
    E: Kline


def next_candidate(c: Kline, entry: Kline) -> Candidate:
    """The popped candidate in open format: F/U/O decompose the claim
    against the candidate, f/o carry the candidate's own nodes, u the
    entry's."""
    f = meet(c.sig, entry.sig)
    t = Decomposition(F=f, O=residue(c.sig, entry.sig), U=residue(entry.sig, c.sig))

    cand = Candidate(
        sig=can(entry.adj),
        F=t.F,
        U=t.U,
        O=t.O,
        n_f=frozenset(n for n in c.adj if contained(n, t.F)),
        n_o=frozenset(n for n in c.adj if contained(n, t.O)),
        n_u=frozenset(n for n in entry.adj if contained(n, t.U)),
        C=c,
        E=entry
    )
    return cand

def grounded(d: Domain, s: KNode) -> bool:
    """A signature is grounded when it exists in the domain."""
    return bool(d.exact(s))


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

class Cogitator:

    MAX_DEPTH = 10

    def __init__(self, d: Domain) -> None:
        self.d = d
        self.stores: list[Kline] = []
        self.ends: list[End] = []

    def _halt(self, reason: str, k: Kline) -> None:
        """A terminal arrival — with the halt measure (Definition 14)."""
        self.ends.append(End(reason, k, effort(k)))

    def _admit(self, c: Kline) -> None:
        """R1 — store; idempotent, reported once per run."""
        self.d.add(c)
        if c not in self.stores:
            self.stores.append(c)

    def rationalise(self, entry: Kline, seen: set[int] = set(), framed: KNode = KNone, depth: int = 0) -> Rationalisation:
        if entry.fingerprint() in seen:
            return Rationalisation(self.stores, self.ends, 0)

        seen.add(entry.fingerprint())
        rounds = 0
        cands = self._reenter(entry, framed, depth)
        for cand in cands:
            while True:
                rounds += 1
                pending = len(self.stores)

                if not cand.n_u and not cand.n_o:  # the sides resolved — it grounds
                    self._admit(Kline(cand.sig, cand.n_f))
                    break

                if cand.sig == can(cand.n_f):  # f composes the claim — ground F == C
                    self._admit(Kline(cand.sig, cand.n_f))
                    break

                """Merge grounded side regions into f; reenter the ungrounded."""
                if cand.n_u:
                    if grounded(self.d, can(cand.n_u)):  # the region's witnesses compose it
                        cand.n_f, cand.n_u = cand.n_f | cand.n_u, frozenset()
                    else:
                        self.rationalise(Kline(cand.U, cand.n_u), seen, framed.merge(cand.F), depth + 1)

                elif cand.n_o:
                    if grounded(self.d, can(cand.n_o)):
                        cand.n_f, cand.n_o = cand.n_f | cand.n_o, frozenset()
                    else:
                        self.rationalise(Kline(cand.O, cand.n_o), seen, framed.merge(cand.F), depth + 1)

                elif pending == len(self.stores):
                    # stalled at S2~S3 — nothing pending resolves it
                    if cand.n_u:
                        self._halt("stalled at S2: underfit ungrounded", Kline(cand.sig, cand.n_f | cand.n_u))
                    if cand.n_o:
                        self._halt("stalled at S3: overfit ungrounded", Kline(cand.sig, cand.n_f | cand.n_o))
                    break               

        return Rationalisation(self.stores, self.ends, rounds)

    def _reenter(self, entry: Kline, framed: KNode, depth: int) -> list[Candidate]:
        """The reentry's guards, then its base candidate onto the stack."""
        if not entry.adj:
            self._halt("R4: entry kline with no adjacency", entry)
            return []
        if depth >= self.MAX_DEPTH:
            self._halt("bound: recursion depth reached", entry)
            return []
        
        """The candidate list: each overlapping memory kline, individually
        formed ⟨can(entry.adj), nb.adj⟩ — identities witness only themselves.
        Order is §7 strategy: ascending effort, ties in signature then store
        order."""
        nbs = [nb for nb in self.d.overlap(entry.sig) if nb.adj and nb.adj != frozenset([nb.sig]) and not contained(nb.sig, framed)]
        return [next_candidate(nb, entry) for nb in nbs]

def proposals(r: Rationalisation) -> list[Proposal]:
    """Every halt generates one proposal (§5 Definition 15), ordered by graded
    significance — ascending effort, ties in generation order."""
    out = []
    for e in r.ends:
        if e.kline is None:
            continue
        if e.reason.startswith("no candidates") or e.reason.startswith("stalled at S2"):
            kind = "connect"  # fit wanted, nobody to talk to — or underfit starved
        elif e.reason.startswith("bound"):
            kind = "continue"
        else:
            kind = "acquire"
        out.append(Proposal(kind, e.kline, e.effort if e.effort is not None else effort(e.kline)))
    return sorted(out, key=lambda p: p.effort)  # least entropy first — most significant next step
