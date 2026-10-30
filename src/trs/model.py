"""Significance-routed model — reference implementation of docs/trs-significance.md."""

from __future__ import annotations

from dataclasses import dataclass, field
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

def side(k: Kline) -> KNode:
    """The full content of a kline — its signature composed with its adjacency."""
    return k.sig.merge(can(k.adj))


@dataclass
class Row:
    """One reentry pairing. The candidate's content decomposes against the
    entry's content into F (fit), U (underfit — the entry beyond the
    candidate) and O (overfit — the candidate beyond the entry)."""

    depth: int
    entry: Kline
    candidate: Kline | None  # None — the reentry stalled at S4
    F: KNode = KNone
    U: KNode = KNone
    O: KNode = KNone
    frame: KNode = KNone  # what the row framed
    resolved: bool = False
    exchange: bool = False  # the stall whose arrival exchanged
    children: tuple = ()  # the pushed child reentries, U first
    re: "_Reentry | None" = None


@dataclass
class _Reentry:
    """A residual region reentered with its witness nodes."""

    kline: Kline
    depth: int
    parent: "Row | None" = None
    side: str = ""  # "u" | "o" — which residual the reentry came from
    rows: list = field(default_factory=list)
    resolved: bool = False
    popped: bool = False  # the reentry was reached on the agenda
    resolved_by: "Row | None" = None  # the row whose resolution closed the reentry
    stalled: "Row | None" = None


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
    rows: list[Row] = field(default_factory=list)
    path: list[Row] = field(default_factory=list)  # the rows that carried the opening pairing
    claim: Kline | None = None

class Cogitator:
    """The reentry runner. Enter with the ask (and its goal as the first
    candidate), pair the entry with each overlapping memory kline, reenter
    the U and O residuals with the witness nodes they contain, frame the
    fit when the candidate covers the entry, and resolve the row when
    both sides are framed. A resolved row frames its fit and its reentry;
    when the opening pairing resolves, the claim grounds as ⟨ask, {goal}⟩.
    Two stalled reentries on one signature witness each other — the S3
    root exchange."""

    MAX_DEPTH = 10
    MAX_ROWS = 4096

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

    def rationalise(self, entry: Kline, goal: Kline | None = None) -> Rationalisation:
        self.stores, self.ends = [], []
        self.rows: list[Row] = []
        self.open_rows: list[Row] = []
        self.frames: list[KNode] = []
        self.frame_re: list[_Reentry | None] = []  # the reentry that produced each frame
        self.seen: set[tuple[int, int]] = set()
        self.stalls: dict[int, list[_Reentry]] = {}
        self.frame_src: dict[int, list[Row]] = {}
        self.claim: Kline | None = None
        self.goal = goal
        self.goal_row: Row | None = None
        self.agenda: list[_Reentry] = []

        if goal is None:
            if not entry.adj:
                self._halt("R4: entry kline with no adjacency", entry)
                return Rationalisation(self.stores, self.ends, 0, self.rows)
            if entry.sig == can(entry.adj):
                self._admit(entry)  # S1 grounds on receipt
                return Rationalisation(self.stores, self.ends, 0, self.rows)

        self.agenda.append(_Reentry(entry, 1))
        self._run()
        for waiting in self.stalls.values():
            for re_ in waiting:
                if not re_.resolved:
                    self._halt("stalled at S4: no candidates", re_.kline)
        return Rationalisation(
            self.stores, self.ends, len(self.rows), self.rows, self._trace(), self.claim
        )

    def _run(self) -> None:
        while self.agenda and self.claim is None and len(self.rows) < self.MAX_ROWS:
            re_ = self.agenda.pop()
            re_.popped = True
            if re_.resolved or self._covered(re_.kline.sig, self._chain(re_)):
                re_.resolved = True
                continue
            self._reenter(re_)

    def _chain(self, re_: "_Reentry | None") -> set[int]:
        """The reentry and its ancestors — frames they produced never cover
        this subtree's regions."""
        out: set[int] = set()
        while re_ is not None:
            out.add(id(re_))
            re_ = re_.parent.re if re_.parent is not None else None
        return out

    def _covered(self, x: KNode, exclude: set[int] | None = None) -> bool:
        """A region is framed when some frame contains it (⊥ always is) —
        never by a frame its own subtree produced."""
        return not x or any(
            x & f == x
            for f, fr in zip(self.frames, self.frame_re)
            if fr is None or id(fr) not in (exclude or ())
        )

    def _frame(self, x: KNode, *src: Row, joint: bool = False) -> None:
        """Frame content — recording the rows that witnessed it; resolutions
        propagate until still. Only F==E matches and exchanges frame; a
        value already covered elsewhere is still recorded — chains exclude
        frames their own subtree produced, so suppressed frames are
        load-bearing for those rows. A joint frame (the exchange's mutual
        witness) belongs to no subtree."""
        if not x:
            return
        if src:
            self.frame_src.setdefault(int(x), []).extend(src)
        if x in self.frames:
            return
        self.frames.append(x)
        self.frame_re.append(None if joint or not src else src[0].re)
        changed = True
        while changed and self.claim is None:
            changed = False
            for row in list(self.open_rows):
                if self._satisfied(row):
                    self._resolve(row)
                    changed = True

    def _side_ok(self, row: Row, region: KNode, side: str) -> bool:
        """A side is satisfied when empty, framed outside its subtree, or
        its reentry resolved — a stalled reentry has no rows; only the
        exchange frame closes it."""
        if not region or self._covered(region, self._chain(row.re)):
            return True
        return any(
            ch.side == side and ch.resolved and ch.stalled is None for ch in row.children
        )

    def _satisfied(self, row: Row) -> bool:
        # an entry is resolved when all of its components — F, U and O — are
        # framed; the opener's fit is the claim's shared content — given
        fit = row is self.goal_row or self._covered(row.F, self._chain(row.re))
        return fit and self._side_ok(row, row.U, "u") and self._side_ok(row, row.O, "o")

    def _after(self, row: "Row | None") -> None:
        """A side was satisfied — the parent row may now resolve."""
        if row is None or row.resolved or self.claim is not None:
            return
        if self._satisfied(row):
            self._resolve(row)

    def _resolve(self, row: Row) -> None:
        row.resolved = True
        if row in self.open_rows:
            self.open_rows.remove(row)
        if row is self.goal_row:
            self.claim = Kline(row.entry.sig, frozenset([self.goal.sig]))
            self._admit(self.claim)  # the opening pairing resolved — the claim grounds
        re_ = row.re
        if re_ is not None and not re_.resolved:
            re_.resolved = True
            re_.resolved_by = row
            self._after(re_.parent)  # the side is satisfied — propagate backwards

    def _reenter(self, re_: _Reentry) -> None:
        k = re_.kline
        if not k.adj:
            self._halt("R4: entry kline with no adjacency", k)
            return
        if re_.depth > self.MAX_DEPTH:
            self._halt("bound: recursion depth reached", k)
            return
        cands = [
            c for c in self.d
            if c.sig & k.sig and c.adj and c.adj != frozenset([c.sig]) and not self._covered(c.sig)
        ]
        if re_.depth == 1 and self.goal is not None:
            # the opening pairing is the ask==goal claim — the first and only
            # candidate at depth 1; the residuals do the exploring
            self._pair(re_, self.goal)
            return
        if not cands and not re_.rows:
            row = Row(re_.depth, k, None, re=re_)
            re_.stalled = row
            self.rows.append(row)
            self.stalls.setdefault(int(k.sig), []).append(re_)
            self._exchange(int(k.sig))
            return
        for c in sorted(cands, key=lambda c: mu(side(c))):
            # least content first — the scaffolding before the canons
            if self._in_path(re_, c):
                continue  # a candidate cannot reappear in the path it was decomposed along
            if self.claim is not None:
                return
            fp = (k.fingerprint(), c.fingerprint())
            if fp in self.seen:
                continue
            self.seen.add(fp)
            self._pair(re_, c)

    def _pair(self, re_: _Reentry, c: Kline) -> None:
        k = re_.kline
        s_e, s_c = side(k), side(c)
        f = meet(s_e, s_c)
        row = Row(re_.depth, k, c, F=f, U=residue(s_e, s_c), O=residue(s_c, s_e), re=re_)
        if re_.depth == 1 and c == self.goal:
            self.goal_row = row
        re_.rows.append(row)
        self.rows.append(row)
        self.open_rows.append(row)
        if f == s_e:  # the candidate covers the entry — the fit frames
            row.frame = f if not row.frame else row.frame.merge(f)
            self._frame(f, row)
        o_child = u_child = None
        if not self._side_ok(row, row.O, "o"):
            o_child = self._child(re_.depth + 1, row.O, "o", (n for n in c.adj if contained(n, row.O)), row)
        if not self._side_ok(row, row.U, "u"):
            u_child = self._child(re_.depth + 1, row.U, "u", (n for n in k.adj if contained(n, row.U)), row)
        row.children = tuple(ch for ch in (u_child, o_child) if ch is not None)
        if self._satisfied(row):
            self._resolve(row)

    def _in_path(self, re_: "_Reentry", c: Kline) -> bool:
        """Is the candidate in the reentry's path — the content flow from the
        opening pairing down? An o-side child's region came from its parent
        row's candidate, so that candidate is in its path; a u-side child
        inherits only the parent's path. A candidate cannot reappear in the
        path it was decomposed along."""
        sig = int(c.sig)
        child = re_
        while child is not None and child.parent is not None:
            if child.side == "o" and child.parent.candidate is not None:
                if int(child.parent.candidate.sig) == sig:
                    return True
            nxt = child.parent.re
            if nxt is None:
                break
            child = nxt
        return False

    def _child(self, depth: int, region: KNode, side: str, nodes, parent: Row) -> "_Reentry | None":
        """Reenter the residual region carrying the witness nodes it
        contains. LIFO: the U-side child, pushed last, leads."""
        if not region:
            return None
        re_ = _Reentry(Kline(region, frozenset(nodes)), depth, parent, side)
        self.agenda.append(re_)
        return re_

    def _exchange(self, sig: int) -> None:
        """An overfit and an underfit reentry stalled at S4 on one
        signature witness each other: the shared content grounds and both
        chains resolve."""
        waiting = [r for r in self.stalls.get(sig, ()) if not r.resolved]
        while len(waiting) >= 2 and self.claim is None:
            a = waiting[0]
            b = next(
                (r for r in waiting[1:] if r.side != a.side and r.parent is not a.parent),
                None,
            )
            if b is None:
                return
            self._admit(Kline(a.kline.sig, a.kline.adj | b.kline.adj))
            a.resolved = b.resolved = True
            if b.stalled is not None:
                b.stalled.resolved = True
                b.stalled.exchange = True  # the arriving stall carries the exchange
                b.stalled.frame = a.kline.sig if not b.stalled.frame else b.stalled.frame.merge(a.kline.sig)
            if a.stalled is not None:
                a.stalled.resolved = True
            self._frame(a.kline.sig, a.stalled, b.stalled)
            # the root exchange: the waiting fits witness each other — what
            # for had, the question's remnant answered by the verb
            if a.parent is not None and b.parent is not None:
                self._frame(a.parent.F, a.parent, b.parent, joint=True)
                self._frame(b.parent.F, a.parent, b.parent, joint=True)
            waiting = [r for r in self.stalls.get(sig, ()) if not r.resolved]

    def _trace(self) -> list[Row]:
        """The optimal path: the opening pairing and, for each side it rests
        on, the one chain that carried it — the child's resolving row, the
        exchange pair, or the source of the one frame that covers the
        region. One row per reentry — paths stay orthogonal. In generation
        order."""
        if self.claim is None or self.goal_row is None:
            return []
        keep: dict[int, Row] = {}
        tabled: dict[int, int] = {}  # reentry id -> the row id holding its slot
        sticky: set[int] = set()  # reentries whose holders are never displaced
        stack: list[Row] = []
        # the rows feeding an exchanged stall hold their reentries' slots —
        # the S3 path is the path, not the shortcut that resolved first
        for waiting in self.stalls.values():
            for re_ in waiting:
                if (
                    re_.stalled is not None
                    and re_.stalled.resolved
                    and re_.parent is not None
                    and re_.parent.re is not None
                ):
                    tabled[id(re_.parent.re)] = id(re_.parent)
                    sticky.add(id(re_.parent.re))  # exchange feeds are never displaced
                    stack.append(re_.parent)
        stack.append(self.goal_row)
        while stack:
            row = stack.pop()
            if id(row) in keep:
                continue
            re_ = row.re
            if re_ is not None and tabled.get(id(re_), id(row)) != id(row):
                continue  # this entry already carries its row — a second candidate is another path
            keep[id(row)] = row
            if re_ is not None:
                tabled[id(re_)] = id(row)
            if re_ is not None and re_.parent is not None:
                stack.append(re_.parent)  # the row that pushed this reentry is on the path
            for side, region in (("u", row.U), ("o", row.O)):
                if not region:
                    continue
                child = next((ch for ch in row.children if ch.side == side), None)
                if child is not None and child.resolved_by is not None:
                    stack.append(child.resolved_by)  # the resolution chain
                    continue
                if child is not None and child.stalled is not None and child.stalled.resolved:
                    pair = self.frame_src.get(int(child.kline.sig), [])  # the exchange pair
                    stack.extend(pair[:2])
                    continue
                chain = self._chain(row.re)
                best = None
                for i, (f, fr) in enumerate(zip(self.frames, self.frame_re)):
                    if fr is not None and id(fr) in chain:
                        continue
                    if region & f == region and (best is None or (mu(f), i) < best):
                        best = (mu(f), i, f)
                if best is not None:
                    srcs = self.frame_src.get(int(best[2]), [])
                    if srcs:
                        stack.append(srcs[0])
        # repair: every tabled entry is claimed by a tabled row's step — a
        # floating row's pusher takes its reentry's slot
        for _ in range(len(keep) + 1):
            claimed: set[int] = set()
            for r in keep.values():
                for ch in r.children:
                    if (ch.stalled is not None and id(ch.stalled) in keep) or any(
                        id(x) in keep for x in ch.rows
                    ):
                        claimed.add(id(ch))
            floating = next(
                (
                    r
                    for r in keep.values()
                    if r is not self.goal_row
                    and r.re is not None
                    and id(r.re) not in claimed
                ),
                None,
            )
            if floating is None:
                break
            pusher = floating.re.parent
            if (
                pusher is None
                or pusher is self.goal_row
                or pusher.re is None
                or tabled.get(id(pusher.re)) == id(pusher)
                or (id(pusher.re) in sticky and tabled.get(id(pusher.re)) != id(pusher))
            ):
                del keep[id(floating)]  # cannot connect it without displacing a sticky holder — drop
                continue
            old = tabled.get(id(pusher.re))
            if old is not None and old in keep:
                del keep[old]
            keep[id(pusher)] = pusher
            tabled[id(pusher.re)] = id(pusher)
        return [r for r in self.rows if id(r) in keep]

    def _pair_effort(self, k: Kline, c: Kline) -> Fraction:
        s_e, s_c = side(k), side(c)
        u, o = residue(s_e, s_c), residue(s_c, s_e)
        t = mu(s_e | s_c)
        return Fraction(mu(u) + BETA * mu(o), t) if t else Fraction(0)


def proposals(r: Rationalisation) -> list[Proposal]:
    """Every halt generates one proposal (§5 Definition 15), ordered by graded
    significance — ascending effort, ties in generation order."""
    out = []
    for e in r.ends:
        if e.kline is None:
            continue
        if e.reason.startswith("stalled") or "no candidates" in e.reason:
            kind = "connect"  # fit wanted, nobody to talk to — the sides starved
        elif e.reason.startswith("bound"):
            kind = "continue"
        else:
            kind = "acquire"
        out.append(Proposal(kind, e.kline, e.effort if e.effort is not None else effort(e.kline)))
    return sorted(out, key=lambda p: p.effort)  # least entropy first — most significant next step
