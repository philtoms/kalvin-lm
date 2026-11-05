"""The ReferenceCogitator — RETIRED HISTORY. The pre-drain baseline of
the rationalisation model, now out of date and ignored: the Cogitator
(rationalisation/cogitator.py) is the model (docs/rationalisation.md
§2). Retained only because old probes and tests still compare against
it; do not build on it. The shared algebra, domain, and report live in
their own modules; what remains here is the retired search itself: a deterministic, bounded, depth-first run in which
unconstructed klines meet held klines' claims and receive their offered
decompositions, until the claim grounds or the run halts. The scripts
are the only guarantee that an ask can be framed as a goal; the model
creates no contrivance to force a resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

from kalvin.kline import KNode, KNone
from rationalisation.algebra import Kline, can, contained, effort, meet, residue
from rationalisation.domain import Domain
from rationalisation.report import End, Event, Rationalisation


@dataclass(eq=False)
class _Pairing:
    """One reentry pairing — the reference's internal working record. The
    candidate's content decomposes against the entry's content into F
    (fit), U (underfit — the entry beyond the candidate) and O (overfit —
    the candidate beyond the entry). Identity equality — pairings are
    occurrences, never values. The run's public trace is Event
    (rationalisation/report.py)."""

    depth: int
    entry: Kline
    candidate: Kline | None  # None — the reentry stalled at S4
    F: KNode = KNone
    U: KNode = KNone
    O: KNode = KNone
    frame: KNode = KNone  # what the pairing framed
    supports: tuple = ()  # frames the pairing leans on
    resolved: bool = False
    exchange: bool = False  # the stall whose arrival exchanged
    children: tuple = ()  # the pushed child reentries, U first
    re: "_Reentry | None" = None
    bp: tuple = ()  # the pairings the resolution back-propagated to
    known: bool = False  # resolved without exploring — already known


@dataclass(eq=False)
class _Reentry:
    """A residual region reentered with its witness nodes."""

    kline: Kline
    depth: int
    parent: "_Pairing | None" = None
    side: str = ""  # "u" | "o" — which residual the reentry came from
    rows: list = field(default_factory=list)
    resolved: bool = False
    popped: bool = False  # the reentry was reached on the agenda
    resolved_by: "_Pairing | None" = None  # the pairing whose resolution closed the reentry
    stalled: "_Pairing | None" = None


class ReferenceCogitator:
    """The reference implementation of the reentry runner — retained as the
    baseline for comparison; its successor derives from the simple reentry
    loop (dev/probes/probe_simple_reentry.py).

    Enter with the ask (and its goal as the first
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

    def _emit(self, kind: str, row: "_Pairing") -> None:
        """One reentry event — the run's trace, in order, never read."""
        self.events.append(Event(
            kind, row.depth, row.entry.sig, row.candidate,
            row.F, row.U, row.O, row.frame,
        ))

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
        self.rows: list[_Pairing] = []
        self.open_rows: list[_Pairing] = []
        self.frames: list[KNode] = []
        self.frame_re: list[_Reentry | None] = []  # the reentry that produced each frame
        self.seen: set[tuple[int, int]] = set()
        self.stalls: dict[int, list[_Reentry]] = {}
        self.frame_src: dict[int, list[_Pairing]] = {}
        self.claim: Kline | None = None
        self.goal = goal
        self.goal_row: _Pairing | None = None
        self.agenda: list[_Reentry] = []
        self.events: list[tuple[_Pairing, str]] = []  # (row, "pair"|"wait"|"close") — noted in order, never read
        self.rframes: list[tuple[KNode, _Pairing]] = []  # resolution frames — record, never settle

        if goal is None:
            if not entry.adj:
                self._halt("R4: entry kline with no adjacency", entry)
                return Rationalisation(self.stores, self.ends, self.rows)
            if entry.sig == can(entry.adj):
                self._admit(entry)  # S1 grounds on receipt
                return Rationalisation(self.stores, self.ends, self.rows)

        self.agenda.append(_Reentry(entry, 1))
        self._run()
        for waiting in self.stalls.values():
            for re_ in waiting:
                if not re_.resolved:
                    self._halt("stalled at S4: no candidates", re_.kline)
        return Rationalisation(
            self.stores, self.ends, self.events, claim=self.claim,
        )

    def _run(self) -> None:
        while self.agenda and self.claim is None and len(self.rows) < self.MAX_ROWS:
            re_ = self.agenda.pop()
            re_.popped = True
            if re_.resolved or self._onpath(re_.kline.sig, self._chain(re_)):
                self._known(re_)  # framed by a proper ancestor — already known
                continue
            self._reenter(re_)

    def _chain(self, re_: "_Reentry | None") -> set[int]:
        """The ancestor rows of the reentry — frames they produced cover this
        subtree's regions; the reentry's own rows and its siblings' never do
        (no self-satisfaction through sibling pairings)."""
        out: set[int] = set()
        if re_ is None:
            return out
        row = re_.parent
        while row is not None:
            out.add(id(row))
            nxt = row.re
            row = nxt.parent if nxt is not None else None
        return out

    def _onpath(self, region: KNode, chain: set[int]) -> bool:
        """Covered by a frame a proper ancestor row produced — or a joint/
        exchange frame, which belongs to no subtree."""
        return not region or any(
            region & f == region
            for f, fr in zip(self.frames, self.frame_re)
            if fr is None or id(fr) in chain
        )

    def _covered(self, x: KNode) -> bool:
        """A region is framed when some frame contains it (⊥ always is)."""
        return not x or any(x & f == x for f in self.frames)

    def _frame(self, x: KNode, *src: _Pairing, joint: bool = False) -> None:
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
        self.frame_re.append(None if joint or not src else src[0])
        changed = True
        while changed and self.claim is None:
            changed = False
            for row in list(self.open_rows):
                if self._satisfied(row):
                    self._resolve(row)
                    changed = True

    def _side_ok(self, row: _Pairing, region: KNode, side: str) -> bool:
        """A side is settled when empty, its reentry resolved — a stalled
        reentry only through the exchange — an ancestor row framed it on
        the way down, or (the o-side) the offered region is canonically
        held: a witnessed signature needs no re-derivation — the goal
        grounds itself."""
        if not region:
            return True
        kids = [ch for ch in row.children if ch.side == side]
        if kids and all(ch.resolved for ch in kids):
            return True
        if side == "o" and self.d.witness(region):
            return True
        return self._onpath(region, self._chain(row.re))

    def _satisfied(self, row: _Pairing) -> bool:
        # an entry is resolved when both residuals are settled — the fit is
        # the agreement between entry and candidate, not something to prove
        return self._side_ok(row, row.U, "u") and self._side_ok(row, row.O, "o")

    def _after(self, row: "_Pairing | None") -> None:
        """A side was satisfied — the parent row may now resolve."""
        if row is None or row.resolved or self.claim is not None:
            return
        if self._satisfied(row):
            self._resolve(row)
        else:
            self._emit("wait", row)  # a side settled, the pairing still waits — stalled

    def _resolve(self, row: _Pairing) -> None:
        row.resolved = True
        self._emit("close", row)
        if row in self.open_rows:
            self.open_rows.remove(row)
        if row is self.goal_row:
            self.claim = Kline(row.entry.sig, frozenset([self.goal.sig]))
            self._admit(self.claim)  # the opening pairing resolved — the claim grounds
            if row.O:
                self.rframes.append((row.O, row))  # the claim's goal side frames at grounding
        re_ = row.re
        if re_ is not None and not re_.resolved:
            re_.resolved = True
            re_.resolved_by = row
            self.rframes.append((re_.kline.sig, row))  # the resolution frames its region — a record
            if re_.parent is not None:
                row.bp = (re_.parent,)
                self._after(re_.parent)  # the side is satisfied — propagate backwards

    def _known(self, re_: _Reentry) -> None:
        """A reentry resolved without exploring — its region is already
        framed or every candidate is proven. It still traces: a known row,
        so the causality it carries stays visible."""
        if re_.resolved:
            return
        row = _Pairing(re_.depth, re_.kline, None, re=re_)
        row.known = True
        self._emit("known", row)
        row.resolved = True
        re_.rows.append(row)
        self.rows.append(row)
        re_.resolved = True
        re_.resolved_by = row
        self.rframes.append((re_.kline.sig, row))
        if re_.parent is not None:
            row.bp = (re_.parent,)
            self._after(re_.parent)

    def _reenter(self, re_: _Reentry) -> None:
        k = re_.kline
        if not k.adj:
            self._halt("R4: entry kline with no adjacency", k)
            return
        if re_.depth > self.MAX_DEPTH:
            self._halt("bound: recursion depth reached", k)
            return
        cands = [
            c
            for c in self.d
            if c.sig & k.sig
            and c.adj
            and c.adj != frozenset([c.sig])
            and c.fingerprint() != k.sig  # like for like — the candidate offers exactly the entry
            and not self._covered(c.sig)
        ]
        if re_.depth == 1 and self.goal is not None:
            # the opening pairing is the ask==goal claim — the first and only
            # candidate at depth 1; the residuals do the exploring
            for ch in self._pair(re_, self.goal):
                self.agenda.append(ch)
            return
        if not cands and not re_.rows:
            row = _Pairing(re_.depth, k, None, U=k.sig, re=re_)  # the whole content is underfit
            self._emit("stall", row)
            re_.stalled = row
            self.rows.append(row)
            self.stalls.setdefault(int(k.sig), []).append(re_)
            self._exchange(int(k.sig))
            return
        batches: list[list[_Reentry]] = []
        turned_away = False  # a candidate refused while the region stood unproven
        for c in cands:
            # closers first — the candidate that frames the whole entry leads
            # (the resolution determinism); then least content — the
            # scaffolding before the canons
            e = k.sig
            cs = c.sig
            proven = (
                e & cs == e and e in self.frames
                and self._pre_settled(residue(e, cs), re_)
                and self._pre_settled(residue(can(c.adj), e), re_)
            )
            if self._in_path(re_, c):
                # a candidate cannot reappear in the path it was decomposed
                # along — but a proven one carried its evidence already
                if not proven:
                    turned_away = True
                continue
            if self.claim is not None:
                break
            fp = (id(re_), k.fingerprint(), c.fingerprint())
            if fp in self.seen:
                continue  # each reentry is its own path — twins explore their own
            if proven:
                continue  # a proven fit with no work left — re-proving adds nothing
            self.seen.add(fp)
            batches.append(self._pair(re_, c))
        if self.claim is not None:
            return
        if not batches and not re_.rows and not re_.resolved:
            # every candidate was skipped. All proven — the region stands,
            # nothing to explore, nothing to learn: known. Any turned away
            # by a path bar — exhaustion is not evidence: the reentry halts
            # rather than resolve on a policy bar alone.
            if turned_away:
                self._halt("exhausted: candidate barred without proof", k)
                return
            self._known(re_)
            return
        for batch in reversed(batches):
            # the least candidate's subtree leads — the reductive chain runs to
            # completion before the next candidate explores. Batches are
            # independent: candidates may explore in parallel.
            for ch in batch:
                self.agenda.append(ch)

    def _pre_settled(self, region: KNode, re_: _Reentry) -> bool:
        """Would the region settle without children — empty, canonically
        witnessed, or framed by a proper ancestor row."""
        return (not region) or bool(self.d.witness(region)) or self._onpath(region, self._chain(re_))

    def _pair(self, re_: _Reentry, c: Kline) -> list[_Reentry]:
        k = re_.kline
        e = k.sig
        cs = c.sig  # the claim
        ca = can(c.adj)  # the decomposition, composed
        # the entry meets the claim; the candidate's construction offers
        # beyond it — the offering is the o-side. The claim's own
        # extension is not the entry's business: the claim fits, the
        # construction offers
        row = _Pairing(
            re_.depth, k, c,
            F=meet(e, cs), U=residue(e, cs),
            O=residue(ca, e), re=re_,
        )
        self._emit("pair", row)
        if re_.depth == 1 and c == self.goal:
            self.goal_row = row
        re_.rows.append(row)
        self.rows.append(row)
        self.open_rows.append(row)
        pushed: list[_Reentry] = []
        if not self._side_ok(row, row.O, "o"):
            ch = self._child(
                re_.depth + 1, row.O, "o",
                (n for n in c.adj if contained(n, row.O)), row,
            )
            if ch is not None:
                pushed.append(ch)
        if not self._side_ok(row, row.U, "u"):
            ch = self._child(re_.depth + 1, row.U, "u", (n for n in k.adj if contained(n, row.U)), row)
            if ch is not None:
                pushed.append(ch)
        row.children = tuple(reversed(pushed))  # U first, then the offering
        if self._satisfied(row):
            self._resolve(row)  # before the frame mints — the row closes, then its cascade
        if row.F == e:  # E == F — the claim covers the entry
            row.frame = e if not row.frame else row.frame.merge(e)
            self._frame(e, row)
            self._emit("frame", row)
        return pushed

    def _in_path(self, re_: _Reentry, c: Kline) -> bool:
        """Is the candidate in the reentry's path — the entry path and the
        content flow from the opening pairing down? A candidate must not
        be an ancestor entry kline: MHALL -> HALL -> MHALL re-derives what
        the decomposition already established. The comparison is the
        kline — signature and adjacency together — never the signature
        alone: a held kline sharing only a signature with an ancestor
        entry makes a different claim and may pair. Beyond that, an
        o-side child's region came from its parent row's candidate, so
        that candidate enters its path; a u-side child inherits only the
        parent's path. A reentry may re-meet its own origin — the
        candidate it was isolated from — but no other candidate in its
        strict path."""
        sig = int(c.sig)
        chain = re_
        while chain is not None:  # the entry path — no candidate may re-derive an ancestor entry
            if int(chain.kline.sig) == sig:
                return True
            chain = chain.parent.re if chain.parent is not None else None
        child = re_
        first = True
        while child is not None and child.parent is not None:
            if child.side == "o" and child.parent.candidate is not None:
                if not first and int(child.parent.candidate.sig) == sig:
                    return True
            first = False
            nxt = child.parent.re
            if nxt is None:
                break
            child = nxt
        return False

    def _child(self, depth: int, region: KNode, side_: str, nodes, parent: _Pairing) -> "_Reentry | None":
        """The residual region as a reentry carrying the witness nodes it
        contains. Not pushed — the caller orders the agenda (LIFO: the
        U-side child leads)."""
        if not region:
            return None
        return _Reentry(Kline(region, frozenset(nodes)), depth, parent, side_)

    def _exchange(self, sig: int) -> None:
        """Reentries stalled at S4 on one signature witness each other:
        the shared C frames and both chains resolve by back propagation —
        when stalled, C is a U or an O by definition. The exchange needs no
        store: the frame is the record, memory learns nothing here."""
        waiting = [r for r in self.stalls.get(sig, ()) if not r.resolved]
        while len(waiting) >= 2 and self.claim is None:
            a = waiting[0]
            b = next((r for r in waiting[1:] if r.parent is not a.parent), None)
            if b is None:
                return
            a.resolved = b.resolved = True
            if b.stalled is not None:
                b.stalled.resolved = True
                b.stalled.exchange = True  # the arriving stall carries the exchange
                b.stalled.O = b.kline.sig  # the matched content arrives as overfit
                b.stalled.frame = a.kline.sig if not b.stalled.frame else b.stalled.frame.merge(a.kline.sig)
                self._emit("exchange", b.stalled)
                b.stalled.bp = tuple(p for p in (a.parent, b.parent) if p is not None)
            if a.stalled is not None:
                a.stalled.resolved = True  # resolved through the exchange — silently
            self._frame(a.kline.sig, a.stalled, b.stalled, joint=True)
            waiting = [r for r in self.stalls.get(sig, ()) if not r.resolved]


def rationalise(d: Domain, entry: Kline, goal: Kline | None = None) -> Rationalisation:
    """The reference one-shot — one ReferenceCogitator, one entry."""
    return ReferenceCogitator(d).rationalise(entry, goal=goal)
