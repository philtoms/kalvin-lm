"""The Cogitator — the reentry recursion, the model's default runner.

    Rationalise to significance:
        S1 - entry held exactly
        S2 - entry aligned with goal
        S3 - entry and goal aligned on common sub-goal
        S4 - entry not held

    The current implementation rationalises to S2 only.

    rationalise(entry):
      explore(entry.sig, stack=(), frame=()):
        push entry onto stack               # this call's own copy
        while stack:                        # the reentry loop
          E = top
          if no candidates: stall or exchange; pop; continue
          for C in candidates(E):
            F, U, O = decompose(E, C)
            if not settled(U): explore(U, stack, frame)  # recursion into
            if not settled(O): explore(O, stack, frame)  # new state, then
            continue to next candidate                  # forwards
            if settled(U) and settled(O):
              if depth == 1: ground(entry, goal)

    Every reentry is a row in a shared heap; a stack is a list of row
    indices, copied on recursion — cheap, and the caller's stack is never
    looked at again. Frames are copies too: the mint travels with the
    candidate's own descent only. Nothing flows up or across; a path only
    moves forward into new state.

"""

from __future__ import annotations

from dataclasses import dataclass

from kalvin.kline import KNode
from rationalisation.algebra import Kline, can, contained, meet, residue
from rationalisation.domain import Domain
from rationalisation.report import End, Event, Rationalisation, effort


@dataclass
class _Row:
    """One reentry, allocated in the shared heap. parent — the heap index
    of the reentry whose pairing pushed this one (-1 for the opening)."""

    sig: KNode
    depth: int
    parent: int
    psig: KNode | None
    pcsig: KNode | None


@dataclass
class _Parked:
    """A path stalled at S4, parked with the state it stalled in."""

    parent: int
    stack: list
    frame: list[KNode]


class Cogitator:
    """The reentry runner: an entry's signature meets held claims, and each
    unsettled side recurses into new state — its own stack copy, its own
    frames. A pairing grounds when both sides settle; the fit needs no
    frame of its own — it is contained in the candidate's held claim."""

    MAX_DEPTH = 10
    MAX_PAIRINGS = 4096

    def __init__(self, d: Domain) -> None:
        self.d = d
        self.stores: list[Kline] = []

    def _halt(self, reason: str, k: Kline) -> None:
        self.ends.append(End(reason, k, effort(k)))

    def _ground(self, k: Kline) -> None:
        self.d.add(k)
        if k not in self.stores:
            self.stores.append(k)

    def rationalise(self, entry: Kline, goal: Kline | None = None) -> Rationalisation:
        self.stores, self.ends = [], []
        self.events: list[Event] = []
        self.claim: Kline | None = None
        self.joint: list[KNode] = []  # exchange frames — the one horizontal channel
        self.rows: list[_Row] = []  # the shared heap
        self.pairings = 0
        self.bound = False
        self.stalled: dict[KNode, list[_Parked]] = {}

        if goal is None:
            if not entry.adj:
                self._halt("R4: entry kline with no adjacency", entry)
                return Rationalisation(self.stores, self.ends, self.events)
            if entry.sig == can(entry.adj):
                self._ground(entry)  # S1 grounds on receipt
                return Rationalisation(self.stores, self.ends, self.events)

        def add_frame(x: KNode, frame: list[KNode]) -> bool:
            if x and x not in frame:
                frame.append(x)
                return True
            return False

        def framed(x: KNode, frame: list[KNode]) -> bool:
            return (not x
                    or any(contained(x, f) for f in frame)
                    or any(contained(x, j) for j in self.joint))

        def explore(E: KNode, stack: list, _frame: list[KNode], memo: set, depth: int,
                    psig: KNode | None, pcsig: KNode | None) -> None:
            """One reentry into new state. The stack, frame and memo are all copied and built
            forward from here."""
            parent = stack[-1] if stack else -1  # read once, for identity
            base = len(stack)
            me = len(self.rows)
            self.rows.append(_Row(E, depth, parent, psig, pcsig))
            stack = stack + [me]  # the lightweight copy — this path's stack
            grounded = False
            while stack:
                if self.bound or self.claim is not None:
                    return
                top = stack[-1]
                row = self.rows[top]
                E = row.sig
                cands = ([goal] if depth == 1 and goal is not None and base == 0
                         else self._candidates(E, _frame, row.psig, row.pcsig))
                if not cands:  # S4 stalled
                    peers = self.stalled.setdefault(E, [])
                    if any(p.parent != row.parent for p in peers):
                        if not any(contained(E, j) for j in self.joint):
                            self.joint.append(E)
                        self.events.append(Event("exchange", row.depth, E, frame=E))
                        grounded = True
                    else:
                        peers.append(_Parked(row.parent, stack, _frame))
                        self.events.append(Event("stall", row.depth, E, U=E))
                    stack.pop()
                    continue
                for C in cands:
                    frame = _frame.copy()
                    if self.pairings > self.MAX_PAIRINGS:
                        if not self.bound:
                            self._halt("bound: pairing cap reached", entry)
                            self.bound = True
                        return
                    self.pairings += 1
                    cs = C.sig
                    F, U, O = meet(E, cs), residue(E, cs), residue(can(C.adj), E)
                    if F == E:
                        add_frame(F, frame)
                    else:
                        self.events.append(Event("pair", row.depth, E, C, F, U, O))

                    u_ok = framed(U, frame)
                    o_ok = framed(O, frame)
                    if u_ok and o_ok:  # S2 aligned
                        self.events.append(Event("close", row.depth, E, C, F, U, O, frame=E))
                        grounded = True
                        if row.depth == 1 and goal is not None:  # the claim grounds
                            self.claim = Kline(entry.sig, frozenset([goal.sig]))
                            self._ground(self.claim)
                            return  # the claim alone stops the run
                    if not u_ok and U not in memo and row.depth < self.MAX_DEPTH:
                        memo.add(U)  # once per parent, across its candidates
                        explore(U, stack, frame, memo.copy(), row.depth + 1, E, cs)
                        continue

                    if not o_ok and O not in memo and row.depth < self.MAX_DEPTH:
                        memo.add(O)
                        explore(O, stack, frame, memo.copy(), row.depth + 1, E, cs)
                        continue

                if not grounded and not framed(E, frame):
                    self._halt("exhausted: no candidate resolved both sides",
                               Kline(E, frozenset()))
                stack.pop()
            

        explore(entry.sig, [], [], set(), 1, None, None)

        def jointly(x: KNode) -> bool:
            return not x or any(contained(x, j) for j in self.joint)

        for v in self.stalled:  # stalls that never found a witness
            if not jointly(v):
                self._halt("stalled at S4: no candidates", Kline(v, frozenset()))

        return Rationalisation(self.stores, self.ends, self.events, claim=self.claim)

    def _candidates(self, e: KNode, frame: list[KNode], psig: KNode | None = None,
                    pcsig: KNode | None = None) -> list[Kline]:
        """The reference CANDIDATES minus loop control, in insertion order.
        A reentry never re-pairs its origin: its parent entry's and parent
        candidate's signatures are barred — the residual would re-derive
        the decomposition the parent pairing already performed. A signature
        covered on this path has nothing to teach."""
        return [
            c
            for c in self.d
            if c.sig & e  # overlap — no-neighbours never pairs
            and c.adj  # a bare signature offers no construction
            and c.adj != frozenset([c.sig])  # an identity is not a partner
            and c.fingerprint() != e  # like for like
            and c.sig != psig  # the parent entry
            and c.sig != pcsig  # the parent candidate
            and not any(contained(c.sig, f) for f in frame)  # covered on the path
            and not any(contained(c.sig, j) for j in self.joint)  # jointly covered
        ]


def rationalise(d: Domain, entry: Kline, goal: Kline | None = None) -> Rationalisation:
    """The one-shot runner — one Cogitator, one entry."""
    return Cogitator(d).rationalise(entry, goal=goal)
