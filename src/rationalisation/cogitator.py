"""The Cogitator — the reentry recursion, the model's default runner.

    Rationalise to significance:
        S1 - entry held exactly
        S2 - entry aligned with goal
        S3 - entry and goal aligned on common sub-goal
        S4 - entry not held

    The current implementation rationalises to S2 only.

    rationalise(entry):
      explore(entry.sig, stack=()):
        push entry onto stack               # this call's own copy
        while stack:                        # the reentry loop
          E = top
          if no candidates: stall or exchange; pop; continue
          for C in candidates(E):
            F, U, O = decompose(E, C)
            if not settled(U): explore(U, stack)  # recursion into
            if not settled(O): explore(O, stack)  # new state, then
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

from dataclasses import dataclass, field

from kalvin.kline import KNode
from rationalisation.algebra import Kline, can, contained, meet, residue
from rationalisation.domain import Domain
from rationalisation.report import End, Event, Rationalisation, effort


@dataclass
class _Entry:
    """One reentry, allocated in the shared heap. parent — the heap index
    of the reentry whose pairing pushed this one (-1 for the opening)."""

    sig: KNode
    depth: int
    parent: int
    memo: set = field(default_factory=set)  # sigs this row has forked — once per parent

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
        self.claim: Kline | None = None
        self.pairings = 0
        self.bound = False
        self.stalled: dict[KNode, int] = {}
        self.frame: list[KNode] = []
        self.events: list[Event] = []

        if goal is None:
            if not entry.adj:
                self._halt("R4: entry kline with no adjacency", entry)
                return Rationalisation(self.stores, self.ends, [])
            if entry.sig == can(entry.adj):
                self._ground(entry)  # S1 grounds on receipt
                return Rationalisation(self.stores, self.ends, [])

        def frame(x: KNode) -> bool:
            if x and x not in self.frame:
                self.frame.append(x)
                return True
            return False

        def framed(x: KNode) -> bool:
            return not x or any(contained(x, f) for f in self.frame)

        def explore(E: KNode, stack: list[_Entry], events: list[Event], psig: KNode | None, pcsig: KNode | None) -> None:
            """One reentry into new state. The stack is copied and built forward from here."""
            parent = len(stack)  # read once, for identity
            depth = parent + 1
            cands = ([goal] if depth == 1 and goal is not None
                        else self._candidates(E, psig, pcsig))
            if not cands:  # S4 stalled
                if E in self.stalled and self.stalled[E] != parent:
                    frame(E) # S3 align
                    events.append(Event("exchange", depth, E, frame=E))
                    self.stalled.pop(E)
                else:
                    self.stalled[E] = parent
                    events.append(Event("stall", depth, E))
                return
            
            while cands:
                _events = events.copy()
                _stack = stack.copy()
                _stack.append(_Entry(E, depth, parent))
                while _stack:
                    if self.bound or self.claim is not None:
                        return
                    if self.pairings > self.MAX_PAIRINGS:
                        if not self.bound:
                            self._halt("bound: pairing cap reached", entry)
                            self.bound = True
                        return
                    self.pairings += 1
                    top = _stack[-1]
                    E = top.sig
                    depth = top.depth
                    memo = top.memo
                    parent = top.parent
                    C = cands[-1]
                    cs = C.sig
                    F, U, O = meet(E, cs), residue(E, cs), residue(can(C.adj), E)
                    u_ok = framed(U)
                    o_ok = framed(O)
                    if u_ok and o_ok:  # S2 aligned
                        frame(E)
                        _events.append(Event("close", depth, E, C, F, U, O, frame=E))
                        if depth == 1 and goal is not None and goal.sig == C.sig:  # the claim grounds
                            self.claim = Kline(entry.sig, frozenset([goal.sig]))
                            self._ground(self.claim)
                            self.events = _events
                            return  # the claim alone stops the run
                    else:
                        _events.append(Event("pair", depth, E, C, F, U, O))

                    if not u_ok and U not in memo and depth < self.MAX_DEPTH:
                        memo.add(U)  # once per parent — the row, not the fork
                        explore(U, _stack, _events, E, cs)
                        continue

                    if not o_ok and O not in memo and depth < self.MAX_DEPTH:
                        memo.add(O)
                        explore(O, _stack, _events, E, cs)
                        continue

                    _stack.pop()

                cands.pop()
                if not cands:
                    self._halt("exhausted: no candidate resolved both sides",
                            Kline(E, frozenset()))
            

        explore(entry.sig, [], [], None, None)

        for e in self.stalled:  # stalls that never found a witness
            self._halt("stalled at S4: no candidates", Kline(e, frozenset()))

        return Rationalisation(self.stores, self.ends, self.events, claim=self.claim)

    def _candidates(self, e: KNode, psig: KNode | None = None,
                    pcsig: KNode | None = None) -> list[Kline]:
        """The reference CANDIDATES minus loop control, in insertion order.
        A reentry never re-pairs its origin: its parent entry's and parent
        candidate's signatures are barred — the residual would re-derive
        the decomposition the parent pairing already performed. A signature
        covered on this path has nothing to teach."""
        cands = [
            c
            for c in self.d
            if c.sig & e  # overlap — no-neighbours never pairs
            and c.adj  # a bare signature offers no construction
            and c.adj != frozenset([c.sig])  # an identity is not a partner
            and c.fingerprint() != e  # like for like
            and c.sig != psig  # the parent entry
            and c.sig != pcsig  # the parent candidate
            and not any(contained(c.sig, f) for f in self.frame)  # covered on the path
        ]
        return cands
        # return list(reversed(cands))

def rationalise(d: Domain, entry: Kline, goal: Kline | None = None) -> Rationalisation:
    """The one-shot runner — one Cogitator, one entry."""
    return Cogitator(d).rationalise(entry, goal=goal)
