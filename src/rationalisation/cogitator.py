"""The Cogitator — the simple reentry loop, the model's default runner.
    
    Rationalise to significance:
        S1 - entry held exactly
        S2 - entry aligned with goal
        S3 - entry and goal aligned on common sub-goal
        S4 - entry not held

    The current implementation rationalises to S2 only.

    rationalise(entry):
      entries.push(entry.sig, candidates(entry))
      while entries:
        E, cands = entries.top()
        if not cands: # S4 stalled
          if E is S3 aligned:
            frame(E)
          entries.pop()
          continue

        for C in cands:
          F, U, O = decompose(E, C)
          if F == E: frame(F)
          if framed(U) or U.empty: U_resolved = True else entries.push(U, candidates(U))
          if framed(O) or O.empty: O_resolved = True else entries.push(O, candidates(O))
          if U_resolved and O_resolved: # S2 aligned
            ground(E, C)
            entries.pop()

"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from kalvin.kline import KNode, KNone
from rationalisation.algebra import Kline, can, contained, meet, residue
from rationalisation.domain import Domain
from rationalisation.report import End, Event, Rationalisation, effort


class Cogitator:
    """The reentry runner: an entry's signature meets held claims, its
    underfit and overfit reenter, and pairings ground when both
    sides resolve. The fit needs no frame of its own — it is contained in
    the candidate's held claim, so the pairing itself is the witness."""

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
        self.frame: list[KNode] = []
        self.pairings = 0
        self.stalled: dict[KNode, list[int]] = {}
        stack: list[_Entry] = []
        seq = itertools.count(1)

        if goal is None: 
            if not entry.adj:
                self._halt("R4: entry kline with no adjacency", entry)
                return Rationalisation(self.stores, self.ends, self.events)
            if entry.sig == can(entry.adj):
                self._ground(entry)  # S1 grounds on receipt
                return Rationalisation(self.stores, self.ends, self.events)

        def framed(x: KNode) -> bool:
            return not x or any(contained(x, f) for f in self.frame)

        def frame(x: KNode) -> None:
            if x and x not in self.frame:
                self.frame.append(x)

        def push(sig: KNode, depth: int, memo: set, parent: int,
                 psig: KNode | None = None, pcsig: KNode | None = None) -> None:
            memo.add(sig)  # once per parent, across its for-runs
            cands = [goal] if depth == 1 and goal is not None else self._candidates(sig, psig, pcsig)
            stack.append(_Entry(sig, cands, depth, set(), next(seq), parent))

        push(entry.sig, 1, set(), 0)

        while stack:
            if self.pairings > self.MAX_PAIRINGS:
                self._halt("bound: pairing cap reached", entry)
                break
            en = stack[-1]
            E, cands, depth, memo = en.sig, en.cands, en.depth, en.memo

            if not cands:  # S4 stalled
                peers = self.stalled.setdefault(E, [])
                if any(q != en.parent for q in peers):
                    frame(E)  # S3 aligned
                    self.events.append(Event("exchange", depth, E, frame=E))
                else:
                    peers.append(en.parent)
                    self.events.append(Event("stall", depth, E, U=E))
                stack.pop()
                continue

            new_push = False
            grounded = False
            for C in cands:
                self.pairings += 1
                cs = C.sig
                F, U, O = meet(E, cs), residue(E, cs), residue(can(C.adj), E)
                if F == E:
                    frame(F)
                    self.events.append(Event("pair", depth, E, C, F, U, O, frame=F))
                else:
                    self.events.append(Event("pair", depth, E, C, F, U, O))
                U_res = framed(U)
                O_res = framed(O)
                if not U_res and U not in memo and depth < self.MAX_DEPTH:  # C5, C7
                    push(U, depth + 1, memo, en.seq, E, cs)
                    new_push = True
                if not O_res and O not in memo and depth < self.MAX_DEPTH:
                    push(O, depth + 1, memo, en.seq, E, cs)
                    new_push = True
                if U_res and O_res:  # S2 aligned
                    frame(E)
                    self.events.append(Event("close", depth, E, C, F, U, O, frame=E))
                    stack[:] = [x for x in stack if x is not en]  # children stay
                    grounded = True
                    if depth == 1 and goal is not None:  # the claim grounds
                        self.claim = Kline(entry.sig, frozenset([goal.sig]))
                        self._ground(self.claim)
                        # break  # the claim alone stops the run — an inner ground never does
            if self.claim is not None:
                break 
            if grounded:
                continue
            if not new_push:  # nothing left to spawn, nothing settled it
                if not framed(E):
                    self._halt("exhausted: no candidate resolved both sides",
                               Kline(E, frozenset()))
                stack.pop()

        for v in self.stalled:  # stalls that never found a witness
            if not framed(v):
                self._halt("stalled at S4: no candidates", Kline(v, frozenset()))

        return Rationalisation(self.stores, self.ends, self.events, claim=self.claim)

    def _candidates(self, e: KNode, psig: KNode | None = None,
                    pcsig: KNode | None = None) -> list[Kline]:
        """The reference CANDIDATES minus loop control, in insertion order."""
        return [
            c
            for c in self.d
            if c.sig & e  # overlap — no-neighbours never pairs
            and c.adj  # a bare signature offers no construction
            and c.adj != frozenset([c.sig])  # an identity is not a partner
            and c.fingerprint() != e  # like for like
            and c.sig != psig  # the parent entry
            and c.sig != pcsig  # the parent candidate
            and not any(contained(c.sig, f) for f in self.frame)  # covered: nothing to teach
        ]


@dataclass
class _Entry:
    """One stack entry. parent — the seq of the pushing entry (0 for the
    opening); stalls match only across different parents."""

    sig: KNode
    cands: list
    depth: int
    memo: set = field(default_factory=set)
    seq: int = 0
    parent: int = 0


def rationalise(d: Domain, entry: Kline, goal: Kline | None = None) -> Rationalisation:
    """The one-shot runner — one Cogitator, one entry."""
    return Cogitator(d).rationalise(entry, goal=goal)
