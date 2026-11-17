"""The Cogitator — the model's default runner.

    Rationalise to significance:
        S1 - entry held exactly
        S2 - entry aligned with goal
        S3 - entry and goal aligned on common sub-goal
        S4 - entry not held

"""

from __future__ import annotations

from dataclasses import dataclass, field

from kalvin.kline import KLine, KNode, KNone, is_identity
from rationalisation.algebra import can, contained, fit, underfit, overfit
from rationalisation.domain import Domain
from rationalisation.report import End, Event, Rationalisation, effort


@dataclass
class _Entry:
    """One entry's exploration.""" 

    A: KNode = KNone # the ask
    cands: list | None = None # the candidate goals
    parent: "_Entry | None" = None  # the entry's provenance
    comp: str = ""  # "F" | "U" | "O" — the component of the parent's reentry
    at: dict = field(default_factory=dict)  # component -> frame count at its last descent

CNone = _Entry()

class Cogitator:
    """The reentry runner: an entry's signature meets held claims, and each
    unframed component recurses into new state — its own stack copy, its own
    frames. An entry grounds when it is S2 aligned - all three components 
    F(it), U(nderfit) and O(verfit) are framed.
    """

    MAX_ENTRIES = 4096

    def __init__(self, d: Domain) -> None:
        self.d = d
        self.stores: list[KLine] = []

    def rationalise(self, ask: KLine, goal: KLine | None = None) -> Rationalisation:
        self.stores, self.ends = [], []
        self.claim: KLine | None = None
        self.entries = 0
        self.bound = False
        self.stalled: dict[KNode, _Entry] = {}
        self.frame: list[KNode] = []
        self.events: list[Event] = []

        if goal is None:
            if not ask.nodes:
                self._halt("R4: entry kline with no adjacency", ask)
                return Rationalisation(self.stores, self.ends, [])
            if ask.signature == can(ask.nodes):
                self._ground(ask)  # S1 grounds on receipt
                return Rationalisation(self.stores, self.ends, [])

        def frame(x: KNode) -> bool:
            if x and x not in self.frame:
                self.frame.append(x)
                return True
            return False

        def framed(x: KNode) -> bool:
            return not x or any(contained(x, f) for f in self.frame)

        def pop(entry: _Entry) -> _Entry:
            if not entry.parent: return CNone
            return entry.parent

        def push(A: KNode, parent: _Entry, comp: str) -> _Entry:
            cands = self._candidates(A, parent)
            return  _Entry(A, cands, parent, comp)

        entry = push(ask.signature, CNone, "")
        entry.cands = [goal]
        while entry.A != KNone:
            if self.entries > self.MAX_ENTRIES:
                self._halt("bound: entry cap reached", ask)
                break
            self.entries += 1

            A = entry.A
            if not entry.cands:  # S4 stalled
                if A in self.stalled and self.stalled[A].comp != entry.comp and self.common_ancestor(entry, self.stalled[A]):
                        frame(A) # S3 align
                        self.events.append(Event("exchange", A, frame=A))
                        self.stalled.pop(A)
                else:
                    self.stalled[A] = entry
                    self.events.append(Event("stall", A))

                entry = pop(entry)  # backtrack: the parent keeps its candidate
                continue
                
            else:
                C = entry.cands[-1]
                cs = C.signature
                F, U, O = fit(A, cs), underfit(A, cs), overfit(C.nodes, A)
                if F == A:
                    frame(F)

                f_ok = framed(F)
                u_ok = framed(U)
                o_ok = framed(O)
                if f_ok and u_ok and o_ok:  # S2 aligned
                    frame(A)
                    if goal.signature == C.signature:  # the claim grounds
                        self.events.append(Event("grounded", A, C, F, U, O, frame=A))
                        self.claim = KLine(ask.signature, [goal.signature])
                        self._ground(self.claim)
                    else:
                        self.events.append(Event("close", A, C, F, U, O, frame=A))
                elif not entry.at:  # the entry notes itself once — re-evaluation after
                    self.events.append(Event("entry", A, C, F, U, O)) 

                def fresh(comp: str) -> bool:
                    # a component reenters once per candidate — and again only
                    # when new frames arrived since its last entry
                    return comp not in entry.at or len(self.frame) > entry.at[comp]

                if not u_ok and fresh("U"):
                    entry.at["U"] = len(self.frame)
                    entry = push(U, entry, "U")
                    continue

                if not o_ok and fresh("O"):
                    entry.at["O"] = len(self.frame)
                    entry = push(O, entry, "O")
                    continue

                if not f_ok and fresh("F"):
                    entry.at["F"] = len(self.frame)
                    entry = push(F, entry, "F")
                    continue

                entry.cands.pop()
                entry.at.clear()

            if not entry.cands:
                entry = pop(entry)

        for a in self.stalled:  # stalls that never aligned
            self._halt("stalled at S4: no candidates", KLine(a, []))

        return Rationalisation(self.stores, self.ends, self.events, claim=self.claim)

    def _candidates(self, sig: KNode, entry: _Entry) -> list[KLine]:
        """An ask never reenters its origin: its parent ask's and parent
        candidate's signatures are barred — the residual would re-derive
        the decomposition the parent entry already performed."""
        cands = [
            c
            for c in self.d
            if c.signature & sig  # overlap — no-neighbours never reenters
            and c.nodes  # a bare signature offers no construction
            and not is_identity(c)  # an identity is not a candidate
            and not self.ancestor(c.signature, entry)
        ]
        return cands
        # return list(reversed(cands))

    def _halt(self, reason: str, k: KLine) -> None:
        self.ends.append(End(reason, k, effort(k)))

    def _ground(self, k: KLine) -> None:
        self.d.add(k)
        if k not in self.stores:
            self.stores.append(k)

    def ancestor(self, sig: KNode, entry: _Entry) -> bool:
        while entry.parent:
            entry = entry.parent
            if entry.A == sig: return True
        return False
    
    def common_ancestor(self, l: _Entry, r: _Entry) -> bool:
        while l.parent:
            l = l.parent
            if self.ancestor(l.A, r.parent):
                return True
        return False
    
def rationalise(d: Domain, ask: KLine, goal: KLine | None = None) -> Rationalisation:
    """The one-shot runner — one Cogitator, one entry."""
    return Cogitator(d).rationalise(ask, goal=goal)
