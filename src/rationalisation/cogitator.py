"""The Cogitator — the model's runner.

    Rationalise to significance:
        S1 - entry held exactly
        S2 - entry aligned with goal
        S3 - entry and goal aligned on common sub-goal
        S4 - entry not held

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from kalvin.kline import KLine, KNode, KNone, is_identity
from rationalisation.algebra import contained, fit, underfit, overfit, can
from rationalisation.domain import Domain
from rationalisation.report import Event, Rationalisation


@dataclass
class _Entry:
    """One entry's exploration.""" 

    A: KNode = KNone # the ask
    cands: list | None = None # the candidate goals
    parent: "_Entry | None" = None  # the entry's provenance
    comp: str = ""  # "F" | "U" | "O" — the component of the parent's reentry
    at: dict = field(default_factory=dict)  # component -> frame count at its last descent

ENone = _Entry()

class Cogitator:
    """The reentry runner: an entry's signature meets held candidate goals, and each
    unframed component recurses into new state — its own stack copy, its own
    frames. An entry is claimed when it is S2 aligned - all three components 
    F(it), U(nderfit) and O(verfit) are framed.
    """

    MAX_ENTRIES = 4096

    def __init__(
        self,
        d: Domain,
        on_claim: Callable[[KLine], None] | None = None,
        on_stall: Callable[[KLine], None] | None = None,
    ) -> None:
        self.d = d
        self.on_claim = on_claim
        self.on_stall = on_stall

    def rationalise(self, ask: KLine, goal: KLine) -> Rationalisation:
        self.entries = 0
        self.stalled: dict[KNode, _Entry] = {}
        self.frame: list[KNode] = []
        self.events: list[Event] = []

        def frame(x: KNode) -> bool:
            if x and x not in self.frame:
                self.frame.append(x)
                return True
            return False

        def framed(x: KNode) -> bool:
            return not x or any(contained(x, f) for f in self.frame)

        def pop(entry: _Entry) -> _Entry:
            if not entry.parent: return ENone
            return entry.parent

        def push(A: KNode, parent: _Entry, comp: str) -> _Entry:
            cands = self._candidates(A, parent)
            return  _Entry(A, cands, parent, comp)

        def fresh(comp: str) -> bool:
            # a component reenters once per candidate — and again only
            # when new frames arrived since its last entry
            return comp not in entry.at or len(self.frame) > entry.at[comp]

        entry = push(ask.signature, ENone, "")
        entry.cands = [goal]
        while entry.A != KNone:
            if self.entries > self.MAX_ENTRIES:
                break  # the entry cap — the run stops
            self.entries += 1

            A = entry.A
            if not entry.cands:  # S4 stalled
                if A in self.stalled and self.stalled[A].comp != entry.comp and self.common_ancestor(entry, self.stalled[A]):
                        frame(A) # S3 aligned
                        self.events.append(Event("exchanged", A, frame=A))
                        self.stalled.pop(A)
                else:
                    self.stalled[A] = entry
                    self.events.append(Event("stalled", A))

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
                    if goal.signature == can(C.nodes):  # the claim is made
                        self.events.append(Event("claimed", A, C, F, U, O, frame=A))
                        if self.on_claim is not None:
                            self.on_claim(KLine(ask.signature, [goal.signature]))  # leaves now, mid-run
                    else:
                        self.events.append(Event("resolved", A, C, F, U, O, frame=A))
                elif not entry.at:  # the entry notes itself once — re-evaluation after
                    self.events.append(Event("entry", A, C, F, U, O)) 

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

        for a in self.stalled:  # stalls that never aligned (S4)
            if self.on_stall is not None:
                self.on_stall(KLine(a, []))

        return Rationalisation(self.events)

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
    
def rationalise(d: Domain, ask: KLine, goal: KLine) -> Rationalisation:
    """The one-shot runner — one Cogitator, one entry."""
    return Cogitator(d).rationalise(ask, goal)
