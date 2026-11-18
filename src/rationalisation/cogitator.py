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
class _State:
    """One run's state — it rides on every entry of the run."""

    ask: KLine
    goal: KLine
    finish: Callable[["_State"], None]
    entries: int = 0
    frame: list[KNode] = field(default_factory=list)
    stalled: dict = field(default_factory=dict)  # signature -> the latest entry stalled on it
    events: list = field(default_factory=list)


@dataclass
class _Entry:
    """One entry's exploration.""" 

    st: _State  # the run the entry belongs to
    A: KNode = KNone # the ask
    cands: list | None = None # the candidate goals
    parent: "_Entry | None" = None  # the entry's provenance
    comp: str = ""  # "F" | "U" | "O" — the component of the parent's reentry
    chain: frozenset = frozenset()  # the ask signatures of the strict parent chain
    at: dict = field(default_factory=dict)  # component -> frame count at its last entry

    def frame(self, x: KNode) -> bool:
        if x and x not in self.st.frame:
            self.st.frame.append(x)
            return True
        return False

    def framed(self, x: KNode) -> bool:
        return not x or any(contained(x, f) for f in self.st.frame)
    
    def fresh(self, comp: str):
        return comp not in self.at or len(self.st.frame) > self.at[comp]

class Cogitator:
    """The reentry runner: an entry's signature meets held candidate goals, and each
    unframed component recurses as a new entry — its own candidates, its own chain.
    An entry is claimed when it is S2 aligned - all three components F(it), U(nderfit)
    and O(verfit) are framed.

    The stack is real: the top entry runs, a push lands the child on top and suspends
    the parent beneath it, a pop backtracks to whatever resumes on top. A turn reads
    only the entry and its run's state — never the entry's position — so independent
    runs may share one stack, interleaved.
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
        self.stack: list[_Entry] = []

    def open(self, ask: KLine, goal: KLine, finish: Callable[["_State"], None]) -> _Entry:
        """A run's root — a fresh state, the goal its one and only candidate."""
        return _Entry(_State(ask, goal, finish), ask.signature, [goal], None, "")

    def rationalise(self, ask: KLine, goal: KLine, end_step: Callable[[Rationalisation]] | None = None):
        def _finish(st: _State) -> None:
            """The run's end — stalls that never aligned (S4) broadcast bare."""
            for a in st.stalled:
                if self.on_stall is not None:
                    self.on_stall(KLine(a, []))
            if end_step is not None:
                end_step(Rationalisation(st.events))

        root = self.open(ask, goal, _finish)
        self.stack.append(root)

    def cogitate(self):
        """The walk — the stack's top entry turns until the stack is empty."""
        while self.stack:
            entry = self.stack[-1]
            st = entry.st
            if st.entries > self.MAX_ENTRIES:
                self._drop_run(st)  # the capped run's entries — dropped
                st.finish(st)  # the entry cap — the run stops, the stack keeps walking
                continue
            st.entries += 1

            A = entry.A
            if not entry.cands:  # S4 stalled
                if A in st.stalled and st.stalled[A].comp != entry.comp and self.common_ancestor(entry, st.stalled[A]):
                    entry.frame(A) # S3 aligned
                    st.events.append(Event("exchanged", A, frame=A))
                    st.stalled.pop(A)
                else:
                    st.stalled[A] = entry
                    st.events.append(Event("stalled", A))

                self._drop(entry)  # backtrack: the parent keeps its candidate
                continue

            C = entry.cands[-1]
            cs = C.signature
            F, U, O = fit(A, cs), underfit(A, cs), overfit(C.nodes, A)
            if F == A:
                entry.frame(F)

            f_ok = entry.framed(F)
            u_ok = entry.framed(U)
            o_ok = entry.framed(O)
            if f_ok and u_ok and o_ok:  # S2 aligned
                entry.frame(A)
                if st.goal.signature == can(C.nodes):  # the claim is made
                    st.events.append(Event("claimed", A, C, F, U, O, frame=A))
                    if self.on_claim is not None:
                        self.on_claim(KLine(st.ask.signature, [st.goal.signature]))  # leaves now, mid-run
                else:
                    st.events.append(Event("resolved", A, C, F, U, O, frame=A))
            elif not entry.at:  # the entry notes itself once — re-evaluation after
                st.events.append(Event("entry", A, C, F, U, O)) 

            if not u_ok and entry.fresh("U"):
                self._push(U, entry, "U")
                continue

            if not o_ok and entry.fresh("O"):
                self._push(O, entry, "O")
                continue

            if not f_ok and entry.fresh("F"):
                self._push(F, entry, "F")
                continue

            entry.cands.pop()
            entry.at.clear()
            if not entry.cands:
                self._drop(entry)
                if entry.parent is None:  # the root popped — the run is complete
                    st.finish(st)

    def _drop(self, entry: _Entry) -> None:
        """The entry leaves the stack wherever it sits — a sink may land
        another run's entries above it mid-iteration."""
        for i in reversed(range(len(self.stack))):
            if self.stack[i] is entry:
                del self.stack[i]
                return

    def _drop_run(self, st: _State) -> None:
        """The run's entries leave the stack wherever they sit."""
        for i in reversed(range(len(self.stack))):
            if self.stack[i].st is st:
                del self.stack[i]

    def _push(self, A: KNode, entry: _Entry, comp: str) -> None:
        """The component reenters — the child lands on top, the parent suspends
        beneath it, its candidate kept for the re-evaluation."""
        entry.at[comp] = len(entry.st.frame)
        cands = self._candidates(A, entry)
        chain = entry.chain | {entry.A}
        self.stack.append(_Entry(entry.st, A, cands, entry, comp, chain))

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
            and c.signature not in entry.chain
        ]
        return cands
        # return list(reversed(cands))

    def ancestor(self, sig: KNode, entry: _Entry) -> bool:
        return sig in entry.chain
    
    def common_ancestor(self, l: _Entry, r: _Entry) -> bool:
        return bool(l.chain & r.parent.chain) if r.parent else False
    
def rationalise(d: Domain, ask: KLine, goal: KLine) -> Rationalisation:
    """The one-shot runner — one Cogitator, one entry."""
    out: list[Rationalisation] = []
    cog = Cogitator(d)
    cog.rationalise(ask, goal, out.append)
    cog.cogitate()
    return out[0]
