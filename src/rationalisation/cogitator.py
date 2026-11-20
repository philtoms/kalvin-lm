"""The Cogitator — the model's runner

    Rationalise to significance:
        S1 - entry held exactly
        S2 - entry aligned with goal
        S3 - entry and goal aligned on common sub-goal
        S4 - entry not held

"""

from __future__ import annotations

from copy import deepcopy, copy
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from kalvin.kline import KLine, KNode, KNone, is_identity
from rationalisation.algebra import contained, fit, underfit, overfit, can
from rationalisation.domain import Domain
from rationalisation.report import Event, Rationalisation



@dataclass
class _Run:
    """One run's state — it rides on every entry of the run."""

    ask: KLine
    goal: KLine
    finish: Callable[["_Run"], None]
    entries: int = 0
    stalled: dict = field(default_factory=dict)  # signature -> the latest entry stalled on it
    events: list = field(default_factory=list)


@dataclass
class Entry:
    """One entry's exploration."""

    run: _Run  # the run the entry belongs to
    A: KNode = KNone # the ask
    C: KLine | None = None # the candidate goal
    parent: "Entry | None" = None  # the entry's provenance
    comp: str = ""  # "F" | "U" | "O" — the component of the parent's reentry
    chain: frozenset = frozenset()  # the ask signatures of the strict parent chain
    seen: dict = field(default_factory=dict)  # the ask's seen candidates — the empty probe (KNone) counts

    _frame: list[KNode] = field(default_factory=list)
    
    def frame(self, x: KNode) -> bool:
        if x and x not in self._frame:
            self._frame.append(x)
            return True
        return False

    def framed(self, x: KNode) -> bool:
        return not x or any(contained(x, f) for f in self._frame)

    def reframe(self, e: Entry):
        self._frame = e._frame.copy()

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
        on_stall: Callable[[KLine, Entry], bool] | None = None,
    ) -> None:
        self.d = d
        self.on_claim = on_claim
        self.on_stall = on_stall
        self.stack: list[Entry] = []
        self._cond = threading.Condition(threading.RLock())  # the server waits on it when the stack is empty
        self._stopped = False

    def open(self, ask: KLine, goal: KLine, finish: Callable[["_Run"], None]) -> Entry:
        """A run's root — a fresh state, the goal its one and only candidate."""
        return Entry(_Run(ask, goal, finish), ask.signature, goal, None, "")

    def rationalise(self, ask: KLine, goal: KLine, end_step: Callable[[Rationalisation]] | None = None, stop = True):
        def _finish(run: _Run) -> None:
            """The run's end — the report leaves; stalls left at S4."""
            if end_step is not None:
                end_step(Rationalisation(run.events))

        root = self.open(ask, goal, _finish)
        with self._cond:
            self.stack.append(root)
            self._cond.notify_all()
        if stop:
            self.stop()

    def cogitate(self):
        """Continue to cogitate until signalled to stop. Synchronous operation is achieved
        by presetting self._stopped (see rationalise)"""
        with self._cond:
            while True:
                entry = self.stack.pop()
                if entry:
                    self._turn(entry)
                while not self.stack:
                    if self._stopped: return
                    self._cond.wait()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    def _turn(self, entry: Entry) -> None:
        run = entry.run
        if run.entries > self.MAX_ENTRIES:
            run.finish(run)  # the entry cap — the run stops, the stack keeps walking
            return
        run.entries += 1

        A = entry.A
        C = entry.C
        if not C:  # S4 stalled
            if A in run.stalled and run.stalled[A].comp != entry.comp and common_ancestor(entry, run.stalled[A]):
                witness = run.stalled.pop(A)
                entry.frame(A) # S3 aligned
                run.events.append(Event("exchanged", A, frame=A))
                if witness.parent:
                    self._pop(witness.parent, entry) # the witness's side learns — its component framed
            elif self.on_stall is not None and self.on_stall(KLine(A, []), entry):
                if self._push(A, entry.parent, entry.comp): # served — the ask re-derives from the grown domain
                    return
            run.stalled[A] = entry
            run.events.append(Event("stalled", A))
            self._pop(entry.parent, entry) # the parent resumes — candidate kept, components re-read
            return

        cs = C.signature
        F, U, O = fit(A, cs), underfit(A, cs), overfit(C.nodes, A)
        if F == A:
            entry.frame(F)

        f_ok = entry.framed(F)
        u_ok = entry.framed(U)
        o_ok = entry.framed(O)
        if f_ok and u_ok and o_ok:  # S2 aligned
            entry.frame(A)
            claim = KLine(run.ask.signature, [run.goal.signature])
            if run.goal.signature == can(C.nodes):
                # the claim is made
                run.events.append(Event("claimed", A, C, F, U, O, frame=A))
                if self.on_claim is not None:
                    self.on_claim(claim)
                run.finish(run)
                return
            else:
                run.events.append(Event("framed", A, C, F, U, O, frame=A))
        else:
            run.events.append(Event("entry", A, C, F, U, O))

        if not u_ok:
            if self._push(U, entry, "U"): return

        if not o_ok:
            if self._push(O, entry, "O"): return

        if not f_ok:
            if self._push(F, entry, "F"): return

        self._pop(entry.parent, entry) # the parent resumes — its next turn rotates


    def _pop(self, parent: Entry, entry: Entry):
        parent = copy(parent)
        if parent:
            parent.reframe(entry)
            self.stack.append(parent)

    def _push(self, A: KNode, parent: Entry, comp: str) -> bool:
        """The component reenters — the child lands on top, the parent suspends
        beneath it, its candidate kept for the re-evaluation. A candidate seen
        for the ask does not return; the empty probe — the stall child —
        counts as seen and fires once."""
        if parent:
            if not A in parent.seen: parent.seen[A] = frozenset()
            chain = parent.chain | {parent.A}
            cands = candidates(A, self.d, parent.seen[A])
            if cands:
                C = cands.pop()
                parent.seen[A] |= {C.signature}
                entry = Entry(parent.run, A, C, parent, comp, chain)
                entry.reframe(parent)
                self.stack.append(entry)
                return True

            if not parent.seen[A]:
                parent.seen[A] = frozenset({KNone}) # the probe is the ask's last candidate
                entry = Entry(parent.run, A, None, parent, comp, chain)
                entry.reframe(parent)
                self.stack.append(entry)
                return True

        return False

def candidates(sig: KNode, d: Domain, chain: frozenset[KNode]) -> list[KLine]:
    """An ask never reenters its origin: its parent ask's and parent
    candidate's signatures are barred — the residual would re-derive
    the decomposition the parent entry already performed."""
    cands = [
        c
        for c in d
        if c.signature & sig  # overlap — no-neighbours never reenters
        and c.nodes  # a bare signature offers no construction
        and not is_identity(c)  # an identity is not a candidate
        and c.signature not in chain
    ]
    return cands
    # return list(reversed(cands))

def common_ancestor(l: Entry, r: Entry) -> bool:
    return bool(l.chain & r.parent.chain) if r.parent else False

def rationalise(d: Domain, ask: KLine, goal: KLine) -> Rationalisation | None:
    """The one-shot runner — one Cogitator, one entry. None: the ungrounded
    run reports nothing."""
    out: list[Rationalisation] = []
    cog = Cogitator(d)
    cog.rationalise(ask, goal, out.append)
    cog.cogitate()
    return out[0] if out else None
