"""The Cogitator — the reentry loop, the model's default runner.

    Rationalise to significance:
        S1 - entry held exactly
        S2 - entry aligned with goal
        S3 - entry and goal aligned on common sub-goal
        S4 - entry not held

    The current implementation rationalises to S2 only.

    rationalise(entry):                        # a call — one reentry
      cands = [goal] at depth 1, else candidates(E)   # its own, last first
      while cands:
        stack = base + [entry]                 # a fresh entry — a fresh memo
        while stack:                           # the reentry loop
          E, depth, memo = top
          C = cands[-1]
          pair E × C — close (frame E) or pair
          if a side unsettled, unmemoed, under the depth bound:
              call(side)                       # the descent — new state
          else: stack.pop()                    # the drain — the entry below
        cands.pop()                            # the next candidate
        entry = the last drained top           # the capture — the opening

    Three moves do the work. THE DESCENT: a pairing's unsettled side
    becomes a new call — its own candidates, its own stack copy; the
    caller resumes against its own candidate when the call returns.
    THE DRAIN: a call's reentry loop does not stop at its own entry —
    it pops through the whole stack, pairing every ancestor against
    the call's candidate, close cascades climbing as each mint frames
    the entry above. The drain bypasses the ancestors' candidate
    filters: a zero-fit pairing is legal here, and its underfit is the
    whole entry — the reentry meets itself one level down, and that
    reentry's stall is the cross-parent witness an exchange needs.
    THE CAPTURE: a call's entry is its loop variables, which the drain
    leaves at the stack's bottom — the opening, depth 1 — so every
    candidate after the first re-explores the opening entry, where the
    claim grounds and stalls re-file at parent 0.

    A stall waits for a witness, and the witness must align: two
    stalls on one signature exchange only when their reentry chains,
    walked upstream, part through the two sides of a shared pairing —
    the underfit and the offering of one decomposition meeting again
    on the same content. The S3 alignment; a second stall from the
    same side witnesses nothing.

    One loop over an explicit call stack — no recursion. The pairing
    cap alone bounds descent and drain alike; the interpreter's stack
    is never spent. Events are the grounding call's trace — the
    candidate path the claim came down; a run that grounds nothing
    reports none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kalvin.kline import KNode
from rationalisation.algebra import Kline, can, contained, meet, residue
from rationalisation.domain import Domain
from rationalisation.report import End, Event, Rationalisation, effort


@dataclass
class _Entry:
    """One reentry on a call's stack. parent — the stack position of the
    reentry whose pairing pushed this one (0 for the opening)."""

    sig: KNode
    depth: int
    parent: int
    memo: set = field(default_factory=set)  # sigs this row has forked — once per parent


@dataclass
class _Call:
    """One reentry's exploration. base_stack/base_events — what it was
    called with (events live: the caller's current list takes its
    stalls); cands — its own candidates, last first; stack/events — the
    current candidate's copies; E/depth/parent — the loop variables the
    drain reassigns (after the first candidate: the last drained top,
    not the call's own reentry)."""

    E: KNode
    depth: int
    parent: int
    psig: KNode | None
    pcsig: KNode | None
    base_stack: list
    base_events: list
    cands: list | None = None
    stack: list | None = None
    events: list | None = None
    caller: "_Call | None" = None  # the call whose loop pushed this one
    side: str = ""  # "u" | "o" — the side of the fork that pushed it
    pfp: int = 0  # the fork pairing's candidate fingerprint


class Cogitator:
    """The reentry runner: an entry's signature meets held claims, and each
    unsettled side descends into new state — a new call on the explicit
    stack, its own stack copy, its own frames. A pairing grounds when both
    sides settle; the fit needs no frame of its own — it is contained in
    the candidate's held claim."""

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
        self.stalled: dict[KNode, _Call] = {}
        self.frame: list[KNode] = []
        self.events: list[Event] = []

        if goal is None:
            if not entry.adj:
                self._halt("R4: entry kline with no adjacency", entry)
                return Rationalisation(self.stores, self.ends, [])
            if entry.sig == can(entry.adj):
                self._ground(entry)  # S1 grounds on receipt
                return Rationalisation(self.stores, self.ends, [])

        def mint(x: KNode) -> bool:
            if x and x not in self.frame:
                self.frame.append(x)
                return True
            return False

        def framed(x: KNode) -> bool:
            return not x or any(contained(x, f) for f in self.frame)

        def legit(a: _Call, b: _Call) -> bool:
            """The S3 alignment: the two stalled chains, walked upstream,
            part through the two sides of a shared pairing — the underfit
            and the offering of one decomposition meeting again. The chains
            are the calls (the capture never touches the forks)."""
            ca, cb = [], []
            x = a
            while x is not None:
                ca.append(x)
                x = x.caller
            x = b
            while x is not None:
                cb.append(x)
                x = x.caller
            i, j = len(ca) - 1, len(cb) - 1
            while i > 0 and j > 0 and ca[i - 1] is cb[j - 1]:
                i -= 1
                j -= 1
            return (i > 0 and j > 0
                    and ca[i - 1].pfp == cb[j - 1].pfp
                    and ca[i - 1].side != cb[j - 1].side)

        calls = [_Call(entry.sig, 1, 0, None, None, [], [])]
        while calls:
            if self.bound or self.claim is not None:
                break
            call = calls[-1]
            if call.cands is None:  # the reentry's candidates — or its stall
                call.cands = ([goal] if call.depth == 1 and goal is not None
                              else self._candidates(call.E, call.psig, call.pcsig))
                if not call.cands:  # S4 stalled
                    E = call.E
                    if E in self.stalled and legit(self.stalled[E], call):
                        mint(E)  # S3 align
                        call.base_events.append(Event("exchange", call.depth, E, frame=E))
                        self.stalled.pop(E)
                    else:
                        self.stalled[E] = call
                        call.base_events.append(Event("stall", call.depth, E))
                    calls.pop()
                    continue
            if call.stack is None:  # a fresh candidate — a fresh entry, a fresh trace
                call.stack = call.base_stack + [_Entry(call.E, call.depth, call.parent)]
                call.events = call.base_events.copy()
            if self.pairings > self.MAX_PAIRINGS:
                if not self.bound:
                    self._halt("bound: pairing cap reached", entry)
                    self.bound = True
                break
            if not call.stack:  # the candidate drained — the next, or done
                call.cands.pop()
                if not call.cands:
                    self._halt("exhausted: no candidate resolved both sides",
                               Kline(call.E, frozenset()))
                    calls.pop()
                else:
                    call.stack = None
                continue
            self.pairings += 1
            top = call.stack[-1]
            E, depth, memo = top.sig, top.depth, top.memo
            call.E, call.depth, call.parent = E, depth, top.parent  # the capture
            C = call.cands[-1]
            cs = C.sig
            fp = C.fingerprint()
            F, U, O = meet(E, cs), residue(E, cs), residue(can(C.adj), E)
            u_ok, o_ok = framed(U), framed(O)
            if u_ok and o_ok:  # S2 aligned
                mint(E)
                call.events.append(Event("close", depth, E, C, F, U, O, frame=E))
                if depth == 1 and goal is not None and goal.sig == C.sig:  # the claim grounds
                    self.claim = Kline(entry.sig, frozenset([goal.sig]))
                    self._ground(self.claim)
                    self.events = call.events
                    break  # the claim alone stops the run
                call.stack.pop()  # a close falls through to the pop
                continue
            call.events.append(Event("pair", depth, E, C, F, U, O))
            if not u_ok and U not in memo and depth < self.MAX_DEPTH:
                memo.add(U)  # once per parent — the row, not the fork
                calls.append(_Call(U, len(call.stack) + 1, len(call.stack), E, cs,
                                   call.stack.copy(), call.events,
                                   caller=call, side="u", pfp=fp))  # the descent
                continue
            if not o_ok and O not in memo and depth < self.MAX_DEPTH:
                memo.add(O)
                calls.append(_Call(O, len(call.stack) + 1, len(call.stack), E, cs,
                                   call.stack.copy(), call.events,
                                   caller=call, side="o", pfp=fp))
                continue
            call.stack.pop()  # the drain — the entry below meets this candidate

        for e in self.stalled:  # stalls that never found a witness
            self._halt("stalled at S4: no candidates", Kline(e, frozenset()))

        return Rationalisation(self.stores, self.ends, self.events, claim=self.claim)

    def _candidates(self, e: KNode, psig: KNode | None = None,
                    pcsig: KNode | None = None) -> list[Kline]:
        """The reference CANDIDATES minus loop control, in insertion order.
        A reentry never re-pairs its origin: its parent entry's and parent
        candidate's signatures are barred — the residual would re-derive
        the decomposition the parent pairing already performed. A signature
        covered on this path has nothing to teach. The drain pairs
        ancestors without this filter — their candidates are the call's."""
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


def rationalise(d: Domain, entry: Kline, goal: Kline | None = None) -> Rationalisation:
    """The one-shot runner — one Cogitator, one entry."""
    return Cogitator(d).rationalise(entry, goal=goal)
