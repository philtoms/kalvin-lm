"""The Cogitator — the simple reentry loop, the model's default runner.

    rationalise(entry):
      entries.push(entry.sig, candidates(entry))
      while entries:
        E, cands = entries.top()
        if not cands:
          frame(E) # stalled — a stall waits for a witness, never self-witnesses
          continue
        for C in cands:
          F, U, O = decompose(E, C)
          if F==E frame(F)
          if framed(U) or U.empty then U_resolved=True else entries.push(U, candidates(U))
          if framed(O) or O.empty then O_resolved=True else entries.push(O, candidates(O))
          if U_resolved and O_resolved:
            ground(E, C) # the fit is witnessed by the pairing itself
            entries.pop()
            break

Derived from dev/probes/probe_simple_reentry.py, measured against the
ReferenceCogitator over the canonical scripts (mhall, wdmh — every
scenario's claim agrees).  The loop's rules beyond the pseudocode:

  C1  A stall pops and waits; two stalls on one value from different
      parents mint one joint frame — nothing stores (the exchange).
  C2  A for-run that grounds nothing and pushes nothing new pops the
      entry — exhausted.  A run that pushed children leaves the entry
      to re-run its candidates when they drain; framed(U)/framed(O)
      then read the frames its children minted (the stack's re-check —
      stack-local: a frame minted below after the pop cannot save it).
  C3  Ground pops the grounding entry itself; pushed children stay.
      The for-run does not break on an inner ground — every candidate
      pairs once per run (the reference's independent batches).
  C4  Ground frames the grounded region — parents read resolutions as
      frames.
  C5  A frame pushes each child signature at most once across its
      for-runs; regions otherwise explore fresh per parent.
  C6  The opening entry pairs the goal alone; grounding it stores the
      claim Kline(ask, {goal}) and stops the run.
  C7  MAX_DEPTH — a push past the bound leaves the side unresolved.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from kalvin.kline import KNode, KNone
from rationalisation.model import (
    Domain,
    End,
    Kline,
    Rationalisation,
    Row,
    can,
    contained,
    effort,
    meet,
    residue,
)


class Cogitator:
    """The reentry runner: an entry's signature meets held claims, its
    underfit and offering residuals reenter, and pairings ground when both
    sides resolve. The fit needs no frame of its own — it is contained in
    the candidate's held claim, so the pairing itself is the witness."""

    MAX_DEPTH = 10
    MAX_PAIRINGS = 4096

    def __init__(self, d: Domain) -> None:
        self.d = d
        self.stores: list[Kline] = []

    def _halt(self, reason: str, k: Kline) -> None:
        self.ends.append(End(reason, k, effort(k)))

    def _admit(self, k: Kline) -> None:
        self.d.add(k)
        if k not in self.stores:
            self.stores.append(k)

    def rationalise(self, entry: Kline, goal: Kline | None = None) -> Rationalisation:
        self.stores, self.ends = [], []
        self.rows: list[Row] = []
        self.events: list[tuple[Row, str]] = []
        self.claim: Kline | None = None
        self.goal_row: Row | None = None
        self.frames: list[KNode] = []
        self.waiting: dict[KNode, list[int]] = {}
        stack: list[_Frame] = []
        seq = itertools.count(1)

        if goal is None:  # the boundary rules, as the reference preamble
            if not entry.adj:
                self._halt("R4: entry kline with no adjacency", entry)
                return Rationalisation(self.stores, self.ends, self.rows)
            if entry.sig == can(entry.adj):
                self._admit(entry)  # S1 grounds on receipt
                return Rationalisation(self.stores, self.ends, self.rows)

        def framed(x: KNode) -> bool:
            return not x or any(contained(x, f) for f in self.frames)

        def mint(x: KNode) -> None:
            if x and x not in self.frames:
                self.frames.append(x)

        def push(sig: KNode, depth: int, memo: set, parent: int) -> None:
            memo.add(sig)  # C5 — once per parent, across its for-runs
            cands = [goal] if depth == 1 and goal is not None else self._candidates(sig)
            stack.append(_Frame(sig, cands, depth, set(), next(seq), parent))

        push(entry.sig, 1, set(), 0)

        while stack:
            if len(self.rows) > self.MAX_PAIRINGS:
                self._halt("bound: pairing cap reached", entry)
                break
            fr = stack[-1]
            e, cands, depth, memo = fr.sig, fr.cands, fr.depth, fr.memo

            if not cands:  # a stall waits — never self-witnesses (C1)
                peers = self.waiting.setdefault(e, [])
                if any(q != fr.parent for q in peers):
                    mint(e)  # the exchange's joint frame — nothing stores
                    self._stall_row(depth, e, framed_now=True)
                else:
                    peers.append(fr.parent)
                    self._stall_row(depth, e)
                stack.pop()
                continue

            new_push = False
            grounded = False
            for C in cands:
                cs = C.sig
                F, U, O = meet(e, cs), residue(e, cs), residue(can(C.adj), e)
                row = Row(depth, Kline(e, frozenset()), C, F=F, U=U, O=O)
                self.events.append((row, "pair"))
                self.rows.append(row)
                if F == e:
                    mint(F)
                    row.frame = F
                U_res = framed(U)
                O_res = framed(O)
                if not U_res and U not in memo and depth < self.MAX_DEPTH:  # C5, C7
                    push(U, depth + 1, memo, fr.seq)
                    new_push = True
                if not O_res and O not in memo and depth < self.MAX_DEPTH:
                    push(O, depth + 1, memo, fr.seq)
                    new_push = True
                if U_res and O_res:  # the fit is witnessed by the pairing itself
                    mint(e)  # C4 — the grounding frames its region
                    row.frame = e if not row.frame else row.frame.merge(e)
                    self.events.append((row, "close"))
                    stack[:] = [x for x in stack if x is not fr]  # C3 — children stay
                    grounded = True
                    if depth == 1 and goal is not None:  # C6 — the opening pairing resolved: the claim grounds
                        self.goal_row = row
                        self.claim = Kline(entry.sig, frozenset([goal.sig]))
                        self._admit(self.claim)
                        break  # the claim alone stops the run — an inner ground never does (C3)

            if self.claim is not None:
                break  # C6 — the claim stops the run
            if grounded:
                continue
            if not new_push:  # C2 — nothing left to spawn, nothing settled it
                if not framed(e):
                    self._halt("exhausted: no candidate resolved both sides",
                               Kline(e, frozenset()))
                stack.pop()

        for v in self.waiting:  # stalls that never found a witness
            if not framed(v):
                self._halt("stalled at S4: no candidates", Kline(v, frozenset()))

        path = [r for r in self.rows if r.frame] or self.rows  # the witnesses
        return Rationalisation(
            self.stores, self.ends, self.rows, path=path, claim=self.claim,
            goal_row=self.goal_row, events=self.events,
        )

    def _stall_row(self, depth: int, e: KNode, framed_now: bool = False) -> None:
        row = Row(depth, Kline(e, frozenset()), None, U=e)
        if framed_now:
            row.frame = e
        self.events.append((row, "pair"))
        self.rows.append(row)

    def _candidates(self, e: KNode) -> list[Kline]:
        """The reference CANDIDATES minus loop control, in insertion order."""
        return [
            c
            for c in self.d
            if c.sig & e  # overlap — no-neighbours never pairs
            and c.adj  # a bare signature offers no construction
            and c.adj != frozenset([c.sig])  # an identity is not a partner
            and c.fingerprint() != e  # like for like
            and not any(contained(c.sig, f) for f in self.frames)  # covered: nothing to teach
        ]


@dataclass
class _Frame:
    """One stack entry. parent — the seq of the pushing frame (0 for the
    opening entry); stalls match only across different parents."""

    sig: KNode
    cands: list
    depth: int
    memo: set = field(default_factory=set)
    seq: int = 0
    parent: int = 0


def rationalise(d: Domain, entry: Kline, goal: Kline | None = None) -> Rationalisation:
    """The one-shot runner — one Cogitator, one entry."""
    return Cogitator(d).rationalise(entry, goal=goal)
