"""The certificate checker — T3 made executable (docs/rationalisation.md §3).

Re-derives the justification of a grounded claim from the run's records,
without re-running the search. Every settled region of the opening
pairing must trace to one of the four leaves:

- empty   — the region is ⊥
- witness — the domain holds a canon with this signature
- frame   — an earned frame: a source row whose candidate (held, or the
            goal itself — the run's given) covers the frame value; or a
            joint frame: two stalls on the value with different parents
- exchange — the child settled through an exchange pair

Navigation reads the run's bookkeeping (which child resolved, which frame
covers) but the search's own conclusions are never evidence: nothing
here consults ``row.resolved`` as proof, only as routing to the record
that must then justify itself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from rationalisation.model import (
    Cogitator,
    Kline,
    Rationalisation,
    Row,
    can,
    side,
)


def _chain(row: Row | None) -> set[int]:
    """The proper-ancestor rows of a row's reentry — the frame owners whose
    frames may settle its regions (mirrors Cogitator._chain)."""
    out: set[int] = set()
    if row is None or row.re is None:
        return out
    cur = row.re.parent
    while cur is not None:
        out.add(id(cur))
        nxt = cur.re
        cur = nxt.parent if nxt is not None else None
    return out


@dataclass
class Certificate:
    """The justification tree of a grounded claim, as leaf counts."""

    claim: Kline
    leaves: Counter = field(default_factory=Counter)

    def leaf_kinds(self) -> list[str]:
        return sorted(self.leaves)


def verify(
    r: Rationalisation,
    cog: Cogitator,
    d0: Iterable[Kline],
    entry: Kline,
    goal: Kline,
) -> Certificate | None:
    """Check the covering certificate of a grounded claim.

    ``d0`` is the domain as the run found it (primed, plus any earlier
    step's stores). Returns the certificate when the records justify the
    claim; ``None`` when the claim did not ground or the records do not
    justify it.
    """
    if r.claim is None or r.goal_row is None:
        return None
    if r.claim != Kline(entry.sig, frozenset([goal.sig])):
        return None
    dom = set(d0) | set(r.stores)
    canons = {k.sig for k in dom if k.sig == can(k.adj)}

    cert = Certificate(r.claim)
    walked: set[tuple[int, str]] = set()

    def stall_pair(sig: int) -> bool:
        """Two stalled reentries on ``sig`` from different parent rows."""
        waiting = cog.stalls.get(sig, ())
        return any(
            a.parent is not b.parent and a.stalled is not None and b.stalled is not None
            for i, a in enumerate(waiting) for b in waiting[i + 1:]
        )

    def frame_evidence(f, owner) -> bool:
        """A frame value carried by evidence — an earned frame's candidate
        (held, or the goal itself — the run's given) covers it, or a joint
        frame's exchange pair stands."""
        if owner is None:
            if stall_pair(int(f)):
                cert.leaves["exchange"] += 1
                return True
            return False
        c = owner.candidate
        if c is None or side(c) & f != f:
            return False
        if c in dom:
            cert.leaves["witness-held"] += 1
            return True
        if c == goal:
            cert.leaves["goal-given"] += 1
            return True
        return False

    def framed(region: int, chain: set[int]) -> bool:
        """A frame covering ``region``, owned by a chain ancestor or joint,
        itself carried by evidence."""
        for f, owner in zip(cog.frames, cog.frame_re):
            if region & f != region:
                continue
            if owner is not None and id(owner) not in chain:
                continue  # a sibling's or own subtree's frame never settles
            if frame_evidence(f, owner):
                return True
        return False

    def proven(k, chain: set[int]) -> bool:
        """The exhausted-candidates known: a covering candidate exists whose
        fit value is framed (any owner) and whose overfit pre-settles —
        empty, canonically held, or framed by a chain ancestor."""
        sk = side(k)
        fit_framed = any(
            int(f) == int(sk) and frame_evidence(f, owner)
            for f, owner in zip(cog.frames, cog.frame_re)
        )
        if not fit_framed:
            return False
        for c in dom:
            sc = side(c)
            if sk & sc != sk:
                continue
            over = int(sc) & ~int(sk)
            if not over or over in canons or framed(over, chain):
                cert.leaves["proven"] += 1
                return True
        return False

    def justified(region: int, side_: str, row: Row) -> bool:
        if not region:
            cert.leaves["empty"] += 1
            return True
        key = (id(row), side_)
        if key in walked:
            return False  # cyclic justification — not a tree
        walked.add(key)
        chain = _chain(row)
        child = next((ch for ch in row.children if ch.side == side_), None)
        if child is not None and child.resolved:
            if child.resolved_by is not None:
                closer = child.resolved_by
                if closer.known:  # settled without exploring — framed or proven
                    chain = _chain_row(child)
                    if framed(int(child.kline.sig), chain):
                        return True
                    return proven(child.kline, chain)
                return (
                    justified(int(closer.U), "u", closer)
                    and justified(int(closer.O), "o", closer)
                )
            if stall_pair(int(child.kline.sig)):
                cert.leaves["exchange"] += 1
                return True
            return False
        if side_ == "o" and region in canons:
            cert.leaves["witness"] += 1
            return True
        return framed(region, chain)

    def _chain_row(re_) -> set[int]:
        out: set[int] = set()
        cur = re_.parent if re_ is not None else None
        while cur is not None:
            out.add(id(cur))
            nxt = cur.re
            cur = nxt.parent if nxt is not None else None
        return out

    root = r.goal_row
    if not (justified(int(root.U), "u", root) and justified(int(root.O), "o", root)):
        return None
    return cert
