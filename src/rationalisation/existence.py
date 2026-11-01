"""The certificate existence checker — T4's instrument.

Decides, without running the search, whether a covering certificate
*exists* for (d0, entry, goal): a fixpoint over abstract reentries.
Every pairing the rules allow is explored; every settle the rules permit
is taken; the certificate exists when the opening pairing's two sides
settle.

The checker over-approximates the run — each liberty makes "no
certificate" rarer, never wrong:

- frames are globally available (the run restricts SETTLE to
  chain-owned frames; only the proven path uses frames globally);
- an exchange needs two distinct abstract reentries on a value, not two
  stalls — the stall precondition reads schedule-shaped bars;
- the path, seen and covered-sig candidate bars are ignored;
- no depth or row bounds — certificates beyond the run's bounds count.

Refusal soundness (with T3): the run grounds only with a certificate,
and every certificate the run can produce is within this
over-approximation — so ``False`` means no schedule, no order, no luck
can ground the scenario.
"""

from __future__ import annotations

from rationalisation.model import Kline, can, side


def _canon_sigs(d0) -> frozenset[int]:
    return frozenset(int(k.sig) for k in d0 if int(k.sig) == int(can(k.adj)))


class _Checker:
    def __init__(self, d0, entry: Kline, goal: Kline, cap: int) -> None:
        self.cands = [
            c for c in d0
            if c.adj and c.adj != frozenset([c.sig])  # non-empty, non-identity
        ]
        self.goal = goal
        self.canons = _canon_sigs(d0)
        self.cap = cap
        self.work = 0
        self.resolved: set[tuple[int, frozenset]] = set()
        self.frames: set[int] = set()
        # region -> the reentry keys enqueued on it
        self.rekeys: dict[int, set[tuple[int, frozenset]]] = {}
        # region -> the pairing slots that reached it (provenance — the
        # run's exchange parties may be value-identical and differ only
        # in parentage)
        self.reach: dict[int, set] = {}
        # (reentry key, candidate index) -> (u child key, o child key)
        self.children: dict[tuple[tuple[int, frozenset], int], tuple] = {}
        self.entry_key = self._enqueue(int(entry.sig), frozenset(entry.adj), tag="open")
        self.goal_side = int(side(goal))
        self.entry_side = int(side(entry))
        # the opening pairing mints its frame when the goal covers the
        # entry's whole side (the goal-given evidence)
        if self.entry_side & self.goal_side == self.entry_side:
            self.frames.add(self.entry_side)

    def _enqueue(self, region: int, nodes: frozenset, tag=None) -> tuple:
        # the opening carries a tag: a later reentry equal to the entry
        # kline is NOT the opening and pairs with candidates, not the goal
        key = (tag, region, frozenset(int(n) for n in nodes))
        self.rekeys.setdefault(region, set()).add(key)
        return key

    def _exchange_ok(self, region: int) -> bool:
        return (
            len(self.rekeys.get(region, ())) >= 2
            or len(self.reach.get(region, ())) >= 2
        )

    def _re_side(self, re_key) -> int:
        _, region, nodes = re_key
        return region | int(can(frozenset(nodes)))

    def _children(self, re_key, cand_index: int) -> tuple:
        """The (u key, o key) of the pairing; created and enqueued on
        first use. ``cand_index`` -1 is the goal — the opening pairing,
        barred by nothing."""
        slot = (re_key, cand_index)
        if slot not in self.children:
            if cand_index == -1:
                c, c_adj = self.goal, self.goal.adj
            else:
                c = self.cands[cand_index]
                c_adj = c.adj
            se, sc = self._re_side(re_key), int(side(c))
            u, o = se & ~sc, sc & ~se
            re_nodes = re_key[2]
            u_key = (
                self._enqueue(u, frozenset(n for n in re_nodes if int(n) & u == int(n)))
                if u else None
            )
            o_key = (
                self._enqueue(o, frozenset(n for n in c_adj if int(n) & o == int(n)))
                if o else None
            )
            self.children[slot] = (u_key, o_key)
            for side_, key in (("u", u_key), ("o", o_key)):
                if key is not None:
                    self.reach.setdefault(key[1], set()).add((re_key, cand_index, side_))
        return self.children[slot]

    def _side(self, re_key, cand_index: int, side_: str) -> tuple[int, tuple | None]:
        u_key, o_key = self._children(re_key, cand_index)
        if side_ == "u":
            return (u_key[1] if u_key else 0), u_key
        return (o_key[1] if o_key else 0), o_key

    def _settled(self, region: int, side_: str, child_key) -> bool:
        if not region:
            return True
        if side_ == "o" and region in self.canons:
            return True  # the o-side witness settle
        if any(region & f == region for f in self.frames):
            return True
        return child_key is not None and child_key in self.resolved

    def _pairing_settles(self, re_key, cand_index: int) -> bool:
        for side_ in ("u", "o"):
            region, child = self._side(re_key, cand_index, side_)
            if not self._settled(region, side_, child):
                return False
        return True

    def _cand_pairable(self, re_key, cand_index: int) -> bool:
        c = self.cands[cand_index]
        region = re_key[1]
        if not int(c.sig) & region:
            return False  # overlap — no-neighbours never pairs
        if int(side(c)) == self._re_side(re_key):
            return False  # like for like
        return True

    def _spend(self, n: int = 1) -> bool:
        self.work += n
        return self.work <= self.cap

    def run(self) -> bool | None:
        changed = True
        while changed:
            changed = False
            before = sum(len(keys) for keys in self.rekeys.values())
            # exchanges: two distinct reentries on one value resolve each
            # other, and the value joins the frames
            for region, keys in list(self.rekeys.items()):
                if self._exchange_ok(region) and not keys <= self.resolved:
                    self.resolved |= keys
                    self.frames.add(region)
                    changed = True
                if not self._spend():
                    return None
            # pairings: the opening with the goal alone; every reentry
            # with every candidate that passes the schedule-free bars
            for _, keys in list(self.rekeys.items()):
                for re_key in list(keys):
                    s = self._re_side(re_key)
                    # a cover mints its frame at pairing time — the run
                    # mints whether or not the row ever resolves
                    if any(
                        self._cand_pairable(re_key, i)
                        and int(side(self.cands[i])) & s == s
                        for i in range(len(self.cands))
                    ):
                        if s not in self.frames:
                            self.frames.add(s)
                            changed = True
                    if re_key in self.resolved:
                        continue
                    if re_key == self.entry_key:
                        # the opening pairs with the goal alone — candidate
                        # pairings for it do not exist in the run
                        settled = self._pairing_settles(re_key, -1)
                    else:
                        # explore every pairing's children first — the run
                        # pushes them all; a short-circuited settle query
                        # would leave reentries unexplored
                        pairable = [
                            i for i in range(len(self.cands))
                            if self._cand_pairable(re_key, i)
                        ]
                        for i in pairable:
                            self._children(re_key, i)
                        settled = any(self._pairing_settles(re_key, i) for i in pairable)
                    if not self._spend(len(self.cands) + 1):
                        return None
                    if settled:
                        self.resolved.add(re_key)
                        changed = True
            # new reentries alone keep the fixpoint going — their
            # pairings have not been tried yet
            after = sum(len(keys) for keys in self.rekeys.values())
            changed = changed or after != before
        u_region, u_child = self._side(self.entry_key, -1, "u")
        o_region, o_child = self._side(self.entry_key, -1, "o")
        return self._settled(u_region, "u", u_child) and self._settled(
            o_region, "o", o_child
        )


def exists_certificate(d0, entry: Kline, goal: Kline, cap: int = 500_000) -> bool | None:
    """Whether a covering certificate exists for the ask==goal scenario.

    ``True`` — one exists (the run may still miss it); ``False`` — none
    exists, and by T3 the scenario cannot ground under any schedule;
    ``None`` — the fixpoint exceeded ``cap`` work, undecided.
    """
    return _Checker(d0, entry, goal, cap).run()
