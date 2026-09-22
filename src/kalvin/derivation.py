"""The derivation model — kalvin-algebra §6–10 as an engine core.

A Derivation rewrites the node sequence of a queued kline under held
correspondences, relative to a goal kline (Def 12). The queued head rides
inert; only the nodes change. Memory grows as meeting walks write bridge
correspondences (Def 15, progressive path).

The run loop implements policy A — the §9 documented order
(canonicalisation → targeting → the meeting walk). The enumerators
are the mechanism/policy boundary: they yield licensed
options in deterministic order; the loop chooses.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations, product

from kalvin.abstract import KSignifier
from kalvin.kline import KLine, is_canon, is_terminal, sig_level
from kalvin.significance import (
    DEFAULT_DELTA,
    misfit_mass,
)

#: T2-class strategy bound: total rewrites per run.
MAX_STEPS = 32
#: T2-class strategy bound: descent edges per party in a meeting walk.
MAX_WALK_EDGES = 8


@dataclass
class DerivationResult:
    """A run's outcome: ending, trace, and the §11 measurement."""

    ending: str  # "done" | "stuck" | "abandoned"
    trace: list[list[int]] = field(default_factory=list)
    composed: list[KLine] = field(default_factory=list)
    j0: float = 0.0
    j1: float = 0.0  # significance: J(final content, goal)
    dbar: float = 0.0
    hbar: float = 0.0
    gamma: float = 0.0  # γ: J·δ^(D̄+Ĥ) — significance net of complexity; never the band


class Derivation:
    """A ⊢_{M,B} … — Defs 12–17 over the memory supplied. The memory is the
    derivation's scope (Def 23): read as given, never extended mid-run;
    writes leave via result.composed for STM (Def 12)."""

    def __init__(
        self,
        memory: Sequence[KLine],
        queued: KLine,
        goal: KLine,
        signifier: KSignifier,
        *,
        max_steps: int = MAX_STEPS,
        max_walk_edges: int = MAX_WALK_EDGES,
        delta: float = DEFAULT_DELTA,
    ) -> None:
        self.memory = list(memory)
        self.queued = queued
        self.goal = goal
        self.signifier = signifier
        self.max_steps = max_steps
        self.max_walk_edges = max_walk_edges
        self.delta = delta
        self._composed_keys: set[tuple[int, tuple[int, ...]]] = set()
        self.nodes: list[int] = list(queued.nodes)
        self.composed: list[KLine] = []
        self.acq: dict[int, int] = {}  # unit value -> acquisition depth (§11)

    # ── state reads ────────────────────────────────────────────────────────

    def content(self) -> int:
        """σ(ν_A)."""
        return int(self.signifier.signature_of(self.nodes))

    def goal_content(self) -> int:
        return int(self.signifier.signature_of(self.goal.nodes))

    def gap(self) -> int:
        """A's atoms beyond the goal — what must be shed (Def 9)."""
        return int(self.signifier.residual(self.content(), self.goal_content()))

    def excess(self) -> int:
        """The goal's atoms beyond A — what must be adopted (Def 9)."""
        return int(self.signifier.residual(self.goal_content(), self.content()))

    def mismatch(self) -> int:
        """|σ(ν_A) Δ σ(ν_B)| (Def 14)."""
        return misfit_mass(self.content(), self.goal_content())

    def relationship(self) -> KLine:
        """C(A,B) — head defined, not claimed (Def 11)."""
        return KLine(self.content(), self.goal.nodes)

    def relationship_band(self) -> str:
        return sig_level(self.relationship(), self.signifier)

    def done(self) -> bool:
        return self.mismatch() == 0

    # ── evidence predicates (Def 13) ───────────────────────────────────────

    def usable(self, k: KLine) -> bool:
        """Unknown has no witness; Identity is inert; evidence carrying
        the queued head s never licenses writing s into its own witness
        (Def 13)."""
        s = int(self.queued.signature)
        if self.signifier.same_content(k.signature, s) or any(
            self.signifier.same_content(n, s) for n in k.nodes
        ):
            return False
        return not is_terminal(k)

    def wellfounded(self, k: KLine) -> bool:
        """n ∉ ν_K — Canon expansion terminates."""
        return k.signature not in k.nodes

    @staticmethod
    def occurs_fwd(k: KLine, nodes: list[int]) -> bool:
        return k.signature in nodes

    @staticmethod
    def occurs_rev(k: KLine, nodes: list[int]) -> bool:
        want = Counter(int(n) for n in k.nodes)
        have = Counter(int(n) for n in nodes)
        return all(have.get(n, 0) >= c for n, c in want.items())

    @staticmethod
    def replace_fwd(k: KLine, nodes: list[int]) -> list[int]:
        i = nodes.index(k.signature)
        return nodes[:i] + list(k.nodes) + nodes[i + 1 :]

    @staticmethod
    def replace_rev(k: KLine, nodes: list[int]) -> list[int]:
        want = Counter(int(n) for n in k.nodes)
        out: list[int] = []
        placed = False
        for n in nodes:
            key = int(n)
            if want.get(key, 0) > 0:
                want[key] -= 1
                if not placed:
                    out.append(k.signature)
                    placed = True
            else:
                out.append(n)
        return out

    # ── licensed-option enumeration ────────────────────────────────────────

    def canonicalisations(self) -> Iterator[tuple[KLine, tuple[int, ...], list[int]]]:
        """Exactly-witnessed proper groups contractable under a held canon,
        smallest group first (Def 13, canonicalisation).

        The survey runs over held witnesses, not node subsets: a group is
        contractable iff some held canon's node multiset matches it. The
        subset enumeration the letter of Def 13 suggests is exponential in
        a grown node list; witness-driven matching is the same relation.
        """
        canons = [
            (i, k) for i, k in enumerate(self.memory)
            if is_canon(k, self.signifier) and self.usable(k)
            and 2 <= len(k.nodes) < len(self.nodes)
        ]
        canons.sort(key=lambda t: (len(t[1].nodes), t[0]))
        for _, k in canons:
            want = Counter(int(x) for x in k.nodes)
            # every placement of the canon's multiset in the current nodes
            idxs_by_val: dict[int, list[int]] = {}
            for i, x in enumerate(self.nodes):
                idxs_by_val.setdefault(int(x), []).append(i)
            if any(len(idxs_by_val.get(v, ())) < c for v, c in want.items()):
                continue
            per_val = [
                list(combinations(sorted(idxs_by_val[v]), want[v])) for v in want
            ]
            for picks in product(*per_val):
                combo = sorted(i for pick in picks for i in pick)
                new = [x for i, x in enumerate(self.nodes) if i not in combo]
                new.insert(combo[0], k.signature)
                yield k, tuple(int(self.nodes[i]) for i in combo), new

    def targetings(self) -> Iterator[tuple[KLine, str, list[int], int, int]]:
        """Licensed targeting replaces with strictly falling misfit mass
        (Def 14). In an S2 region the restriction reads on both ends of
        the move: forward departs the gap or adopts the excess; reverse
        consumes the gap or lands in the excess. Order: memory order,
        forward before reverse."""
        cur = self.content()
        d0 = self.mismatch()
        gap = self.gap()
        excess = self.excess()
        s2 = self.relationship_band() == "S2"
        for k in self.memory:
            if not self.usable(k):
                continue
            if k.signature in self.nodes:
                if s2 and not (
                    self.signifier.signifies(k.signature, gap)
                    or self.signifier.signifies(
                        int(self.signifier.signature_of(k.nodes)), excess
                    )
                ):
                    continue
                new = self.replace_fwd(k, self.nodes)
                d1 = misfit_mass(
                    int(self.signifier.signature_of(new)), self.goal_content()
                )
                if d1 < d0:
                    yield k, "forward", new, d0, d1
            if self.occurs_rev(k, self.nodes):
                if s2 and not (
                    all(self.signifier.signifies(n, gap) for n in k.nodes)
                    or self.signifier.signifies(k.signature, excess)
                ):
                    continue
                new = self.replace_rev(k, self.nodes)
                d1 = misfit_mass(
                    int(self.signifier.signature_of(new)), self.goal_content()
                )
                if d1 < d0:
                    yield k, "reverse", new, d0, d1

    def _walk_neighbours(
        self, v: int, delivered: tuple[int, tuple[int, ...]] | None,
        *, a_side: bool = False,
    ) -> list[tuple[int, tuple[int, tuple[int, ...]]]]:
        """Def 15 — the two step forms from value ``v``: a descent crosses
        a held kline ``v`` heads into its witness; an ascent steps into
        the head of a held kline covering ``v``. An ascent into the head
        of the kline that delivered ``v`` is inert. A-side walks cross
        single-node witnesses only — a step licenses the kline's whole
        witness, and a multi-node witness cannot be partially consumed
        (its siblings would dangle); B reads ν_B's klines as walk
        material, the goal's own canon enumerating its slots."""
        out: list[tuple[int, tuple[int, tuple[int, ...]]]] = []
        for k in self.memory:
            if not self.usable(k):
                continue
            key = (int(k.signature), tuple(int(n) for n in k.nodes))
            if int(k.signature) == v:
                if v in [int(n) for n in k.nodes]:
                    continue  # self-containing — an inert witness
                if a_side and len(k.nodes) != 1:
                    continue  # the whole witness or none — no dangling siblings
                out.extend((int(n), key) for n in k.nodes)
            elif (
                key != delivered
                and int(self.signifier.residual(v, int(k.signature))) == 0
            ):
                out.append((int(k.signature), key))
        return out

    def descend(
        self, starts: Sequence[int]
    ) -> dict[int, tuple[int, tuple[int, tuple[int, ...]], int]]:
        """Def 15 — the walk from ``starts``, descents and ascents, as
        ``{value: (depth, delivering kline key, start)}``; bounded by the
        walk-edge bound (T2)."""
        reached: dict[int, tuple[int, tuple[int, tuple[int, ...]], int]] = {}
        frontier = [(int(v), 0, int(v), None) for v in starts]
        while frontier:
            nxt: list[tuple[int, int, int, tuple[int, tuple[int, ...]] | None]] = []
            for v, depth, start, delivered in frontier:
                if depth >= self.max_walk_edges:
                    continue
                for value, key in self._walk_neighbours(v, delivered, a_side=True):
                    if value not in reached:
                        reached[value] = (depth + 1, key, start)
                        nxt.append((value, depth + 1, start, key))
            frontier = nxt
        return reached

    # ── the run (policy A) ─────────────────────────────────────────────────

    def run(self) -> DerivationResult:
        """Policy A (§9): canonicalise, then target, then walk slots.
        Done, stuck, or abandoned at the step bound (Def 15, T2)."""
        result = DerivationResult(ending="abandoned", trace=[list(self.nodes)])
        result.j0 = self._jaccard()
        for _ in range(self.max_steps):
            if self.done():
                return self._finish(result, "done")
            if self._canonicalise():
                result.trace.append(list(self.nodes))
                continue
            if self._target():
                result.trace.append(list(self.nodes))
                continue
            if self._walk():
                continue
            return self._finish(result, "stuck")
        return self._finish(result, "abandoned")

    def _canonicalise(self) -> bool:
        """Take the first exactly-witnessed contraction (§9: the survey
        — held witnesses propose the configurations, no exposure
        requirement)."""
        for _, _, new in self.canonicalisations():
            self.nodes = new
            return True
        return False

    def _target(self) -> bool:
        t = next(self.targetings(), None)
        if t is None:
            return False
        k, _, new, _, _ = t
        self._record_arrival(new, k)
        self.nodes = new
        return True

    def _walk(self) -> bool:
        """Def 15 — the slot walk is a meeting of two walks: A's from
        its underfit slots, B's from the held value containing the
        overfit. The meeting — a value delivered by distinct klines on
        the two sides — writes the bridge slot_a:[slot_b]."""
        if not self.excess():
            return False  # no overfit to bridge to
        a_starts = [
            n for n in self.nodes if self.signifier.signifies(n, self.gap())
        ]
        if not a_starts:
            return False
        path_a = self.descend(a_starts)
        if not path_a:
            return False
        excess = self.excess()
        b_values: list[int] = []
        for k in self.memory:
            if int(self.signifier.residual(excess, int(k.signature))) == 0:
                b_values.append(int(k.signature))
            for n in k.nodes:
                if int(self.signifier.residual(excess, int(n))) == 0:
                    b_values.append(int(n))
        b_starts = [
            v for v in dict.fromkeys(b_values)  # first-occurrence order
            if not self.signifier.same_content(v, self.queued.signature)  # never bridge to s
        ]
        if not b_starts:
            return False
        return self._meet(path_a, b_starts)

    def _meet(
        self,
        path_a: dict[int, tuple[int, tuple[int, tuple[int, ...]], int]],
        b_starts: Sequence[int],
    ) -> bool:
        """Walk from B's side until a value of ``path_a`` is reached; the
        first meeting delivered by a distinct kline writes the bridge
        slot_a:[slot_b] — the A-side underfit slot to the B-side overfit
        slot: the meeting value when it is a node of ν_B, else B's
        departure (§9: the compound is itself the slot)."""
        frontier = [(int(v), 0, int(v), None) for v in b_starts]
        seen = {int(v) for v in b_starts}
        while frontier:
            nxt: list[tuple[int, int, int, tuple[int, tuple[int, ...]] | None]] = []
            for v, depth, start, delivered in frontier:
                if depth >= self.max_walk_edges:
                    continue
                for value, key in self._walk_neighbours(v, delivered):
                    if value in path_a:
                        a_depth, a_key, a_start = path_a[value]
                        if a_key != key and a_start != start:
                            b_slot = (
                                value
                                if any(int(n) == value for n in self.goal.nodes)
                                else start
                            )
                            bridge = KLine(
                                a_start, [b_slot], acq_depth=a_depth + depth + 1
                            )
                            if self._ground_composed(bridge):
                                self.composed.append(bridge)
                                return True
                        continue
                    if value not in seen:
                        seen.add(value)
                        nxt.append((value, depth + 1, start, key))
            frontier = nxt
        return False

    def _ground_composed(self, composed: KLine) -> bool:
        """Write a composed correspondence once per run: a repeated
        bridge is not progress, and re-deriving it wedges the loop. The
        write leaves with the result — the scope never sees it."""
        key = (int(composed.signature), tuple(int(n) for n in composed.nodes))
        if key in self._composed_keys:
            return False
        self._composed_keys.add(key)
        return True

    def _exposes(self, new_nodes: list[int]) -> bool:
        return any(
            self.usable(k)
            and not is_canon(k, self.signifier)
            and (k.signature in new_nodes or self.occurs_rev(k, new_nodes))
            for k in self.memory
        )

    def _record_arrival(self, new_nodes: list[int], k: KLine) -> None:
        """§11: arriving atoms record their cost — the composed depth, 0 for
        S1 evidence, 1 for unratified evidence."""
        arriving = int(
            self.signifier.residual(
                int(self.signifier.signature_of(new_nodes)), self.content()
            )
        )
        cost = (
            k.acq_depth
            if k.acq_depth
            else 0
            if sig_level(k, self.signifier) == "S1"
            else 1
        )
        for u in self.signifier.units(arriving):
            self.acq[int(u)] = cost

    def _jaccard(self) -> float:
        cur, goal = self.content(), self.goal_content()
        union = self.signifier.measure(cur | goal)
        return self.signifier.measure(cur & goal) / union if union else 1.0

    def _finish(self, result: DerivationResult, ending: str) -> DerivationResult:
        result.ending = ending
        result.j1 = self._jaccard()
        node_depths = [
            max(
                (self.acq[int(u)] for u in self.signifier.units(n) if int(u) in self.acq),
                default=0,
            )
            for n in self.nodes
        ]
        result.hbar = sum(node_depths) / len(node_depths) if node_depths else 0.0
        result.dbar = 0.0  # policy A: no main-line witnessed expansion
        result.gamma = result.j1 * self.delta ** (result.dbar + result.hbar)
        result.composed = list(self.composed)
        return result
