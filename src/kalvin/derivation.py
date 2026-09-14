"""The derivation model — kalvin-algebra §6–10 as an engine core.

A Derivation rewrites the node sequence of a queued kline under held
correspondences, relative to a goal kline (Def 12). The queued head rides
inert; only the nodes change. Memory grows as slot walks write composed
correspondences (Def 17, progressive path).

The run loop implements policy A — the §9 documented order
(canonicalisation → targeting → slot walk). The enumerators are the
mechanism/policy boundary: they yield licensed options in deterministic
order; the loop chooses.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations

from kalvin.abstract import KSignifier
from kalvin.kline import KLine, is_canon, is_terminal, sig_level
from kalvin.significance import (
    DEFAULT_DELTA,
    WORD_BITS,
    misfit_mass,
    word_atom_count,
)

#: T2-class strategy bound: total rewrites per run.
MAX_STEPS = 32
#: T2-class strategy bound: edges per slot walk.
MAX_WALK_EDGES = 8


@dataclass
class DerivationResult:
    """A run's outcome: ending, trace, and the §11 measurement."""

    ending: str  # "done" | "stuck" | "abandoned"
    trace: list[list[int]] = field(default_factory=list)
    composed: list[KLine] = field(default_factory=list)
    j0: float = 0.0
    j1: float = 0.0
    dbar: float = 0.0
    hbar: float = 0.0
    gamma: float = 0.0


def _atom_bits(value: int) -> Iterator[int]:
    v = value & WORD_BITS
    while v:
        b = v & -v
        v ^= b
        yield b


class Derivation:
    """A ⊢_{M,B} … — Defs 12–17 over held memory, on engine primitives."""

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
        self.nodes: list[int] = list(queued.nodes)
        self.composed: list[KLine] = []
        self.acq: dict[int, int] = {}  # atom bit -> acquisition depth (§11)

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
        """Unknown has no witness; Identity is inert."""
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
        smallest group first (Def 13, canonicalisation)."""
        n = len(self.nodes)
        for size in range(2, n):
            for idxs in combinations(range(n), size):
                group = tuple(int(self.nodes[i]) for i in idxs)
                for k in self.memory:
                    if not is_canon(k, self.signifier):
                        continue
                    if Counter(int(x) for x in k.nodes) != Counter(group):
                        continue
                    new = [x for i, x in enumerate(self.nodes) if i not in idxs]
                    new.insert(idxs[0], k.signature)
                    yield k, group, new

    def targetings(self) -> Iterator[tuple[KLine, str, list[int], int, int]]:
        """Licensed targeting replaces with strictly falling misfit mass
        (Def 14). In an S2 region the departure is restricted to the misfit
        region. Order: memory then composed, forward before reverse."""
        cur = self.content()
        d0 = self.mismatch()
        gap = self.gap()
        s2 = self.relationship_band() == "S2"
        for k in self.memory + self.composed:
            if not self.usable(k):
                continue
            if k.signature in self.nodes:
                if s2 and not self.signifier.signifies(k.signature, gap):
                    continue
                new = self.replace_fwd(k, self.nodes)
                d1 = misfit_mass(
                    int(self.signifier.signature_of(new)), self.goal_content()
                )
                if d1 < d0:
                    yield k, "forward", new, d0, d1
            if self.occurs_rev(k, self.nodes):
                if s2 and not all(self.signifier.signifies(n, gap) for n in k.nodes):
                    continue
                new = self.replace_rev(k, self.nodes)
                d1 = misfit_mass(
                    int(self.signifier.signature_of(new)), self.goal_content()
                )
                if d1 < d0:
                    yield k, "reverse", new, d0, d1

    def slot_walk(
        self, slot: int
    ) -> tuple[list[int], int, list[tuple[str, KLine]]] | None:
        """Goal-less walk from the slot identity, licensed by occurrence on
        either side (Def 17). T2 no-revisit keys on correspondence identity
        — signature together with witness, not signature alone."""
        excess = self.excess()
        queue: deque = deque([([slot], frozenset(), 0, [])])
        seen = {(int(slot),)}
        while queue:
            nodes, used, edges, path = queue.popleft()
            if edges and self.signifier.signifies(
                int(self.signifier.signature_of(nodes)), excess
            ):
                return nodes, edges, path
            if edges >= self.max_walk_edges:
                continue
            for k in self.memory:
                if not self.usable(k):
                    continue
                key = (int(k.signature), tuple(int(n) for n in k.nodes))
                if key in used:
                    continue
                moves = []
                if k.signature in nodes and (
                    not is_canon(k, self.signifier) or self.wellfounded(k)
                ):
                    moves.append(("forward", self.replace_fwd(k, nodes)))
                if self.occurs_rev(k, nodes):
                    moves.append(("reverse", self.replace_rev(k, nodes)))
                for direction, new in moves:
                    t = tuple(sorted(int(x) for x in new))
                    if t in seen:
                        continue
                    seen.add(t)
                    queue.append((new, used | {key}, edges + 1, path + [(direction, k)]))
        return None

    def refine(
        self, nodes: list[int], edges: int, path: list[tuple[str, KLine]]
    ) -> tuple[list[int], int, list[tuple[str, KLine]]]:
        """Write granularity (Def 17): expand the walk's nodes covering the
        excess toward the goal's witness resolution, under held
        well-founded canons. Each expansion is an edge."""
        excess = self.excess()
        while True:
            for i, n in enumerate(nodes):
                kanon = next(
                    (
                        k
                        for k in self.memory
                        if k.signature == n
                        and is_canon(k, self.signifier)
                        and self.wellfounded(k)
                    ),
                    None,
                )
                if kanon is None or not self.signifier.signifies(n, excess):
                    continue
                nodes = nodes[:i] + list(kanon.nodes) + nodes[i + 1 :]
                path = path + [("expand", kanon)]
                edges += 1
                break
            else:
                return nodes, edges, path

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
        """Take the first contraction that exposes an applicable misfit
        correspondence (the §9 survey condition)."""
        for _, _, new in self.canonicalisations():
            if self._exposes(new):
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
        for slot in self.nodes:
            if not self.signifier.signifies(slot, self.gap()):
                continue
            walk = self.slot_walk(slot)
            if walk is None:
                continue
            end, edges, path = walk
            end, edges, _ = self.refine(end, edges, path)
            composed = KLine(slot, end, acq_depth=edges)
            self.memory.append(composed)
            self.composed.append(composed)
            return True
        return False

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
            if k in self.composed
            else 0
            if sig_level(k, self.signifier) == "S1"
            else 1
        )
        for b in _atom_bits(arriving):
            self.acq[b] = cost

    def _jaccard(self) -> float:
        cur, goal = self.content(), self.goal_content()
        union = word_atom_count(cur | goal)
        return word_atom_count(cur & goal) / union if union else 1.0

    def _finish(self, result: DerivationResult, ending: str) -> DerivationResult:
        result.ending = ending
        result.j1 = self._jaccard()
        node_depths = [
            max((self.acq[b] for b in _atom_bits(n) if b in self.acq), default=0)
            for n in self.nodes
        ]
        result.hbar = sum(node_depths) / len(node_depths) if node_depths else 0.0
        result.dbar = 0.0  # policy A: no main-line witnessed expansion
        result.gamma = result.j1 * self.delta ** (result.dbar + result.hbar)
        result.composed = list(self.composed)
        return result
