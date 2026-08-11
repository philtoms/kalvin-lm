"""The similar-fit S2 strategy.

Shape one S2 proposal for a misfit entry by recombining grounded klines: (1)
node-expansion — replace each node that is a grounded kline's signature with
that kline's nodes; (2) node-graft — fold in each grounded kline sharing a
node value with the entry, resolving the accumulated target against it. Every
substituted node comes from a grounded kline (no invention). The entry
persists in the work-list until ratified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dialogue.engine_state import EngineState
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.significance import SIG_S2

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from kalvin.abstract import KSignifier

__all__ = ["SimilarFit"]


class SimilarFit:
    """The original S2 strategy: recombine grounded klines onto the entry."""

    def __init__(
        self,
        state: EngineState,
    ) -> None:
        self._state: EngineState = state

    @property
    def signifier(self) -> KSignifier:
        return self._state.signifier

    @property
    def state(self) -> EngineState:
        return self._state

    def propose(
        self,
        entry: KLine,
        ground: Callable[[KLine], None],
    ) -> list[KValue]:
        state = self.state
        target = _expand_nodes(state, list(entry.nodes))
        for candidate in state.similar_fit_candidates(entry):
            core = _resolve_against(state, target, list(candidate.nodes))
            if core:
                target = core + [n for n in candidate.nodes if n not in core]

        proposal = KLine(entry.signature, target)
        if state.is_in_ltm(proposal):
            return []
        return [KValue(proposal, SIG_S2)]


def _expand_nodes(state: EngineState, target: list[int]) -> list[int]:
    """Rule 1 — replace each node that is a grounded kline's signature with its nodes."""
    expanded: list[int] = []
    for node in target:
        sub = state.ltm_nodes(node)
        expanded.extend(sub if sub is not None else [node])
    return expanded


def _resolve_against(state: EngineState, target: list[int], candidate_nodes: list[int]) -> list[int]:
    """Rule 2 — the portion of ``target`` that resolves into ``candidate_nodes``.

    A target node resolves if it is in the candidate directly, or via a
    grounded kline whose signature is in the candidate. Unresolvable nodes
    drop out (open slots the caller fills from the candidate's surplus).
    Iterates to fixed point.
    """
    candidate_set = set(candidate_nodes)
    core: list[int] = []
    remaining = list(target)
    while True:
        direct = [n for n in remaining if n in candidate_set]
        failed = [n for n in remaining if n not in candidate_set]
        if not failed:
            return core + direct
        resolved = _cover_with_groundeds(state, failed)
        newly_matched = [n for n in resolved if n in candidate_set]
        core.extend(direct)
        core.extend(newly_matched)
        if not newly_matched:
            return core
        remaining = [n for n in resolved if n not in candidate_set]


def _cover_with_groundeds(state: EngineState, failed: list[int]) -> list[int]:
    """Maximally cover ``failed`` with disjoint grounded-kline node-sets.

    Each coverable subset is replaced by its kline's signature; uncoverable
    leftovers are passed through. Greedy is insufficient (a larger kline may
    block two smaller ones covering more), so this searches for a maximal
    disjoint cover.
    """
    failed_set = set(failed)
    covers: list[tuple[tuple[int, ...], int]] = []
    seen_sigs: set[int] = set()
    for bucket in state.ltm.values():
        for kline in bucket:
            if kline.signature in seen_sigs or not kline.nodes:
                continue
            effective = tuple(kline.nodes)
            if set(effective).issubset(failed_set):
                covers.append((effective, kline.signature))
                seen_sigs.add(kline.signature)

    best = _max_disjoint_cover(covers)
    if not best:
        return list(failed)
    # Map each covered node to its cover's signature, then walk ``failed``
    # in order so the output preserves the entry's node order (the order
    # nodes appeared in the expansion) rather than the cover-emission order.
    # A cover's signature is emitted once, at the position of its first
    # covered node; subsequent covered nodes of the same cover are dropped.
    node_to_sig: dict[int, int] = {}
    for canon_nodes, canon_sig in best:
        for n in canon_nodes:
            node_to_sig[n] = canon_sig
    resolved: list[int] = []
    emitted_sigs: set[int] = set()
    for n in failed:
        sig = node_to_sig.get(n)
        if sig is None:
            resolved.append(n)          # leftover (uncoverable)
        elif sig not in emitted_sigs:
            resolved.append(sig)         # first node of this cover
            emitted_sigs.add(sig)
    return resolved


def _max_disjoint_cover(
    covers: list[tuple[tuple[int, ...], int]]
) -> list[tuple[tuple[int, ...], int]]:
    """The disjoint subset of ``covers`` maximising total nodes covered."""
    best: list[tuple[tuple[int, ...], int]] = []
    best_covered = 0

    def _recurse(idx: int, chosen, used: set[int], covered: int) -> None:
        nonlocal best, best_covered
        if covered > best_covered:
            best_covered = covered
            best = list(chosen)
        for i in range(idx, len(covers)):
            kline_nodes, _ = covers[i]
            if used & set(kline_nodes):
                continue
            chosen.append(covers[i])
            _recurse(i + 1, chosen, used | set(kline_nodes), covered + len(kline_nodes))
            chosen.pop()

    _recurse(0, [], set(), 0)
    return best
