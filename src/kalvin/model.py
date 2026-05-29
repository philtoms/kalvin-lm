"""Model — three-tier layered KLine collection (STM → Frame → Base).

The model provides storage, deduplication, lookup by signature, and
graph traversal. Significance computation and misfit classification
live in expand.py and misfit.py respectively, and are re-exported here
for backward compatibility.

See specs/model.md for the full specification.
"""

from __future__ import annotations

from typing import Callable, Iterator, NamedTuple

from kalvin.kline import KLine, KSig
from kalvin.stm import STM
from kalvin.signature import make_signature, signifies, is_literal_node

# Re-export from expand module for backward compatibility
from kalvin.expand import (
    QueryCandidate,
    D_MAX,
    MASK64,
    MAX_HOP,
    _S3_BIAS,
    _pack,
    expand as _expand_fn,
    is_canon as _is_canon_fn,
    is_s1 as _is_s1_fn,
    is_countersigned as _is_countersigned_fn,
    edge_hops as _edge_hops_fn,
    promote_participating as _promote_participating_fn,
)

# Re-export from misfit module for backward compatibility
from kalvin.misfit import (
    classify_misfit as _classify_misfit_fn,
    generate_expansions as _generate_expansions_fn,
)


class Model:
    """Three-tier Model: STM → Frame → Base.

    - STM: bounded rolling window (default 256). add() writes here.
    - Frame: populated by promote() from STM.
    - Base: optional long-term knowledge store (read-only, set at construction).

    The caller sees a single unified API. Significance and misfit logic
    are delegated to expand.py and misfit.py.
    """

    def __init__(self, base: Model | None = None, stm_bound: int = 256):
        self._base = base
        self._stm = STM(bound=stm_bound)

        # Frame storage — ordered dict for reverse-insertion-order iteration
        self._frame_list: list[KLine] = []
        self._frame_by_sig: dict[KSig, list[int]] = {}
        self._frame_dedup: set[tuple[KSig, tuple[int, ...]]] = set()

    # ── Storage Operations ────────────────────────────────────────────

    def add(self, kline: KLine, dedup: bool = False) -> bool:
        """Add a KLine to STM only.

        If kline.is_literal() and dedup=True, reject if an equal KLine
        exists in any tier. Non-literal Klines are always accepted.

        Returns True if added, False if rejected.
        """
        if dedup and kline.is_literal():
            if self._exists_any(kline):
                return False

        self._stm.add(kline, True)
        return True

    def exists(self, kline: KLine) -> bool:
        """Check if an equal KLine exists in any tier."""
        return self._exists_any(kline)

    def _exists_any(self, kline: KLine) -> bool:
        """Check STM, frame, then base."""
        if self._stm.contains(kline):
            return True
        if (kline.signature, tuple(kline.nodes)) in self._frame_dedup:
            return True
        if self._base:
            return self._base.exists(kline)
        return False

    def find(self, signature: KSig) -> KLine | None:
        """Find the most recently added KLine by signature.

        Searches STM, then frame, then base.
        """
        # STM first — filter for actual signature match (STM is dual-keyed)
        for kl in reversed(self._stm.find_by_signature(signature)):
            if kl.signature == signature:
                return kl
        # Frame
        indices = self._frame_by_sig.get(signature)
        if indices:
            for idx in reversed(indices):
                kl = self._frame_list[idx]
                if kl is not None:
                    return kl
        # Base
        if self._base:
            return self._base.find(signature)
        return None

    def find_all(self, signature: KSig) -> list[KLine]:
        """Return all KLines with the given signature across all tiers."""
        results: list[KLine] = []
        seen: set[tuple[KSig, tuple[int, ...]]] = set()

        # STM results — filter for actual signature match (STM is dual-keyed)
        for kl in self._stm.find_by_signature(signature):
            if kl.signature != signature:
                continue
            key = (kl.signature, tuple(kl.nodes))
            if key not in seen:
                seen.add(key)
                results.append(kl)

        # Frame results
        indices = self._frame_by_sig.get(signature, [])
        for idx in indices:
            kl = self._frame_list[idx]
            if kl is None:
                continue
            key = (kl.signature, tuple(kl.nodes))
            if key not in seen:
                seen.add(key)
                results.append(kl)

        # Base results
        if self._base:
            for kl in self._base.find_all(signature):
                key = (kl.signature, tuple(kl.nodes))
                if key not in seen:
                    seen.add(key)
                    results.append(kl)

        return results

    def find_by_nodes(self, nodes_signature: KSig) -> KLine | None:
        """Find the most recently added KLine whose nodes signature matches."""
        # STM first
        klines = self._stm.find_by_nodes(nodes_signature)
        if klines:
            return klines[-1]
        # Scan frame
        for kl in reversed(self._frame_list):
            ns = make_signature(kl.nodes)
            if ns == nodes_signature:
                return kl
        # Base
        if self._base:
            return self._base.find_by_nodes(nodes_signature)
        return None

    # ── Count ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of KLines in the frame (excluding STM and base)."""
        return sum(1 for kl in self._frame_list if kl is not None)

    def __iter__(self) -> Iterator[KLine]:
        return iter(kl for kl in self._frame_list if kl is not None)

    def __getitem__(self, signature: KSig) -> KLine | None:
        return self.find(signature)

    # ── Iteration ─────────────────────────────────────────────────────

    def klines(self) -> list[KLine]:
        """All KLines in reverse insertion order, deduplicated across tiers."""
        seen: set[tuple[KSig, tuple[int, ...]]] = set()
        results: list[KLine] = []

        # STM entries (most recent)
        for kl in reversed(self._stm):
            key = (kl.signature, tuple(kl.nodes))
            if key not in seen:
                seen.add(key)
                results.append(kl)

        # Frame entries not in STM
        for kl in reversed(self._frame_list):
            if kl is None:
                continue
            key = (kl.signature, tuple(kl.nodes))
            if key not in seen:
                seen.add(key)
                results.append(kl)

        # Base entries not in frame
        if self._base:
            for kl in self._base.klines():
                key = (kl.signature, tuple(kl.nodes))
                if key not in seen:
                    seen.add(key)
                    results.append(kl)

        return results

    def where(self, predicate: Callable[[KLine], bool] | KSig) -> list[KLine]:
        """Return KLines matching a predicate or signature overlap.

        If predicate is an int, it's treated as a signature for AND matching:
            where(sig) returns klines where kline.signature & sig != 0.
        """
        if isinstance(predicate, int):
            sig = predicate
            return [kl for kl in self.klines() if signifies(kl.signature, sig)]
        return [kl for kl in self.klines() if predicate(kl)]

    # ── Promotion ─────────────────────────────────────────────────────

    def promote(self, kline: KLine) -> bool:
        """Promote a KLine from STM to the frame.

        The KLine must currently be in STM. Returns True if added to frame,
        False if not in STM or already in frame.
        """
        key = (kline.signature, tuple(kline.nodes))
        # Must be in STM
        if not self._stm.contains(kline):
            return False
        # Already in frame
        if key in self._frame_dedup:
            return False
        # Add to frame
        idx = len(self._frame_list)
        self._frame_list.append(kline)
        if kline.signature not in self._frame_by_sig:
            self._frame_by_sig[kline.signature] = []
        self._frame_by_sig[kline.signature].append(idx)
        self._frame_dedup.add(key)
        return True

    def promote_all(self) -> int:
        """Promote all STM KLines to the frame."""
        count = 0
        for kl in self._stm.all_klines():
            if self.promote(kl):
                count += 1
        return count

    # ── Graph Traversal ───────────────────────────────────────────────

    def resolve(self, node: int) -> KLine | None:
        """Resolve a node value to a KLine."""
        return self.find(node)

    def query_expand(self, kline: KLine, depth: int = 2) -> list[KLine]:
        """Expand graph from kline up to *depth* levels.

        depth=0 → []
        depth=1 → []
        depth=2 → direct children
        depth=N → children up to N-1 levels deep.
        """
        if depth <= 1:
            return []
        visited: set[int] = set()
        results: list[KLine] = []
        self._query_expand_inner(kline, depth, 1, visited, results)
        return results

    def _query_expand_inner(
        self,
        kline: KLine,
        max_depth: int,
        current_depth: int,
        visited: set[int],
        results: list[KLine],
    ) -> None:
        if id(kline) in visited:
            return
        visited.add(id(kline))

        if current_depth >= max_depth:
            return

        for node in kline.nodes:
            child = self.find(node)
            if child is not None:
                results.append(child)
                self._query_expand_inner(child, max_depth, current_depth + 1, visited, results)

    def descendants(self, node: int) -> set[int]:
        """Recursively collect all descendant node values."""
        visited: set[int] = set()
        result: set[int] = set()
        self._descendants_inner(node, visited, result)
        return result

    def _descendants_inner(self, node: int, visited: set[int], result: set[int]) -> None:
        if node in visited:
            return
        visited.add(node)
        kline = self.find(node)
        if kline is None:
            return
        for child_node in kline.nodes:
            result.add(child_node)
            self._descendants_inner(child_node, visited, result)

    def query(self, signature: KSig, depth: int = 1) -> list[KLine]:
        """Find all KLines with signature, then expand each."""
        matches = self.find_all(signature)
        results: list[KLine] = list(matches)
        for kl in matches:
            results.extend(self.query_expand(kl, depth))
        return results

    # ── Significance API (delegated to expand.py) ─────────────────────

    def _is_canon(self, kline: KLine) -> bool:
        return _is_canon_fn(kline)

    def _edge_hops(self, sig: int) -> Iterator[tuple[int, int]]:
        return _edge_hops_fn(self, sig)

    def _as_kline(self, node: int) -> KLine:
        kline = self.find(node)
        if kline is None:
            raise ValueError(f"Node {node:#x} does not resolve to any KLine")
        return kline

    def expand(
        self,
        query: KLine,
        candidate: KLine,
        distance: int = 0,
        *,
        _visited: set[tuple[int, int]] | None = None,
    ) -> Iterator[QueryCandidate]:
        """Expand a query-candidate pair. Delegates to expand.expand()."""
        return _expand_fn(self, query, candidate, distance, _visited=_visited)

    def is_s1(self, kline: KLine) -> bool:
        """Determine if a kline is structurally grounded (S1). Delegates to expand.is_s1()."""
        return _is_s1_fn(self, kline)

    def is_countersigned(self, kline: KLine) -> bool:
        """Check if kline is countersigned. Delegates to expand.is_countersigned()."""
        return _is_countersigned_fn(self, kline)

    def promote_participating(self, query: KLine, candidate: KLine) -> int:
        """Promote all STM klines in a ratification event. Delegates to expand.promote_participating()."""
        return _promote_participating_fn(self, query, candidate)

    # ── Misfit API (delegated to misfit.py) ────────────────────────────

    def classify_misfit(self, kline: KLine) -> tuple[bool, bool]:
        """Classify a kline's misfit type. Delegates to misfit.classify_misfit()."""
        return _classify_misfit_fn(kline)

    def generate_expansions(
        self,
        kline: KLine,
        underfit_gap: int,
        overfit_mask: int,
    ) -> Iterator[tuple[KLine, list[KLine]]]:
        """Generate expansion proposals. Delegates to misfit.generate_expansions()."""
        return _generate_expansions_fn(self, kline, underfit_gap, overfit_mask)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def base(self) -> Model | None:
        return self._base

    # ── STM Interface ─────────────────────────────────────────────────

    def stm_contains(self, kline: KLine) -> bool:
        """Check if an equal KLine is in the STM (first tier only).

        Unlike ``exists()``, this only checks the STM tier — not the frame
        or base.
        """
        return self._stm.contains(kline)

    def iter_stm(self) -> Iterator[KLine]:
        """Iterate all KLines currently in the STM, in insertion order."""
        return self._stm.iter_all()

    # ── Compatibility ─────────────────────────────────────────────────

    def find_kline(self, signature: KSig) -> KLine | None:
        """Alias for find() — backwards compat."""
        return self.find(signature)

    def find_signed_klines(self, signature: KSig) -> list[KLine]:
        """Alias for find_all() — backwards compat."""
        return self.find_all(signature)

    def query_graph(self, query: KSig, depth: int = 1):
        """Alias for query() returning list — backwards compat."""
        return self.query(query, depth)

    def duplicate(self) -> Model:
        """Create a duplicate of this model's frame."""
        klines = [KLine(kl.signature, list(kl.nodes), kl.literal, kl.dbg_text)
                   for kl in self._frame_list if kl is not None]
        m = Model()
        for kl in klines:
            m.add(kl)
            m.promote(kl)
        return m

    def get_all_descendants(self, node: int, visited: set[int] | None = None) -> set[int]:
        """Backwards-compat alias for descendants()."""
        return self.descendants(node)

    @property
    def klines_prop(self) -> list[KLine]:
        """Backwards compat: return frame klines."""
        return [kl for kl in self._frame_list if kl is not None]

    def as_kline_list(self, limit: int = 0):
        """Backwards compat: iterate KLines."""
        items = [kl for kl in self._frame_list if kl is not None]
        if limit > 0:
            items = items[:limit]
        return reversed(items)

    def upgrade(self, kline: KLine, significance: KSig) -> None:
        """Upgrade significance — backwards compat."""
        kline.signature |= significance

    @property
    def kline(self) -> _KLineAccessor:
        return _KLineAccessor(self)


class _KLineAccessor:
    __slots__ = ("_model",)

    def __init__(self, model: Model):
        self._model = model

    def __getitem__(self, signature: KSig) -> KLine | None:
        return self._model.find(signature)
