"""Model — three-tier layered KLine collection (STM → Frame → Base).

The model provides storage, deduplication, lookup by signature, graph
traversal, and the significance API functions consumed by the pipeline.

See specs/model.md for the full specification.
"""

from __future__ import annotations

from typing import Callable, Iterator

from kalvin.kline import KLine, KSig
from kalvin.stm import STM
from kalvin.signature import make_signature, signifies

# D_boundary hyperparameter — midpoint between S2 and S3 distance ranges
D_BOUNDARY = 0x8000_0000_0000_0000
D_MAX = 0xFFFF_FFFF_FFFF_FFFF

# MAX_HOP hyperparameter — upper bound on edge hop chain depth
MAX_HOP = 100


class Model:
    """Three-tier Model: STM → Frame → Base.

    - STM: bounded rolling window (default 256).
    - Frame: unbounded session write surface.
    - Base: optional long-term knowledge store (read-through).

    The caller sees a single unified API.
    """

    def __init__(self, base: Model | None = None, stm_bound: int = 256,
                 is_literal_fn: Callable[[int], bool] | None = None):
        self._base = base
        self._stm = STM(is_literal_fn=is_literal_fn, bound=stm_bound)
        self._is_literal_fn = is_literal_fn or (lambda _: False)

        # Frame storage — ordered dict for reverse-insertion-order iteration
        self._frame_list: list[KLine] = []
        self._frame_by_sig: dict[KSig, list[int]] = {}
        self._frame_dedup: set[tuple[KSig, tuple[int, ...]]] = set()

    def _make_sig(self, nodes: list[int]) -> int:
        return make_signature(nodes, self._is_literal_fn)

    # ── Storage Operations ────────────────────────────────────────────

    def add(self, kline: KLine, dedup: bool = False) -> bool:
        """Add a KLine to both STM and frame.

        If kline.is_literal() and dedup=True, reject if an equal KLine
        exists in any tier. Non-literal Klines are always accepted.

        Returns True if added, False if rejected.
        """
        if dedup and kline.is_literal():
            if self._exists_any(kline):
                return False

        # Add to frame
        idx = len(self._frame_list)
        self._frame_list.append(kline)
        if kline.signature not in self._frame_by_sig:
            self._frame_by_sig[kline.signature] = []
        self._frame_by_sig[kline.signature].append(idx)
        self._frame_dedup.add((kline.signature, tuple(kline.nodes)))

        # Add to STM
        self._stm.add(kline, dedup=False)

        return True

    def exists(self, kline: KLine) -> bool:
        """Check if an equal KLine exists in any tier."""
        return self._exists_any(kline)

    def _exists_any(self, kline: KLine) -> bool:
        """Check frame, then base (STM is a window over frame)."""
        key = (kline.signature, tuple(kline.nodes))
        if key in self._frame_dedup:
            return True
        if self._base:
            return self._base.exists(kline)
        return False

    def find(self, signature: KSig) -> KLine | None:
        """Find the most recently added KLine by signature."""
        # Frame first (exact signature match)
        indices = self._frame_by_sig.get(signature)
        if indices:
            # Walk backwards to find a non-None entry
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

        # Frame results
        indices = self._frame_by_sig.get(signature, [])
        for idx in indices:
            kl = self._frame_list[idx]
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
            ns = self._make_sig(kl.nodes)
            if ns == nodes_signature:
                return kl
        # Base
        if self._base:
            return self._base.find_by_nodes(nodes_signature)
        return None

    def remove(self, signature: KSig) -> bool:
        """Remove the most recently added KLine with the given signature.

        Removal never affects the base model.
        """
        # Try STM
        stm_klines = self._stm.find_by_signature(signature)
        if stm_klines:
            self._stm.remove(stm_klines[-1])

        # Try frame
        indices = self._frame_by_sig.get(signature)
        if indices:
            idx = indices.pop()
            kline = self._frame_list[idx]
            self._frame_dedup.discard((kline.signature, tuple(kline.nodes)))
            # Don't remove from _frame_list to preserve indices; just clear slot
            self._frame_list[idx] = None  # type: ignore
            if not indices:
                del self._frame_by_sig[signature]
            return True
        return False

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
        for kl in reversed(self._stm._order):
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
        """Promote a KLine to the base model."""
        if self._base is None:
            return False
        return self._base.add(kline, dedup=True)

    def promote_all(self) -> int:
        """Promote all frame KLines to the base model."""
        if self._base is None:
            return 0
        count = 0
        for kl in self._frame_list:
            if kl is not None and self._base.add(kl, dedup=True):
                count += 1
        return count

    # ── Graph Traversal ───────────────────────────────────────────────

    def resolve(self, node: int) -> KLine | None:
        """Resolve a node value to a KLine."""
        return self.find(node)

    def expand(self, kline: KLine, depth: int = 2) -> list[KLine]:
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
        self._expand_inner(kline, depth, 1, visited, results)
        return results

    def _expand_inner(
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
                self._expand_inner(child, max_depth, current_depth + 1, visited, results)

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
            results.extend(self.expand(kl, depth))
        return results

    # ── Significance API ──────────────────────────────────────────────

    def is_s1(self, node: int) -> bool:
        """Test whether a node value resolves to a kline in the model.

        A node achieves S1 when its value equals the signature of some
        kline stored in any tier.
        """
        return self.find(node) is not None

    def _is_canon(self, kline: KLine) -> bool:
        """Test whether a kline is canonical (signature = make_signature of nodes)."""
        return kline.signature == self._make_sig(kline.nodes)

    def _edge_hops(self, sig: int) -> Iterator[tuple[int, int]]:
        """Yield (hop_count, next_sig) for each non-canonical resolution step.

        Follows: resolve sig → kline → make_signature(kline.nodes) → repeat.
        Stops at a dead end (unresolvable) or a canonical kline.
        Yields (hop_count, next_sig) at each step.
        """
        hop_count = 0
        while hop_count < MAX_HOP:
            kline = self.find(sig)
            if kline is None or self._is_canon(kline):
                break
            hop_count += 1
            sig = self._make_sig(kline.nodes)
            yield hop_count, sig

    def s2_distance(self, query: KLine, candidate: KLine) -> int:
        """Distance when some nodes match. Returns value in [1, D_BOUNDARY).

        Per-node hop-distance algorithm with grounding credit:
        - Mismatched nodes contribute edge_hops (0 hops = MAX_HOP penalty)
        - Matched nodes that resolve to known klines credit -1 each
        """
        if not query.nodes:
            return 1

        q_set = set(query.nodes)
        c_set = set(candidate.nodes)
        mismatched_q = q_set - c_set
        mismatched_c = c_set - q_set
        matched = q_set & c_set

        distance = 0

        # Mismatched query nodes: find hops that land in mismatched_c
        for n in mismatched_q:
            hop_distance = MAX_HOP
            for hops, match_sig in self._edge_hops(n):
                if match_sig in mismatched_c:
                    hop_distance = hops
                    break
            distance += hop_distance

        # Mismatched candidate nodes: find hops that land in mismatched_q
        for n in mismatched_c:
            hop_distance = MAX_HOP
            for hops, match_sig in self._edge_hops(n):
                if match_sig in mismatched_q:
                    hop_distance = hops
                    break
            distance += hop_distance

        # Grounding credit: matched nodes that resolve to known klines
        for n in matched:
            if self.is_s1(n):
                distance -= 1

        return max(1, min(int(distance), D_BOUNDARY - 1))

    def s3_distance(self, query: KLine, candidate: KLine) -> int:
        """Distance when no nodes achieve S1. Returns value in [D_BOUNDARY, D_MAX)."""
        # Simple heuristic: use bit overlap ratio
        if not query.nodes:
            return D_BOUNDARY
        q_sig = self._make_sig(query.nodes)
        c_sig = candidate.signature
        if q_sig == 0:
            return D_MAX - 1
        overlap = bin(q_sig & c_sig).count("1")
        total = bin(q_sig | c_sig).count("1")
        if total == 0:
            return D_MAX - 1
        ratio = overlap / total
        distance = D_BOUNDARY + int((1 - ratio) * (D_MAX - D_BOUNDARY))
        return max(D_BOUNDARY, min(distance, D_MAX - 1))

    def is_countersigned(self, a: KLine, b: KLine) -> bool:
        """Test whether two Klines are countersigned (mutual reference)."""
        return (b.signature in a.nodes) and (a.signature in b.nodes)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def base(self) -> Model | None:
        return self._base

    @property
    def stm(self) -> STM:
        return self._stm

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
        m = Model(is_literal_fn=self._is_literal_fn)
        for kl in klines:
            m.add(kl)
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
