"""Model — four-tier layered KLine collection (STM → Frame → LTM → Base).

The model provides storage, deduplication, lookup by signature, and
graph traversal. Significance computation and misfit classification
live in expand.py and misfit.py respectively.

See specs/model.md for the full specification.
"""

from __future__ import annotations

from typing import Callable, Iterator

from kalvin.kline import KLine, KSig
from kalvin.stm import STM
from kalvin.signature import make_signature, signifies, is_literal_node


class KLineStore:
    """Shared storage structure for Frame and LTM tiers.

    Maintains an indexed list with dedup set — supports O(1) membership
    checks and efficient lookup by signature while preserving insertion
    order for iteration.
    """

    def __init__(self) -> None:
        self._list: list[KLine | None] = []
        self._by_sig: dict[KSig, list[int]] = {}
        self._dedup: set[tuple[KSig, tuple[int, ...]]] = set()

    def add(self, kline: KLine) -> None:
        """Append a KLine to the store, indexing by signature and dedup key."""
        idx = len(self._list)
        self._list.append(kline)
        if kline.signature not in self._by_sig:
            self._by_sig[kline.signature] = []
        self._by_sig[kline.signature].append(idx)
        self._dedup.add((kline.signature, tuple(kline.nodes)))

    def contains(self, kline: KLine) -> bool:
        """Check if a KLine with matching signature and nodes exists."""
        return (kline.signature, tuple(kline.nodes)) in self._dedup

    def find(self, signature: KSig) -> KLine | None:
        """Find the most recently added KLine by signature."""
        indices = self._by_sig.get(signature)
        if indices:
            for idx in reversed(indices):
                kl = self._list[idx]
                if kl is not None:
                    return kl
        return None

    def find_all(self, signature: KSig) -> list[KLine]:
        """Return all KLines with the given signature in insertion order."""
        indices = self._by_sig.get(signature, [])
        results: list[KLine] = []
        for idx in indices:
            kl = self._list[idx]
            if kl is not None:
                results.append(kl)
        return results

    def __len__(self) -> int:
        """Count non-None entries."""
        return sum(1 for kl in self._list if kl is not None)

    def __iter__(self):
        """Yield non-None entries in insertion order."""
        for kl in self._list:
            if kl is not None:
                yield kl

    def __reversed__(self):
        """Yield non-None entries in reverse insertion order."""
        for kl in reversed(self._list):
            if kl is not None:
                yield kl

    def all_klines(self) -> list[KLine]:
        """Return list of all non-None entries in insertion order."""
        return [kl for kl in self._list if kl is not None]


class _TierAdapter:
    """Uniform read interface over STM, KLineStore, or Model tiers."""

    __slots__ = ("_tier", "_kind")

    def __init__(self, tier):
        self._tier = tier
        if isinstance(tier, STM):
            self._kind = "stm"
        elif isinstance(tier, KLineStore):
            self._kind = "store"
        else:
            self._kind = "model"

    def contains(self, kline: KLine) -> bool:
        if self._kind == "model":
            return self._tier.exists(kline)
        return self._tier.contains(kline)

    def find_first(self, sig: KSig) -> KLine | None:
        if self._kind == "stm":
            matches = self._tier.find_by_signature(sig)
            for kl in reversed(matches):
                if kl.signature == sig:
                    return kl
            return None
        # KLineStore and Model both have find(sig) → KLine|None
        return self._tier.find(sig)

    def find_all(self, sig: KSig) -> list[KLine]:
        if self._kind == "stm":
            return [kl for kl in self._tier.find_by_signature(sig) if kl.signature == sig]
        return self._tier.find_all(sig)

    def reversed_klines(self) -> Iterator[KLine]:
        if self._kind == "model":
            return iter(self._tier.klines())
        return reversed(self._tier)

    def find_by_nodes(self, nodes_sig: KSig) -> KLine | None:
        if self._kind == "stm":
            matches = self._tier.find_by_nodes(nodes_sig)
            return matches[-1] if matches else None
        if self._kind == "model":
            return self._tier.find_by_nodes(nodes_sig)
        # KLineStore: scan via __reversed__
        for kl in reversed(self._tier):
            if make_signature(kl.nodes) == nodes_sig:
                return kl
        return None


class _TierChain:
    """Ordered chain of tier adapters providing unified cross-tier search."""

    __slots__ = ("_adapters",)

    def __init__(self, tiers: list):
        self._adapters = [_TierAdapter(t) for t in tiers]

    def contains(self, kline: KLine) -> bool:
        return any(a.contains(kline) for a in self._adapters)

    def find_first(self, sig: KSig) -> KLine | None:
        for a in self._adapters:
            result = a.find_first(sig)
            if result is not None:
                return result
        return None

    def find_all(self, sig: KSig) -> list[KLine]:
        results: list[KLine] = []
        seen: set[tuple[KSig, tuple[int, ...]]] = set()
        for a in self._adapters:
            for kl in a.find_all(sig):
                key = (kl.signature, tuple(kl.nodes))
                if key not in seen:
                    seen.add(key)
                    results.append(kl)
        return results

    def all_klines(self) -> list[KLine]:
        """Deduplicated klines in tier-priority order (STM first, base last).

        Iterates adapters in forward order (stm → frame → ltm → base).
        Each adapter yields its own entries in reverse-insertion order
        (most recent first). Dedup ensures each unique kline appears once,
        with higher-priority tiers winning.
        """
        seen: set[tuple[KSig, tuple[int, ...]]] = set()
        results: list[KLine] = []
        for a in self._adapters:
            for kl in a.reversed_klines():
                key = (kl.signature, tuple(kl.nodes))
                if key not in seen:
                    seen.add(key)
                    results.append(kl)
        return results

    def find_by_nodes_first(self, nodes_sig: KSig) -> KLine | None:
        for a in self._adapters:
            result = a.find_by_nodes(nodes_sig)
            if result is not None:
                return result
        return None


class Model:
    """Four-tier Model: STM → Frame → LTM → Base.

    Write API (cascade methods):
    - add_stm(kl) — STM only. Always refreshes FIFO position.
    - add_frame(kl) — Frame + STM. Literal dedup guard on entry.
    - add_ltm(kl) — LTM + Frame + STM. Literal dedup guard on entry.

    Tier descriptions:
    - STM: bounded rolling window (default 256). Fast recency index.
    - Frame: working context. Additive, monotonic for non-literals.
    - LTM: long-term memory. Populated via add_ltm(). Additive.
    - Base: optional long-term knowledge store (read-only, set at construction).

    Read API (unchanged):
    - exists(kl), find(sig), find_all(sig), find_by_nodes(sig), where(pred)
    - klines(), iter_stm(), stm_contains(kl)

    The caller sees a single unified API. Significance and misfit logic
    live in expand.py and misfit.py respectively.

    Frame and LTM storage are backed by ``KLineStore`` — a reusable indexed list
    with dedup set.
    """

    def __init__(self, base: Model | None = None, ltm: Model | None = None, stm_bound: int = 256):
        self._base = base
        self._stm = STM(bound=stm_bound)

        # Frame storage — delegated to KLineStore
        self._frame: KLineStore = KLineStore()

        # LTM (Long-Term Memory) storage — populated from a previous session's model
        self._ltm: KLineStore = KLineStore()
        if ltm is not None:
            for kl in ltm.klines():
                self._ltm.add(kl)

        # Tier chain — unified cross-tier search
        tiers = [self._stm, self._frame, self._ltm]
        if base is not None:
            tiers.append(base)
        self._chain = _TierChain(tiers)

    # ── Storage Operations ────────────────────────────────────────────

    def add_stm(self, kline: KLine) -> None:
        """Write to STM only. Always refreshes FIFO (remove-if-present then add).

        Literal dedup guard: returns early if literal exists in any tier.
        Non-literal klines are always written.
        """
        if kline.is_literal() and self._exists_any(kline):
            return
        self._stm.add(kline)

    def add_frame(self, kline: KLine) -> None:
        """Write to Frame and STM. Literal dedup guard on entry.

        Frame and STM are both written unconditionally (after dedup check).
        Frame _dedup set is a membership index, not a write guard.
        Non-literal klines are always written.
        """
        if kline.is_literal() and self._exists_any(kline):
            return
        self._frame.add(kline)
        self._stm.add(kline)

    def add_ltm(self, kline: KLine) -> None:
        """Write to LTM, Frame, and STM. Literal dedup guard on entry.

        All three tiers written unconditionally (after dedup check).
        LTM and Frame _dedup sets are membership indexes, not write guards.
        Non-literal klines are always written.
        """
        if kline.is_literal() and self._exists_any(kline):
            return
        self._ltm.add(kline)
        self._frame.add(kline)
        self._stm.add(kline)

    def exists(self, kline: KLine) -> bool:
        """Check if an equal KLine exists in any tier."""
        return self._exists_any(kline)

    def _exists_any(self, kline: KLine) -> bool:
        """Check STM, Frame, LTM, then Base."""
        return self._chain.contains(kline)

    def find(self, signature: KSig) -> KLine | None:
        """Find the most recently added KLine by signature.

        Searches STM, then Frame, then LTM, then Base.
        """
        return self._chain.find_first(signature)

    def find_all(self, signature: KSig) -> list[KLine]:
        """Return all KLines with the given signature across all tiers."""
        return self._chain.find_all(signature)

    def find_by_nodes(self, nodes_signature: KSig) -> KLine | None:
        """Find the most recently added KLine whose nodes signature matches."""
        return self._chain.find_by_nodes_first(nodes_signature)

    # ── Count ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of KLines in the Frame (excluding STM, LTM, and Base)."""
        return len(self._frame)

    def __iter__(self) -> Iterator[KLine]:
        return iter(self._frame)

    def __getitem__(self, signature: KSig) -> KLine | None:
        return self.find(signature)

    # ── Iteration ─────────────────────────────────────────────────────

    def klines(self) -> list[KLine]:
        """All KLines in reverse insertion order, deduplicated across tiers.

        STM entries first (most recent), then Frame entries not in STM,
        then LTM entries not in Frame, then Base entries not in LTM.
        """
        return self._chain.all_klines()

    def where(self, predicate: Callable[[KLine], bool] | KSig) -> list[KLine]:
        """Return KLines matching a predicate or signature overlap.

        If predicate is an int, it's treated as a signature for AND matching:
            where(sig) returns klines where kline.signature & sig != 0.
        """
        if isinstance(predicate, int):
            sig = predicate
            return [kl for kl in self.klines() if signifies(kl.signature, sig)]
        return [kl for kl in self.klines() if predicate(kl)]



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
                   for kl in self._frame.all_klines()]
        m = Model()
        for kl in klines:
            m.add_frame(kl)
        return m

    def get_all_descendants(self, node: int, visited: set[int] | None = None) -> set[int]:
        """Backwards-compat alias for descendants()."""
        return self.descendants(node)

    @property
    def klines_prop(self) -> list[KLine]:
        """Backwards compat: return frame klines."""
        return self._frame.all_klines()

    def as_kline_list(self, limit: int = 0):
        """Backwards compat: iterate KLines."""
        items = self._frame.all_klines()
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
