"""Short-Term Memory (STM) — bounded, dual-keyed KLine index.

The STM indexes KLines by two keys:
  1. signature — kline.signature
  2. nodes_signature — make_signature(kline.nodes)

Both keys map into the same backing store. When the two keys are identical
the KLine is stored under a single key.

The STM has a configurable bound (default 256). When the bound is exceeded,
the oldest entries are evicted.
"""

from __future__ import annotations

from kalvin.kline import KLine, KSig, KNode
from kalvin.signature import make_signature, is_literal_node


class STM:
    """Short-Term Memory: bounded, dual-keyed KLine dictionary.

    Parameters
    ----------
    bound:
        Maximum number of KLines to retain. Default 256.
    """

    __slots__ = ("_store", "_order", "_dedup", "_bound")

    def __init__(self, bound: int = 256):
        self._store: dict[KSig, list[KLine]] = {}
        self._order: list[KLine] = []
        self._dedup: set[tuple[KSig, tuple[int, ...]]] = set()
        self._bound = bound

    # ── Core API ────────────────────────────────────────────────────────

    def add(self, kline: KLine, dedup: bool = True) -> bool:
        """Add a KLine to the STM.

        Indexes by both signature and nodes signature. Enforces bound
        via FIFO eviction.

        Returns:
            True if added, False if rejected as duplicate.
        """
        if dedup:
            key = (kline.signature, tuple(kline.nodes))
            if key in self._dedup:
                return False
            self._dedup.add(key)

        # Evict oldest if at bound
        while len(self._order) >= self._bound:
            self._evict_oldest()

        nodes_sig = make_signature(kline.nodes) if kline.nodes else 0
        sig = kline.signature or nodes_sig

        self._order.append(kline)
        self._append(sig, kline)
        if nodes_sig and nodes_sig != sig:
            self._append(nodes_sig, kline)
        return True

    def get(self, key: KSig) -> list[KLine]:
        """Return all KLines indexed under *key*, or an empty list."""
        return list(self._store.get(key, []))

    def get_kline(self, key: KSig) -> KLine | None:
        """Return the most recently added KLine under *key*, or None."""
        bucket = self._store.get(key)
        if bucket:
            return bucket[-1]
        return None

    def find_by_signature(self, signature: KSig) -> list[KLine]:
        return self.get(signature)

    def find_by_nodes(self, nodes_signature: KSig) -> list[KLine]:
        return self.get(nodes_signature)

    def query(self, sig: KSig) -> list[KLine]:
        """Return all KLines whose signatures overlap *sig* (AND ≠ 0)."""
        if sig == 0:
            return []
        seen: set[int] = set()
        results: list[KLine] = []
        n = len(self._order)
        for i in range(n - 1, -1, -1):
            kline = self._order[i]
            kid = id(kline)
            if kid not in seen and (kline.signature & sig) != 0:
                seen.add(kid)
                results.append(kline)
        return results

    def remove(self, kline: KLine) -> None:
        """Remove a KLine from all index entries."""
        sig = kline.signature
        nodes_sig = make_signature(kline.nodes) if kline.nodes else 0
        self._remove_from(sig, kline)
        if nodes_sig and nodes_sig != sig:
            self._remove_from(nodes_sig, kline)
        try:
            self._order.remove(kline)
        except ValueError:
            pass
        self._dedup.discard((sig, tuple(kline.nodes)))

    def contains(self, kline: KLine) -> bool:
        """Check if an equal KLine is in the STM (by signature + nodes)."""
        return (kline.signature, tuple(kline.nodes)) in self._dedup

    def all_klines(self) -> list[KLine]:
        """Return all KLines in insertion order."""
        return list(self._order)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()
        self._dedup.clear()

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, key: KSig) -> bool:
        return key in self._store

    def __iter__(self):
        return iter(self._order)

    def __reversed__(self):
        return reversed(self._order)

    def __repr__(self) -> str:
        return f"STM(klines={len(self._order)}, bound={self._bound})"

    # ── Internals ───────────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        """Evict the oldest KLine from the STM."""
        if not self._order:
            return
        oldest = self._order.pop(0)
        sig = oldest.signature
        nodes_sig = make_signature(oldest.nodes) if oldest.nodes else 0
        self._remove_from(sig, oldest)
        if nodes_sig and nodes_sig != sig:
            self._remove_from(nodes_sig, oldest)
        self._dedup.discard((sig, tuple(oldest.nodes)))

    def _append(self, key: KSig, kline: KLine) -> None:
        bucket = self._store.get(key)
        if bucket is None:
            self._store[key] = [kline]
        else:
            bucket.append(kline)

    def _remove_from(self, key: KSig, kline: KLine) -> None:
        bucket = self._store.get(key)
        if not bucket:
            return
        for i, existing in enumerate(bucket):
            if existing is kline:
                bucket.pop(i)
                break
        if not bucket:
            del self._store[key]
