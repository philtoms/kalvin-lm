"""Short-Term Memory (STM) — bounded, dual-keyed KLine index.

The STM indexes KLines by two keys:
  1. signature — kline.signature
  2. nodes_signature — make_signature(kline.nodes)

Both keys map into the same backing store. When the two keys are identical
the KLine is stored under a single key.

The STM has a configurable bound (default 256). When the bound is exceeded,
the oldest entries are evicted.

Thread safety
-------------
All public methods are guarded by a re-entrant lock (:class:`threading.RLock`)
so each operation is individually atomic. A re-entrant lock is required because
public methods call other public methods (``add`` → ``remove``; ``find_*`` →
``get``); a plain ``Lock`` would deadlock on re-entry.

Iterator-returning methods (``iter_all``, ``__iter__``, ``__reversed__``)
materialise a snapshot **under the lock** and return an iterator over that
snapshot, so a caller that iterates after the lock is released still observes a
consistent point-in-time view (no live-list mutation mid-iteration).

Lock ordering: an :class:`~kalvin.model.Model` always acquires its own lock
*before* acquiring ``STM._lock`` (Model → STM). STM never calls back into a
Model, so the ordering is acyclic and deadlock-free. ``STM._lock`` is a member
added to ``__slots__``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

from kalvin.kline import KLine, KSig
from kalvin.signifier import NLPSignifier

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier


class STM:
    """Short-Term Memory: bounded, dual-keyed KLine dictionary.

    Parameters
    ----------
    bound:
        Maximum number of KLines to retain. Default 256.
    signifier:
        KSignifier for computing nodes signatures. Defaults to NLPSignifier().

    Thread safety: see module docstring. All public methods are guarded by
    ``self._lock`` (an :class:`threading.RLock`); iterator-returning methods
    materialise a snapshot under the lock.
    """

    __slots__ = ("_store", "_order", "_dedup", "_bound", "_lock", "_signifier")

    def __init__(self, bound: int = 256, signifier: KSignifier | None = None):
        self._store: dict[KSig, list[KLine]] = {}
        self._order: list[KLine] = []
        self._dedup: set[tuple[KSig, tuple[int, ...]]] = set()
        self._bound = bound
        self._lock = threading.RLock()
        self._signifier = signifier or NLPSignifier()

    # Core API

    def add(self, kline: KLine) -> None:
        """Add a KLine to the STM.

        Indexes by both signature and nodes signature. Enforces bound
        via FIFO eviction. If an equal KLine already exists (same
        signature + nodes), the old entry is removed first and the
        new one added fresh (FIFO position refreshed).
        """
        with self._lock:
            key = (kline.signature, tuple(kline.nodes))
            if key in self._dedup:
                # Find the actual stored object (identity may differ from kline)
                for existing in reversed(self._order):
                    if existing == kline:
                        self.remove(existing)
                        break
            self._dedup.add(key)

            # Evict oldest if at bound
            while len(self._order) >= self._bound:
                self._evict_oldest()

            nodes_sig = self._signifier.make_signature(kline.nodes) if kline.nodes else 0
            sig = kline.signature or nodes_sig

            self._order.append(kline)
            self._append(sig, kline)
            if nodes_sig and nodes_sig != sig:
                self._append(nodes_sig, kline)

    def get(self, key: KSig) -> list[KLine]:
        """Return all KLines indexed under *key*, or an empty list."""
        with self._lock:
            return list(self._store.get(key, []))

    def get_kline(self, key: KSig) -> KLine | None:
        """Return the most recently added KLine under *key*, or None."""
        with self._lock:
            bucket = self._store.get(key)
            if bucket:
                return bucket[-1]
            return None

    def find_by_signature(self, signature: KSig) -> list[KLine]:
        with self._lock:
            return self.get(signature)

    def find_by_nodes(self, nodes_signature: KSig) -> list[KLine]:
        with self._lock:
            return self.get(nodes_signature)

    def query(self, sig: KSig) -> list[KLine]:
        """Return all KLines whose signatures overlap *sig* (AND ≠ 0)."""
        with self._lock:
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
        with self._lock:
            sig = kline.signature
            nodes_sig = self._signifier.make_signature(kline.nodes) if kline.nodes else 0
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
        with self._lock:
            return (kline.signature, tuple(kline.nodes)) in self._dedup

    def all_klines(self) -> list[KLine]:
        """Return all KLines in insertion order."""
        with self._lock:
            return list(self._order)

    def iter_all(self) -> Iterator[KLine]:
        """Iterate all KLines in insertion order.

        Returns an iterator over a snapshot taken under the lock, so it is
        safe to iterate after the lock is released.
        """
        with self._lock:
            return iter(list(self._order))

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._order.clear()
            self._dedup.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._order)

    def __contains__(self, key: KSig) -> bool:
        with self._lock:
            return key in self._store

    def __iter__(self):
        # Snapshot under the lock; the returned iterator is over a private
        # copy, so it is safe to exhaust after the lock is released.
        with self._lock:
            return iter(list(self._order))

    def __reversed__(self):
        with self._lock:
            return iter(list(reversed(self._order)))

    def __repr__(self) -> str:
        # Reads only atomic scalars (len of a list + an int bound); no guard
        # needed. Avoids acquiring the lock from repr/debug paths.
        return f"STM(klines={len(self._order)}, bound={self._bound})"

    # Internals

    def _evict_oldest(self) -> None:
        """Evict the oldest KLine from the STM."""
        if not self._order:
            return
        oldest = self._order.pop(0)
        sig = oldest.signature
        nodes_sig = self._signifier.make_signature(oldest.nodes) if oldest.nodes else 0
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
            if existing == kline:
                bucket.pop(i)
                break
        if not bucket:
            del self._store[key]
