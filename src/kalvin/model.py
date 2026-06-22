"""Model — four-tier layered KLine collection (STM → Frame → LTM → Base).

The model provides storage, deduplication, lookup by signature, and
graph traversal. Significance computation and misfit classification
live in expand.py and misfit.py respectively.

See specs/model.md for the full specification.

Thread safety
-------------
Concurrent access (e.g. the background Cogitator thread mutating the model
while a subscriber reading the model from ``on_event`` runs on the same
thread) is made safe by encapsulating all locking inside the data structures:

- ``Model``, ``KLineStore`` and :class:`~kalvin.stm.STM` each hold their own
  :class:`threading.RLock`. Every public mutator/reader takes its lock for the
  whole operation, making each call individually atomic. Re-entrant locks are
  required because public methods call other public methods on the same object
  (e.g. ``Model.unpack`` → ``_resolve_for_unpack`` → ``klines`` → tier chain;
  ``STM.add`` → ``remove``); a plain ``Lock`` would deadlock on re-entry.
- Iterator-returning methods (``Model.__iter__``, ``Model.iter_stm``,
  ``KLineStore.__iter__``/``__reversed__``) snapshot the backing list **under
  the lock** and return an iterator over the snapshot, so callers that iterate
  after the lock is released still observe a consistent point-in-time view.
- Lock ordering is strictly **Model → inner tier** (STM / KLineStore / a base
  ``Model``). ``Model`` always acquires its own lock first, then — via the tier
  chain/adapters or directly from ``add_to_*`` — the inner tier's lock. Inner
  tiers never call back into a ``Model``, so the ordering is acyclic and
  deadlock-free. Cogitator/agent code therefore needs no locking of its own.

See specs/model.md §Thread Safety for the full contract.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from kalvin.kline import KLine, KSig, is_canon, is_identity
from kalvin.signifier import NLPSignifier
from kalvin.stm import STM

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier


class KLineStore:
    """Shared storage structure for Frame and LTM tiers.

    Maintains an indexed list with dedup set — supports O(1) membership
    checks and efficient lookup by signature while preserving insertion
    order for iteration.

    Thread safety: all public methods are guarded by ``self._lock`` (an
    :class:`threading.RLock`). Iterator-returning methods (``__iter__``,
    ``__reversed__``) materialise a snapshot under the lock and return an
    iterator over that snapshot, so a caller iterating after the lock is
    released observes a consistent point-in-time view. A ``Model`` always
    acquires its own lock before this store's lock (Model → KLineStore).
    """

    def __init__(self) -> None:
        self._list: list[KLine | None] = []
        self._by_sig: dict[KSig, list[int]] = {}
        self._dedup: set[tuple[KSig, tuple[int, ...]]] = set()
        self._lock = threading.RLock()

    def add(self, kline: KLine) -> None:
        """Append a KLine to the store, indexing by signature and dedup key."""
        with self._lock:
            idx = len(self._list)
            self._list.append(kline)
            if kline.signature not in self._by_sig:
                self._by_sig[kline.signature] = []
            self._by_sig[kline.signature].append(idx)
            self._dedup.add((kline.signature, tuple(kline.nodes)))

    def contains(self, kline: KLine) -> bool:
        """Check if a KLine with matching signature and nodes exists."""
        with self._lock:
            return (kline.signature, tuple(kline.nodes)) in self._dedup

    def find(self, signature: KSig) -> KLine | None:
        """Find the most recently added KLine by signature."""
        with self._lock:
            indices = self._by_sig.get(signature)
            if indices:
                for idx in reversed(indices):
                    kl = self._list[idx]
                    if kl is not None:
                        return kl
            return None

    def find_all(self, signature: KSig) -> list[KLine]:
        """Return all KLines with the given signature in insertion order."""
        with self._lock:
            indices = self._by_sig.get(signature, [])
            results: list[KLine] = []
            for idx in indices:
                kl = self._list[idx]
                if kl is not None:
                    results.append(kl)
            return results

    def __len__(self) -> int:
        """Count non-None entries."""
        with self._lock:
            return sum(1 for kl in self._list if kl is not None)

    def __iter__(self):
        """Yield non-None entries in insertion order.

        Returns an iterator over a snapshot taken under the lock, so it is
        safe to iterate after the lock is released.
        """
        with self._lock:
            return iter([kl for kl in self._list if kl is not None])

    def __reversed__(self):
        """Yield non-None entries in reverse insertion order.

        Returns an iterator over a snapshot taken under the lock.
        """
        with self._lock:
            return iter([kl for kl in reversed(self._list) if kl is not None])

    def all_klines(self) -> list[KLine]:
        """Return list of all non-None entries in insertion order."""
        with self._lock:
            return [kl for kl in self._list if kl is not None]


class _TierAdapter:
    """Uniform read interface over STM, KLineStore, or Model tiers."""

    __slots__ = ("_tier", "_kind", "_signifier")

    def __init__(self, tier, signifier: KSignifier):
        self._tier = tier
        self._signifier = signifier
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
            if self._signifier.make_signature(kl.nodes) == nodes_sig:
                return kl
        return None


class _TierChain:
    """Ordered chain of tier adapters providing unified cross-tier search."""

    __slots__ = ("_adapters", "_signifier")

    def __init__(self, tiers: list, signifier: KSignifier):
        self._signifier = signifier
        self._adapters = [_TierAdapter(t, signifier) for t in tiers]

    def contains(self, kline: KLine) -> bool:
        return any(a.contains(kline) for a in self._adapters)

    def contains_excluding_first(self, kline: KLine) -> bool:
        """Check all tiers except the first (STM).

        Used by Model.grounded() to check Frame, LTM, and Base without
        matching transient STM entries.
        """
        return any(a.contains(kline) for a in self._adapters[1:])

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
    - add_to_stm(kl) — STM only. Always refreshes FIFO position.
    - add_to_frame(kl) — Frame + STM. Unconditional write.
    - add_to_ltm(kl) — LTM + Frame + STM. Unconditional write.

    Tier descriptions:
    - STM: bounded rolling window (default 256). Fast recency index.
    - Frame: working context. Additive, monotonic.
    - LTM: long-term memory. Populated via add_to_ltm(). Additive.
    - Base: optional long-term knowledge store (read-only, set at construction).

    Read API (unchanged):
    - exists(kl), find(sig), find_all(sig), find_by_nodes(sig), where(pred)
    - klines(), iter_stm(), stm_contains(kl)

    The caller sees a single unified API. Significance and misfit logic
    live in expand.py and misfit.py respectively.

    Frame and LTM storage are backed by ``KLineStore`` — a reusable indexed list
    with dedup set.

    Thread safety: see module docstring. All public methods are guarded by
    ``self._lock`` (an :class:`threading.RLock`); iterator-returning methods
    materialise a snapshot under the lock. Lock ordering is strictly
    Model → inner tier (STM / KLineStore / base Model).
    """

    def __init__(self, base: Model | None = None, ltm: Model | None = None, stm_bound: int = 256, signifier: KSignifier | None = None):
        # The lock must exist before any guarded public method can run. It is
        # established first, ahead of building the tiers. (Construction itself
        # is single-threaded — the object is not published until __init__
        # returns — so the direct internal writes below need no synchronisation
        # against other threads.)
        self._lock = threading.RLock()
        self._signifier = signifier or NLPSignifier()
        self._base = base
        self._stm = STM(bound=stm_bound, signifier=self._signifier)

        self._frame: KLineStore = KLineStore()

        # Populated from a previous session's model.
        self._ltm: KLineStore = KLineStore()
        if ltm is not None:
            for kl in ltm.klines():
                self._ltm.add(kl)

        tiers = [self._stm, self._frame, self._ltm]
        if base is not None:
            tiers.append(base)
        self._chain = _TierChain(tiers, self._signifier)

    # Storage Operations

    def add_to_stm(self, kline: KLine) -> None:
        """Write to STM only. Always refreshes FIFO (remove-if-present then add)."""
        with self._lock:
            self._stm.add(kline)

    def add_to_frame(self, kline: KLine) -> None:
        """Write to Frame and STM.

        Frame and STM are both written unconditionally.
        Frame _dedup set is a membership index, not a write guard.
        """
        with self._lock:
            self._frame.add(kline)
            self._stm.add(kline)

    def add_to_ltm(self, kline: KLine) -> None:
        """Write to LTM, Frame, and STM.

        All three tiers written unconditionally.
        LTM and Frame _dedup sets are membership indexes, not write guards.
        """
        with self._lock:
            self._ltm.add(kline)
            self._frame.add(kline)
            self._stm.add(kline)

    def exists(self, kline: KLine) -> bool:
        """Check if an equal KLine exists in any tier."""
        with self._lock:
            return self._exists_any(kline)

    def grounded(self, kline: KLine) -> bool:
        """Check if an equal KLine exists in Frame, LTM, or Base (not STM).

        STM is transient — entries pre-registered there haven't been
        rationalised yet and shouldn't count as grounded knowledge.
        """
        with self._lock:
            return self._chain.contains_excluding_first(kline)

    def _exists_any(self, kline: KLine) -> bool:
        """Check STM, Frame, LTM, then Base. Runs under the caller's lock."""
        return self._chain.contains(kline)

    def find(self, signature: KSig) -> KLine | None:
        """Find the most recently added KLine by signature.

        Searches STM, then Frame, then LTM, then Base.
        """
        with self._lock:
            return self._chain.find_first(signature)

    def find_all(self, signature: KSig) -> list[KLine]:
        """Return all KLines with the given signature across all tiers."""
        with self._lock:
            return self._chain.find_all(signature)

    def find_by_nodes(self, nodes_signature: KSig) -> KLine | None:
        """Find the most recently added KLine whose nodes signature matches."""
        with self._lock:
            return self._chain.find_by_nodes_first(nodes_signature)

    # Count

    def __len__(self) -> int:
        """Number of KLines in the Frame (excluding STM, LTM, and Base)."""
        with self._lock:
            return len(self._frame)

    def __iter__(self) -> Iterator[KLine]:
        # Snapshot the frame under the lock; the returned iterator is over a
        # private copy, so it is safe to exhaust after the lock is released.
        with self._lock:
            return iter(self._frame.all_klines())

    def __getitem__(self, signature: KSig) -> KLine | None:
        with self._lock:
            return self.find(signature)

    # Iteration

    def klines(self) -> list[KLine]:
        """All KLines in reverse insertion order, deduplicated across tiers.

        STM entries first (most recent), then Frame entries not in STM,
        then LTM entries not in Frame, then Base entries not in LTM.
        """
        with self._lock:
            return self._chain.all_klines()

    def where(self, predicate: Callable[[KLine], bool] | KSig) -> list[KLine]:
        """Return KLines matching a predicate or signature overlap.

        If predicate is an int, it's treated as a signature for AND matching:
            where(sig) returns klines where kline.signature & sig != 0.
        """
        with self._lock:
            if isinstance(predicate, int):
                sig = predicate
                return [kl for kl in self.klines() if self._signifier.signifies(kl.signature, sig)]
            return [kl for kl in self.klines() if predicate(kl)]

    # Graph Traversal

    def resolve(self, node: int) -> KLine | None:
        """Resolve a node value to a KLine."""
        with self._lock:
            return self.find(node)

    def query_expand(self, kline: KLine, depth: int = 2) -> list[KLine]:
        """Expand graph from kline up to *depth* levels.

        depth=0 → []
        depth=1 → []
        depth=2 → direct children
        depth=N → children up to N-1 levels deep.
        """
        with self._lock:
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
        """Runs under the caller's lock; calls guarded self.find (RLock re-entry)."""
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

    def unpack(self, kline: KLine) -> list[int]:
        """Flatten a kline's signature decomposition to identity signatures.

        Walks the kline's node tree, returning an ordered list of the
        single-token identity signatures it decomposes into. See
        @specs/model.md §Graph Traversal › Unpack.

        - Identity (``is_identity`` — empty nodes OR self-referential
          ``{S: [S]}``) → [signature]. Base case.
        - Canon (``is_canon`` — non-empty, non-self-referential, signature ==
          make_signature(nodes)) → concatenation of unpack(child) per node,
          in node order.
        - Any other input (connoted, undersigned, misfit) → ValueError.
        - Child resolution: identity preferred over canon; within a kind,
          most recently added wins (Recency Precedence).
        - Raises ValueError if a child node resolves to no identity or canon.
        """
        with self._lock:
            if is_identity(kline):
                return [kline.signature]
            if not is_canon(kline, self._signifier):
                raise ValueError(
                    f"unpack: input kline {kline.signature:#x} is not decomposable "
                    f"(not identity, not canon)"
                )
            out: list[int] = []
            for node in kline.nodes:
                child = self._resolve_for_unpack(node)
                out.extend(self.unpack(child))
            return out

    def _resolve_for_unpack(self, node: int) -> KLine:
        """Resolve a node value to its identity or canon kline. Runs under caller's lock.

        Precedence (highest first):
          1. empty-nodes identity,
          2. genuine canon,
          3. self-referential identity ``{S: [S]}``.

        The self-referential form is identity by :func:`is_identity`, but it
        carries no decomposition, so it loses to a genuine canon for the same
        signature — otherwise it would displace the canon and collapse
        ``unpack()`` to identity (the degenerate-canons bug). Within each kind
        the most recently added kline wins (Recency Precedence); ``klines()``
        yields most-recent-first. Raises ValueError if the node resolves to
        none of the three.
        """
        empty_identity: KLine | None = None
        canon: KLine | None = None
        self_identity: KLine | None = None
        for kl in self.klines():
            if kl.signature != node:
                continue
            if not kl.nodes:
                if empty_identity is None:
                    empty_identity = kl
            elif is_canon(kl, self._signifier):
                if canon is None:
                    canon = kl
            elif is_identity(kl):
                # Self-referential {S: [S]} — identity, but lowest precedence.
                if self_identity is None:
                    self_identity = kl
        if empty_identity is not None:
            return empty_identity
        if canon is not None:
            return canon
        if self_identity is not None:
            return self_identity
        raise ValueError(f"unpack: node {node:#x} resolves to no identity or canon kline")

    def query(self, signature: KSig, depth: int = 1) -> list[KLine]:
        """Find all KLines with signature, then expand each."""
        with self._lock:
            matches = self.find_all(signature)
            results: list[KLine] = list(matches)
            for kl in matches:
                results.extend(self.query_expand(kl, depth))
            return results

    # Properties

    @property
    def base(self) -> Model | None:
        return self._base

    # STM Interface

    def stm_contains(self, kline: KLine) -> bool:
        """Check if an equal KLine is in the STM (first tier only).

        Unlike ``exists()``, this only checks the STM tier — not the frame
        or base.
        """
        with self._lock:
            return self._stm.contains(kline)

    def iter_stm(self) -> Iterator[KLine]:
        """Iterate all KLines currently in the STM, in insertion order.

        Returns an iterator over an STM snapshot taken under this Model's
        lock, so it is safe to iterate after the lock is released.
        """
        with self._lock:
            return self._stm.iter_all()

    # Compatibility

    def find_kline(self, signature: KSig) -> KLine | None:
        """Alias for find() — backwards compat."""
        with self._lock:
            return self.find(signature)

    def find_signed_klines(self, signature: KSig) -> list[KLine]:
        """Alias for find_all() — backwards compat."""
        with self._lock:
            return self.find_all(signature)

    def query_graph(self, query: KSig, depth: int = 1):
        """Alias for query() returning list — backwards compat."""
        with self._lock:
            return self.query(query, depth)

    def duplicate(self) -> Model:
        """Create a duplicate of this model's frame."""
        with self._lock:
            frame = self._frame.all_klines()
            klines = [KLine(kl.signature, list(kl.nodes), kl.dbg) for kl in frame]
        m = Model()
        for kl in klines:
            m.add_to_frame(kl)
        return m

    @property
    def klines_prop(self) -> list[KLine]:
        """Backwards compat: return frame klines."""
        with self._lock:
            return self._frame.all_klines()

    def as_kline_list(self, limit: int = 0):
        """Backwards compat: iterate KLines."""
        with self._lock:
            items = self._frame.all_klines()
        if limit > 0:
            items = items[:limit]
        return reversed(items)

    def upgrade(self, kline: KLine, significance: KSig) -> None:
        """Upgrade significance — backwards compat."""
        with self._lock:
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
