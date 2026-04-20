"""Short-Term Memory (STM) — dual-keyed dictionary for KLine lookup.

The STM indexes KLines by two distinct keys so the agent can retrieve
them efficiently by either their *signature* or their *nodes signature*:

    * **signature**      — ``kline.signature``  (the KLine's own signature)
    * **nodes_signature** — ``make_signature(kline.nodes)``  (derived from nodes)

Both keys map into the same backing store ``dict[KSig, list[KLine]]``.  If the
two keys are identical the KLine is stored only once under that key.
"""

from __future__ import annotations

from kalvin.abstract import KLine, KNodes, KSig, KTokenizer
from kalvin.mod_tokenizer import Mod32Tokenizer


class STM:
    """Short-Term Memory: dual-keyed KLine dictionary.

    Structure
    ---------
    ``_store: dict[KSig, list[KLine]]``

    A single entry holds *multiple* KLines so that different KLines sharing
    the same key are not silently deduplicated.

    ``_order: list[KLine]`` tracks insertion order so that ``query()``
    can return results deterministically.

    Parameters
    ----------
    tokenizer:
        The KTokenizer used to derive *nodes signatures* from node lists
        via ``tokenizer.make_signature(nodes)``.
    """

    __slots__ = ("_store", "_order", "_dedup", "_tokenizer")

    def __init__(self, tokenizer: KTokenizer):
        self._store: dict[KSig, list[KLine]] = {}
        self._order: list[KLine] = []
        self._dedup: set[tuple[KSig, tuple[int, ...]]] = set()
        self._tokenizer = tokenizer if tokenizer else Mod32Tokenizer()

    # ── Core API ────────────────────────────────────────────────────────

    def add(self, kline: KLine, dedup: bool = True) -> bool:
        """Add a KLine to the STM, indexed by both signature and nodes signature.

        1. Compute *nodes_sig* from ``kline.nodes`` via ``make_signature``.
        2. Index by ``kline.signature`` (always).
        3. Index by *nodes_sig* when it differs from ``kline.signature``.

        Args:
            kline: KLine to add.
            dedup: If True, reject klines with identical (signature, nodes).

        Returns:
            True if the kline was added, False if rejected as a duplicate.
        """
        nodes_sig = self._tokenizer.make_signature(kline.nodes)
        sig = kline.signature or nodes_sig

        if dedup:
            key = (sig, tuple(kline.as_node_list()))
            if key in self._dedup:
                return False
            self._dedup.add(key)

        self._order.append(kline)
        self._append(sig, kline)
        if nodes_sig != sig:
            self._append(nodes_sig, kline)
        return True

    def get(self, key: KSig) -> list[KLine]:
        """Return all KLines indexed under *key*, or an empty list."""
        return list(self._store.get(key, []))

    def get_kline(self, key: KSig) -> KLine | None:
        """Return the most recently added KLine under *key*, or ``None``."""
        bucket = self._store.get(key)
        if bucket:
            return bucket[-1]
        return None

    def find_by_signature(self, signature: KSig) -> list[KLine]:
        """Look up KLines by their KLine signature."""
        return self.get(signature)

    def find_by_nodes(self, nodes: KNodes) -> list[KLine]:
        """Look up KLines by a nodes signature derived from *nodes*.

        This mirrors the key that ``add()`` computes via
        ``tokenizer.make_signature(nodes)``.
        """
        nodes_sig = self._tokenizer.make_signature(
            nodes if isinstance(nodes, list) else [nodes] if nodes is not None else []
        )
        return self.get(nodes_sig)

    def query(self, sig: KSig) -> list[KLine]:
        """Return all KLines whose signatures overlap *sig* (AND ≠ 0).

        Scans every KLine in lifo order, returning those whose
        ``kline.signature & sig != 0``.  Results are deduplicated by
        identity (the same KLine indexed under two keys appears once).

        Returns:
            List of matching KLines in lifo order.
        """
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
        """Remove a specific KLine from all index entries.

        Only the *first* matching entry per key is removed (identity check).
        """
        sig = kline.signature
        nodes_sig = self._tokenizer.make_signature(kline.nodes)
        self._remove_from(sig, kline)
        if nodes_sig != sig:
            self._remove_from(nodes_sig, kline)
        # Remove from insertion-order list and dedup set
        try:
            self._order.remove(kline)
        except ValueError:
            pass
        self._dedup.discard((sig, tuple(kline.as_node_list())))

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()
        self._order.clear()
        self._dedup.clear()

    def __len__(self) -> int:
        """Return the number of unique keys in the STM."""
        return len(self._store)

    def __contains__(self, key: KSig) -> bool:
        """Check if *key* has any indexed KLines."""
        return key in self._store

    def __repr__(self) -> str:
        return f"STM(keys={len(self._store)}, klines={len(self._order)}, tokenizer={type(self._tokenizer).__name__})"

    # ── Internals ───────────────────────────────────────────────────────

    def _append(self, key: KSig, kline: KLine) -> None:
        """Append *kline* to the bucket at *key*, creating the bucket if needed."""
        bucket = self._store.get(key)
        if bucket is None:
            self._store[key] = [kline]
        else:
            bucket.append(kline)

    def _remove_from(self, key: KSig, kline: KLine) -> None:
        """Remove the first occurrence of *kline* (identity) from *key*'s bucket."""
        bucket = self._store.get(key)
        if not bucket:
            return
        for i, existing in enumerate(bucket):
            if existing is kline:
                bucket.pop(i)
                break
        if not bucket:
            del self._store[key]
