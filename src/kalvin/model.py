"""KLine model for knowledge graph operations."""

from typing import Iterator

from kalvin.abstract import KLine, KModel, KNode, KNone, KSig


class Model(KModel):
    """A collection of KLines with query and expansion operations.

    Optimized internal structure:
    - _klines: list[KLine] - flat list for O(1) iteration and indexing
    - _by_key: dict[KSig, list[KNode]] - maps signature to indices in _klines
    - _dedup: set[tuple[KSig, tuple[KNode, ...]]] - O(1) duplicate detection

    Query iteration follows reverse insertion order (newest first).
    """

    __slots__ = ("_klines", "_by_key", "_dedup")

    def __init__(self, klines: list[KLine] | None = None, frame: KModel | None = None):
        self.frame = frame
        """Initialize the model with optional existing KLines."""
        self._klines: list[KLine] = []  # Flat list for O(1) iteration
        self._by_key: dict[KSig, list[KNode]] = {}  # signature -> list of indices
        self._dedup: set[tuple[KSig, tuple[KNode, ...]]] = set()  # For O(1) duplicate check
        if klines:
            for kline in klines:
                self._add_kline_internal(kline)

    def _add_kline_internal(self, kline: KLine) -> None:
        """Internal method to add a kline without duplicate checking."""
        idx = len(self._klines)
        self._klines.append(kline)
        if kline.signature not in self._by_key:
            self._by_key[kline.signature] = []
        self._by_key[kline.signature].append(idx)

    def add(self, kline: KLine, train: bool = False) -> bool:
        """Add a KLine, enforcing the key invariant.

        Args:
            kline: KLine to add
            train: If True, enforce training mode (dedup by signature only)

        Returns:
            True if added, False if rejected (duplicate)
        """
        if train:
            if kline.signature in self._by_key:
                return False

        key_nodes = (kline.signature, tuple(kline.nodes))
        if key_nodes in self._dedup:
            return False  # O(1) duplicate check
        self._add_kline_internal(kline)
        if train:
            self._dedup.add(key_nodes)

        return True

    def find_kline(self, signature: KSig, significance: KSig | None = None) -> KLine:
        """Find a KLine by its signature.

        Returns the most recently added KLine with the given signature.
        O(1) lookup.

        Args:
            signature: The signature to search for
            significance: Optional significance filter

        Returns:
            KLine if found, KNone otherwise
        """
        if signature not in self._by_key:
            if self.frame:
                return self.frame.find_kline(signature, significance)
            return KNone

        if significance is not None:
            for idx in self._by_key[signature]:
                kline = self._klines[idx]
                if kline.signifies(significance):
                    return kline

        # Return the most recently added (last index in the list)
        return self._klines[self._by_key[signature][-1]]

    def find_signed_klines(self, signature: KSig) -> list[KLine]:
        """Find all KLines matching the given signature.

        Returns all KLines with the given signature.
        O(1) lookup.

        Args:
            signature: The signature to search for

        Returns:
            KLine list
        """
        if signature not in self._by_key:
            if self.frame:
                return self.frame.find_signed_klines(signature)
            return []

        return [self._klines[idx] for idx in self._by_key[signature]]

    def query(
        self,
        query: KSig,
        focus_limit: int = 0,
    ) -> tuple[Iterator[KLine], Iterator[KLine]]:
        """Query KLines by ANDing significance with a query.

        Returns two independent generators for concurrent processing:
        1. Fast stream: yields matching KLines immediately (up to focus_limit)
        2. Slow stream: yields remaining matching KLines

        Iteration follows reverse insertion order (newest first).
        O(1) setup, O(N) iteration.

        Args:
            query: The query value to match (AND operation on signature)
            focus_limit: Number of top-level matches in fast (0 = all in fast)

        Returns:
            Tuple of (fast_generator, slow_generator) that yield KLines.
        """
        klines = self._klines
        n = len(klines)

        def fast_generator() -> Iterator[KLine]:
            count = 0
            for i in range(n - 1, -1, -1):
                kline = klines[i]
                if kline.signifies(query):
                    if focus_limit > 0 and count >= focus_limit:
                        break
                    yield kline
                    count += 1
            if self.frame and count < focus_limit:
                fast, _ = self.frame.query(query, focus_limit - count)
                for kline in fast:
                    yield kline

        def slow_generator() -> Iterator[KLine]:
            if focus_limit <= 0:
                return
            count = 0
            for i in range(n - 1, -1, -1):
                kline = klines[i]
                if kline.signifies(query):
                    if count >= focus_limit:
                        yield kline
                    count += 1
            if self.frame and count < focus_limit:
                _, slow = self.frame.query(query, focus_limit - count)
                for kline in slow:
                    yield kline

        return fast_generator(), slow_generator()

    def expand(
        self,
        focus_set: list[KLine],
        depth: int = 1,
        focus_limit: int = 0,
    ) -> tuple[Iterator[KLine], Iterator[KLine]]:
        """Expand KLines and their descendants up to a given depth.

        Returns two independent generators for concurrent processing:
        1. Fast stream: yields first `focus_limit` KLines and their descendants
        2. Slow stream: yields remaining KLines and their descendants

        Args:
            focus_set: List of KLines to expand (e.g., from query)
            depth: Maximum recursion depth for expanding child nodes
            focus_limit: Number of klines in fast (0 = all in fast)

        Returns:
            Tuple of (fast_generator, slow_generator) that yield expanded KLines.
        """
        if depth <= 0:
            return iter([]), iter([])

        model = self

        def get_node_klines(nodes: list[KNode]) -> list[KLine]:
            """Get all KLines from a list of node keys."""
            found = []
            for node in nodes:
                kline = model.find_kline(node)
                if kline is not KNone:
                    found.append(kline)
            return found

        def expand_kline_generator(
            kline: KLine,
            current_depth: int,
            visited: set[int],
        ) -> Iterator[KLine]:
            """Expand a KLine and yield results immediately."""
            if kline.signature in visited:
                return
            else:
                visited.add(kline.signature)

            yield kline

            if current_depth >= depth:
                return

            for child in get_node_klines(kline.nodes):
                yield from expand_kline_generator(child, current_depth + 1, visited)

        def fast_generator() -> Iterator[KLine]:
            visited: set[KSig] = set()
            count = 0
            for kline in focus_set:
                if focus_limit > 0 and count >= focus_limit:
                    break
                yield from expand_kline_generator(kline, 1, visited)
                count += 1
            if self.frame and count < focus_limit:
                fast, _ = self.frame.expand(focus_set, depth, focus_limit - count)
                for kline in fast:
                    yield kline

        def slow_generator() -> Iterator[KLine]:
            if focus_limit <= 0:
                return
            visited: set[KSig] = set()
            count = 0
            for kline in focus_set:
                if count >= focus_limit:
                    yield from expand_kline_generator(kline, 1, visited)
                count += 1
            if self.frame and count < focus_limit:
                _, slow = self.frame.expand(focus_set, depth, focus_limit - count)
                for kline in slow:
                    yield kline

        return fast_generator(), slow_generator()

    def duplicate(self) -> "Model":
        """Create a duplicate of this model."""
        klines = [KLine(signature=k.signature, nodes=k.nodes.copy(), dbg_text=k.dbg_text) for k in self._klines]
        return Model(klines)

    def get_all_descendants(self, node: KNode, visited: set[KSig] | None = None) -> set[KSig]:
        """Recursively collect all descendant nodes.

        Args:
            node: The node to start from
            visited: Set of already visited nodes (cycle detection)

        Returns:
            Set of all descendant node keys
        """
        if visited is None:
            visited = set()

        if node in visited:
            return set()
        visited.add(node)

        descendants: set[KSig] = set()
        kline = self.find_kline(node)

        if kline is KNone:
            return descendants

        for child in kline.nodes:
            descendants.add(child)
            # Recursively get child's descendants
            child_descendants = self.get_all_descendants(child, visited.copy())
            descendants.update(child_descendants)

        return descendants


    def __len__(self) -> int:
        """Return the number of KLines in the model. O(1)."""
        return len(self._klines)

    def __iter__(self) -> Iterator[KLine]:
        """Iterate over all KLines in insertion order. O(1) setup."""
        return iter(self._klines)

    def __getitem__(self, signature: KSig) -> KLine:
        """Get a KLine by index. O(1)."""
        return self.find_kline(signature)

    @property
    def kline(self) -> "_KLineAccessor":
        """Return an accessor for finding KLines by signature via bracket notation.

        Usage: model.kline[signature] == model.find_kline(signature)
        """
        return _KLineAccessor(self)


class _KLineAccessor:
    """Helper class for model.kline[signature] access."""

    __slots__ = ("_model",)

    def __init__(self, model: Model):
        self._model = model

    def __getitem__(self, signature: KSig) -> KLine:
        return self._model.find_kline(signature)
