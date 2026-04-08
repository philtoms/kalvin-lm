"""KLine model for knowledge graph operations."""

from typing import Iterator

from kalvin.abstract import KLine, KModel, KNode, KNone, KNodes, KSig


class Model(KModel):
    """A collection of KLines with query and expansion operations.

    Optimized internal structure:
    - _klines: list[KLine] - flat list for O(1) iteration and indexing
    - _by_key: dict[KSig, KNodes] - maps signature to indices in _klines
    - _dedup: set[tuple[KSig, tuple[KNode, ...]]] - O(1) duplicate detection

    Query iteration follows reverse insertion order (newest first).
    """

    __slots__ = ("_klines", "_by_key", "_dedup")

    def __init__(self, klines: list[KLine] | None = None, frame: KModel | None = None):
        self.frame = frame
        """Initialize the model with optional existing KLines."""
        self._klines: list[KLine] = []  # Flat list for O(1) iteration
        self._by_key: dict[KSig, KNodes] = {}  # signature -> list of indices
        self._dedup: set[tuple[KSig, tuple[KNode, ...]]] = set()  # For O(1) duplicate check
        if klines:
            for kline in klines:
                self._add_kline_internal(kline)

    def exists(self, kline: KLine):
        """Check if a kline already exists in the model."""
        if kline.signature in self._by_key:
            key_nodes = (kline.signature, tuple(kline.nodes))
            if key_nodes in self._dedup:
                return True
        return False

    def _add_kline_internal(self, kline: KLine) -> None:
        """Internal method to add a kline without duplicate checking."""
        idx = len(self._klines)
        self._klines.append(kline)
        if kline.signature not in self._by_key:
            self._by_key[kline.signature] = []
        self._by_key[kline.signature].append(idx)

    def add(self, kline: KLine) -> bool:
        """Add a KLine, enforcing the key invariant.

        Args:
            kline: KLine to add

        Returns:
            True if added, False if rejected (duplicate)
        """
        key_nodes = (kline.signature, tuple(kline.nodes))
        if key_nodes in self._dedup:
            return False  # O(1) duplicate check
        self._dedup.add(key_nodes)
        self._add_kline_internal(kline)

        return True

    def upgrade(self, kline: KLine, significance: KSig) -> None:
        """Upgrade the significance of a kline.

        Args:
            kline: KLine to upgrade
            significance: New significance value to OR with existing signature
        """
        kline.signature |= significance

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

    def query(self, query: KLine) -> Iterator[KLine]:
        """Query KLines by ANDing significance with a query.

        Iteration follows reverse insertion order (newest first).
        O(1) setup, O(N) iteration.

        Args:
            query: The query value to match (AND operation on signature)

        Returns:
            Generator that yields matching KLines.
        """
        klines = self._klines
        n = len(klines)

        def generator() -> Iterator[KLine]:
            for i in range(n - 1, -1, -1):
                kline = klines[i]
                if kline.signifies(query.signature):
                    yield kline
            if self.frame:
                for kline in self.frame.query(query):
                    yield kline

        return generator()

    def expand(
        self,
        kline: KLine,
        depth: int = 1,
    ) -> Iterator[KLine]:
        """Expand a KLine and its descendants up to a given depth.

        Args:
            kline: KLine to expand
            depth: Maximum recursion depth for expanding child nodes

        Returns:
            Generator that yields expanded KLines.
        """
        if depth <= 0:
            return iter([])

        model = self

        def get_node_klines(nodes: KNodes) -> list[KLine]:
            """Get all KLines from a list of node keys."""
            found = []
            if nodes is None:
                return found
            if isinstance(nodes, int):
                kline = model.find_kline(nodes)
                if kline is not KNone:
                    found.append(kline)
                return found
            for node in nodes:
                kline = model.find_kline(node)
                if kline is not KNone:
                    found.append(kline)
            return found

        visited: set[KSig] = set()

        def expand_inner(kl: KLine, current_depth: int) -> Iterator[KLine]:
            """Expand a KLine and yield results immediately."""
            if kl.signature in visited:
                return
            visited.add(kl.signature)

            yield kl

            if current_depth >= depth:
                return

            for child in get_node_klines(kl.nodes):
                yield from expand_inner(child, current_depth + 1)

        return expand_inner(kline, 1)

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
    def klines(self) -> list[KLine]:
        """Return the list of KLines."""
        return self._klines

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
