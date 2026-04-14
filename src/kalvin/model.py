"""Model - Collection of KLines for knowledge graph operations."""

from typing import Iterator

from kalvin.abstract import KLine, KModel, KNode, KNodes, KSig, KGraph


class Model(KModel):
    """A collection of KLines with query and expansion operations.

    Optimized internal structure:
    - _klines: list[KLine] - flat list for O(1) iteration and indexing
    - _by_key: dict[KSig, KNodes] - maps signature to indices in _klines
    - _dedup: set[tuple[KSig, tuple[KNode, ...]]] - O(1) duplicate detection

    Query iteration follows reverse insertion order (newest first).
    """

    __slots__ = ("_klines", "_by_key", "_dedup")

    def __init__(self, klines: list[KLine] | None = None, base: "KModel | None" = None):
        self.base = base
        """Initialize the model with optional existing KLines."""
        self._klines: list[KLine] = []  # Flat list for O(1) iteration
        self._by_key: dict[KSig, list[int]] = {}  # signature -> list of indices
        self._dedup: set[tuple[KSig, tuple[KNode, ...]]] = set()  # For O(1) duplicate check
        if klines:
            for kline in klines:
                self.add(kline)

    def exists(self, kline: KLine):
        """Check if a kline already exists in the frame."""
        if kline.signature in self._by_key:
            key_nodes = (kline.signature, tuple(kline.as_node_list()))
            if key_nodes in self._dedup:
                return True

        if self.base:
            return self.base.exists(kline)
        
        return False

    def add(self, kline: KLine, dedup: bool = False) -> bool:
        """Add a KLine, enforcing the key invariant.

        Args:
            kline: KLine to add

        Returns:
            True if added, False if rejected (duplicate)
        """
        key_nodes = (kline.signature, tuple(kline.as_node_list()))
        if not dedup or key_nodes not in self._dedup:
            if kline.signature not in self._by_key:
                self._by_key[kline.signature] = []
            elif not kline.nodes: # already signed
                return False

            # A frame remembers all signatures
            if self.base:
                node_sig = 0 # TODO refactor into signature module
                for node in kline.as_node_list():
                    node_sig |= node
                    if not self.find_kline(node):
                        return self.add(KLine(signature=node, nodes=None))
                if not self.find_kline(node_sig):
                    return self.add(KLine(signature=node_sig, nodes=kline.nodes))

            idx = len(self._klines)
            self._klines.append(kline)
            self._by_key[kline.signature].append(idx)
            self._dedup.add(key_nodes)
            return True
        return False

    def upgrade(self, kline: KLine, significance: KSig) -> None:
        """Upgrade the significance of a kline.

        Args:
            kline: KLine to upgrade
            significance: New significance value to OR with existing signature
        """
        kline.signature |= significance

    def find_kline(self, signature: KSig, significance: KSig | None = None) -> KLine | None:
        """Find a KLine by its signature.

        Returns the most recently added KLine with the given signature.
        O(1) lookup.

        Args:
            signature: The signature to search for
            significance: Optional significance filter

        Returns:
            KLine if found, None otherwise
        """
        if signature not in self._by_key:
            if self.base:
                return self.base.find_kline(signature, significance)
            return None

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
            if self.base:
                return self.base.find_signed_klines(signature)
            return []

        return [self._klines[idx] for idx in self._by_key[signature]]

    def query_graph(self, query: KSig, depth: int = 1) -> KGraph:
        """Query KLines by ANDing significance with a query.

        Iteration follows reverse insertion order (newest first).
        O(1) setup, O(N) iteration.

        Args:
            query: The signature value to match (AND operation on signature)
            depth: Maximum recursion depth for expanding child nodes


        Returns:
            Generator that yields matching KLines.
        """
        klines = self._klines
        n = len(klines)

        def generator() -> KGraph:
            for i in range(n - 1, -1, -1):
                kline = klines[i]
                if kline.signifies(query):
                    yield kline
                    for child in self.expand(kline, depth):
                        yield child

            if self.base:
                for kline in self.base.query_graph(query):
                    yield kline

        return generator()

    def expand(
        self,
        kline: KLine,
        depth: int = 2,
    ) -> KGraph:
        """Expand a KLine and its descendants up to a given depth.

        Args:
            kline: KLine to expand
            depth: Maximum recursion depth for expanding child nodes

        Returns:
            Generator that yields expanded KLines.
        """
        if depth <= 0:
            return iter([])

        visited: set[KSig] = set()

        def expand_inner(kl: KLine, current_depth: int) -> KGraph:
            """Expand a KLine and yield results immediately."""
            if kl.signature in visited:
                return
            visited.add(kl.signature)

            if current_depth >= depth:
                return

            for node in kl.as_node_list():
                child = self.find_kline(node)
                if child:
                    yield child
                    yield from expand_inner(child, current_depth + 1)
 
            if self.base:
                for kline in self.base.expand(kl):
                    yield kline

        return expand_inner(kline, 1)

    def duplicate(self) -> "Model":
        """Create a duplicate of this model."""
        klines = [KLine(signature=k.signature, nodes=k.nodes.copy() if isinstance(k.nodes, list) else k.nodes, dbg_text=k.dbg_text) for k in self._klines]
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

        if not kline:
            return descendants

        for child in kline.as_node_list():
            descendants.add(child)
            # Recursively get child's descendants
            child_descendants = self.get_all_descendants(child, visited.copy())
            descendants.update(child_descendants)

        return descendants


    def __len__(self) -> int:
        """Return the number of KLines in the frame. O(1)."""
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

        Usage: frame.kline[signature] == frame.find_kline(signature)
        """
        return _KLineAccessor(self)


class _KLineAccessor:
    """Helper class for model.kline[signature] access."""

    __slots__ = ("_model",)

    def __init__(self, model: Model):
        self._model = model

    def __getitem__(self, signature: KSig) -> KLine:
        return self._model.find_kline(signature)
