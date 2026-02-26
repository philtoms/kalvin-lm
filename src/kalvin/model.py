"""KLine model for knowledge graph operations."""

from dataclasses import dataclass
from typing import TypeAlias, Iterator


# Type alias for a KNode (64-bit int)
KNode: TypeAlias = int
KSig: TypeAlias = int | None


@dataclass
class KLine:
    """A structure with a 64-bit significance s_key and list of child KNodes.

    Attributes:
        s_key: 64-bit integer s_key
        nodes: List of child KNode integers
    """

    s_key: int  # 64-bit s_key
    nodes: list[KNode]  # list of child KNodes

    @classmethod
    def create(cls, significance: int, token: int, nodes: list[KNode]) -> "KLine":
        """Create a KLine from significance, token, and nodes.

        The s_key is constructed from significance | token.

        Args:
            significance: Significance value to OR with token
            token: Token value to OR with significance
            nodes: List of child KNode integers

        Returns:
            KLine with s_key = significance | token
        """
        return cls(s_key=significance | token, nodes=nodes)

    def signifies(self, query: int) -> bool:
        """Check if this KLine signifies a query via AND operation.

        Args:
            query: The query node to signify

        Returns:
            True if (s_key & query) != 0
        """
        return (self.s_key & query) != 0


def nodes_equal(nodes1: list[KNode], nodes2: list[KNode]) -> bool:
    """Check if two node lists are equal."""
    if len(nodes1) != len(nodes2):
        return False
    for i in range(len(nodes1)):
        if nodes1[i] != nodes2[i]:
            return False
    return True


class Model:
    """A collection of KLines with query and expansion operations.

    Optimized internal structure:
    - _klines: list[KLine] - flat list for O(1) iteration and indexing
    - _by_key: dict[int, list[int]] - maps s_key to indices in _klines
    - _dedup: set[tuple[int, tuple[int, ...]]] - O(1) duplicate detection

    Query iteration follows reverse insertion order (newest first).
    """

    __slots__ = ("_klines", "_by_key", "_dedup")

    def __init__(self, klines: list[KLine] | None = None):
        """Initialize the model with optional existing KLines."""
        self._klines: list[KLine] = []  # Flat list for O(1) iteration
        self._by_key: dict[int, list[int]] = {}  # s_key -> list of indices
        self._dedup: set[tuple[int, tuple[int, ...]]] = set()  # For O(1) duplicate check
        if klines:
            for kline in klines:
                self._add_kline_internal(kline)

    def _add_kline_internal(self, kline: KLine) -> None:
        """Internal method to add a kline without duplicate checking."""
        idx = len(self._klines)
        self._klines.append(kline)
        if kline.s_key not in self._by_key:
            self._by_key[kline.s_key] = []
        self._by_key[kline.s_key].append(idx)
        # Store (s_key, tuple(nodes)) for O(1) dedup
        self._dedup.add((kline.s_key, tuple(kline.nodes)))

    def add(self, kline: KLine) -> KSig:
        """Add a KLine, enforcing the key invariant.

        Args:
            kline: KLine to add

        Returns:
            s_key if added, None if rejected (exact duplicate)
        """
        key_nodes = (kline.s_key, tuple(kline.nodes))
        if key_nodes in self._dedup:
            return None  # O(1) duplicate check
        self._add_kline_internal(kline)
        return kline.s_key

    def find_by_key(self, key: int | None) -> KLine | None:
        """Find a KLine by its s_key.

        Returns the most recently added KLine with the given key.
        O(1) lookup.

        Args:
            key: The s_key to search for

        Returns:
            KLine if found, None otherwise
        """
        if key is None or key not in self._by_key:
            return None
        # Return the most recently added (last index in the list)
        return self._klines[self._by_key[key][-1]]

    def query(
        self,
        query: int,
        focus_limit: int = 0,
    ) -> tuple[Iterator[KLine], Iterator[KLine]]:
        """Query KLines by ANDing significance with a query.

        Returns two independent generators for concurrent processing:
        1. Fast stream: yields matching KLines immediately (up to focus_limit)
        2. Slow stream: yields remaining matching KLines

        Iteration follows reverse insertion order (newest first).
        O(1) setup, O(N) iteration.

        Args:
            query: The query value to match (AND operation on s_key)
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
            for node_key in nodes:
                kline = model.find_by_key(node_key)
                if kline is not None:
                    found.append(kline)
            return found

        def expand_kline_generator(
            kline: KLine,
            current_depth: int,
            visited: set[int],
        ) -> Iterator[KLine]:
            """Expand a KLine and yield results immediately."""
            if kline.s_key in visited:
                v_kline = model.find_by_key(kline.s_key)
                if v_kline and nodes_equal(v_kline.nodes, kline.nodes):
                    return
            else:
                visited.add(kline.s_key)

            yield kline

            if current_depth >= depth:
                return

            for child in get_node_klines(kline.nodes):
                yield from expand_kline_generator(child, current_depth + 1, visited)

        def fast_generator() -> Iterator[KLine]:
            visited: set[int] = set()
            count = 0
            for kline in focus_set:
                if focus_limit > 0 and count >= focus_limit:
                    break
                yield from expand_kline_generator(kline, 1, visited)
                count += 1

        def slow_generator() -> Iterator[KLine]:
            if focus_limit <= 0:
                return
            visited: set[int] = set()
            count = 0
            for kline in focus_set:
                if count >= focus_limit:
                    yield from expand_kline_generator(kline, 1, visited)
                count += 1

        return fast_generator(), slow_generator()

    def duplicate(self) -> "Model":
        """Create a duplicate of this model."""
        klines = [KLine(s_key=k.s_key, nodes=k.nodes.copy()) for k in self._klines]
        return Model(klines)

    def __len__(self) -> int:
        """Return the number of KLines in the model. O(1)."""
        return len(self._klines)

    def __iter__(self) -> Iterator[KLine]:
        """Iterate over all KLines in insertion order. O(1) setup."""
        return iter(self._klines)

    def __getitem__(self, index: int) -> KLine:
        """Get a KLine by index. O(1)."""
        return self._klines[index]
