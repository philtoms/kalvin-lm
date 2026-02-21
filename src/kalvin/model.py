"""KLine model for knowledge graph operations."""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from struct import pack, unpack
from typing import TypeAlias, Iterator, Literal
import json


# High bit mask (bit 63)
HIGH_BIT_MASK = 0x8000_0000_0000_0000


class KLineType(IntEnum):
    """Type based on the high bit of a key."""
    SIGNATURE = 1  # high bit = 1 (branch)
    EMBEDDING = 0  # high bit = 0 (leaf)


# Type alias for a KNode (64-bit int with high bit reserved for type)
KNode: TypeAlias = int


def get_node_type(node: KNode) -> KLineType:
    """Get the type of a KNode based on its high bit."""
    return KLineType.SIGNATURE if (node & HIGH_BIT_MASK) else KLineType.EMBEDDING


def create_signature_key(key: int) -> KNode:
    """Create a NODE key (sets high bit to 1)."""
    assert not (key & HIGH_BIT_MASK), "Key value must not use high bit"
    return key | HIGH_BIT_MASK


def create_embedding_key(key: int) -> KNode:
    """Create an EMBEDDING key (ensures high bit is 0)."""
    assert not (key & HIGH_BIT_MASK), "Key value must not use high bit"
    return key & ~HIGH_BIT_MASK


@dataclass
class KLine:
    """A structure with a 64-bit significance s_key and list of child KNodes.

    The high bit of s_key indicates the type:
    - 1: NODE (branch - can have children)
    - 0: EMBEDDING (leaf - no children)

    Attributes:
        s_key: 64-bit integer s_key (high bit reserved for type)
        nodes: List of child KNode integers (high bit indicates type)
    """
    s_key: int           # 64-bit s_key
    nodes: list[KNode]   # list of child KNodes (ints with type bit)

    @property
    def type(self) -> KLineType:
        """Return the type based on the high bit of s_key."""
        return KLineType.SIGNATURE if (self.s_key & HIGH_BIT_MASK) else KLineType.EMBEDDING

    @classmethod
    def create_signature(cls, s_key: int, nodes: list[KNode]) -> "KLine":
        """Create a NODE KLine (sets high bit to 1)."""
        return cls(s_key=s_key | HIGH_BIT_MASK, nodes=nodes)

    @classmethod
    def create_embedding(cls, s_key: int, nodes: list[KNode]) -> "KLine":
        """Create an EMBEDDING KLine (ensures high bit is 0)."""
        return cls(s_key=s_key & ~HIGH_BIT_MASK, nodes=nodes)

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

    Maintains an ordered list of KLines and provides methods for:
    - Adding KLines with duplicate detection
    - Querying by significance (AND operation on s_key)
    - Expanding KLines to traverse their children

    The model supports two-stream processing for concurrent consumption:
    - Fast stream: for immediate processing by a fast thread
    - Slow stream: for deferred processing by a slow thread
    """

    def __init__(self, klines: list[KLine] | None = None):
        """Initialize the model with optional existing KLines.

        Args:
            klines: Optional list of KLines to initialize with
        """
        self._klines: list[KLine] = klines.copy() if klines else []
        self._signatures = set(kline.s_key for kline in self._klines)

    def add_signature(self, kline: KLine) -> bool:
        """Add a KLine, enforcing the duplicate key invariant.

        Invariant: Duplicate keys are allowed, but nodes must differ.
        - If s_key is new: add the kline
        - If s_key exists with different nodes: add the kline (allowed)
        - If s_key exists with same nodes: reject (would be exact duplicate)

        Args:
            kline: KLine to add

        Returns:
            index if added, None if rejected (exact duplicate)
        """
        kline.s_key |= HIGH_BIT_MASK
        if kline.s_key in self._signatures:
            existing = self.find_by_key(kline.s_key)
            if existing and nodes_equal(existing.nodes, kline.nodes):
                return False  # Exact duplicate, reject

        self._klines.insert(0,kline)
        self._signatures.add(kline.s_key)
        return True
    

    def add_embedding(self, kline: KLine) -> bool:
        """Add a KLine, enforcing the duplicate key invariant.

        Invariant: Duplicate keys are allowed, but nodes must differ.
        - If s_key is new: add the kline
        - If s_key exists with different nodes: add the kline (allowed)
        - If s_key exists with same nodes: reject (would be exact duplicate)

        Args:
            kline: KLine to add

        Returns:
            index if added, None if rejected (exact duplicate)
        """
        kline.s_key &= ~HIGH_BIT_MASK
        if kline.s_key in self._signatures:
            existing = self.find_by_key(kline.s_key)
            if existing and nodes_equal(existing.nodes, kline.nodes):
                return False  # Exact duplicate, reject

        self._klines.insert(0,kline)
        self._signatures.add(kline.s_key)
        return True
    

    def find_by_key(self, key: int) -> KLine | None:
        """Find a KLine by its s_key.

        Args:
            key: The s_key to search for

        Returns:
            KLine if found, None otherwise
        """
        for kline in self._klines:
            if kline.s_key == key:
                return kline
        return None

    def query(
        self,
        query: int,
        focus_limit: int = 0,
    ) -> tuple[Iterator[KLine], Iterator[KLine]]:
        """Query KLines by ANDing significance with a query.

        Returns two independent generators for concurrent processing:
        1. Fast stream: yields matching KLines immediately (up to focus_limit)
        2. Slow stream: yields remaining matching KLines

        Args:
            query: The query value to match (AND operation on s_key)
            focus_limit: Number of top-level matches in fast (0 = all in fast)
                - focus_limit=2: first 2 matches in fast, rest in slow

        Returns:
            Tuple of (fast_generator, slow_generator) that yield KLines.
        """
        klines = self._klines

        def fast_generator() -> Iterator[KLine]:
            count = 0
            for kline in klines:
                if kline.signifies(query):
                    if focus_limit > 0 and count >= focus_limit:
                        break
                    yield kline
                    count += 1

        def slow_generator() -> Iterator[KLine]:
            if focus_limit <= 0:
                return  # No slow when focus_limit is 0
            count = 0
            for kline in klines:
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

        Only NODE type KLines (high bit = 1) are traversed for children.
        EMBEDDING nodes (high bit = 0) are leaves and not expanded.

        Args:
            focus_set: List of KLines to expand (e.g., from query)
            depth: Maximum recursion depth for expanding child nodes:
                - depth=0: yield nothing
                - depth=1: yield klines only (no child expansion)
                - depth=2: yield klines + their direct children
                - depth=N: expand N levels of children
            focus_limit: Number of klines in fast (0 = all in fast)
                - focus_limit=2: first 2 klines in fast, rest in slow

        Returns:
            Tuple of (fast_generator, slow_generator) that yield expanded KLines.
        """
        if depth <= 0:
            return iter([]), iter([])

        model = self

        def get_node_klines(nodes: list[KNode]) -> list[KLine]:
            """Get all NODE type KLines from a list of node keys."""
            found = []
            for node_key in nodes:
                if get_node_type(node_key) == KLineType.SIGNATURE:
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

            # Check for cycle - if kline already visited, stop this branch
            if kline.s_key in visited:
                v_kline = model.find_by_key(kline.s_key)
                if v_kline and nodes_equal(v_kline.nodes, kline.nodes):
                    return
            else:
                visited.add(kline.s_key)

            yield kline

            # Stop if we've reached max depth
            if current_depth >= depth:
                return

            # Expand child NODE klines
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
                return  # No slow when focus_limit is 0
            visited: set[int] = set()
            count = 0
            for kline in focus_set:
                if count >= focus_limit:
                    yield from expand_kline_generator(kline, 1, visited)
                count += 1

        return fast_generator(), slow_generator()

    def __len__(self) -> int:
        """Return the number of KLines in the model."""
        return len(self._klines)

    def __iter__(self) -> Iterator[KLine]:
        """Iterate over all KLines in the model."""
        return iter(self._klines)

    def __getitem__(self, index: int) -> KLine:
        """Get a KLine by index."""
        return self._klines[index]
