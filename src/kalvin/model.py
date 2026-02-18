from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias


# High bit mask (bit 63)
HIGH_BIT_MASK = 0x8000_0000_0000_0000


class KLineType(IntEnum):
    """Type based on the high bit of a key."""
    NODE = 0       # high bit = 0 (branch)
    EMBEDDING = 1  # high bit = 1 (leaf)


# Type alias for a KNode (64-bit int with high bit reserved for type)
KNode: TypeAlias = int


def get_node_type(node: KNode) -> KLineType:
    """Get the type of a KNode based on its high bit."""
    return KLineType.EMBEDDING if (node & HIGH_BIT_MASK) else KLineType.NODE


def create_node_key(key: int) -> KNode:
    """Create a NODE key (ensures high bit is 0)."""
    return key & ~HIGH_BIT_MASK


def create_embedding_key(key: int) -> KNode:
    """Create an EMBEDDING key (sets high bit to 1)."""
    return key | HIGH_BIT_MASK


@dataclass
class KLine:
    """A structure with a 64-bit significance s_key and list of child KNodes.

    The high bit of s_key indicates the type:
    - 0: NODE (branch - can have children)
    - 1: EMBEDDING (leaf - no children)

    Attributes:
        s_key: 64-bit integer s_key (high bit reserved for type)
        nodes: List of child KNode integers (high bit indicates type)
    """
    s_key: int           # 64-bit s_key
    nodes: list[KNode]   # list of child KNodes (ints with type bit)

    @property
    def type(self) -> KLineType:
        """Return the type based on the high bit of s_key."""
        return KLineType.EMBEDDING if (self.s_key & HIGH_BIT_MASK) else KLineType.NODE

    @classmethod
    def create_node(cls, s_key: int, nodes: list[KNode]) -> "KLine":
        """Create a NODE KLine (ensures high bit is 0)."""
        return cls(s_key=s_key & ~HIGH_BIT_MASK, nodes=nodes)

    @classmethod
    def create_embedding(cls, s_key: int, nodes: list[KNode]) -> "KLine":
        """Create an EMBEDDING KLine (sets high bit to 1)."""
        return cls(s_key=s_key | HIGH_BIT_MASK, nodes=nodes)

    def signifies(self, query: int) -> bool:
        """Check if this KLine signifies a query via AND operation.

        Args:
            query: The query node to signify

        Returns:
            True if (s_key & query) != 0
        """
        return (self.s_key & query) != 0


def query_significance(
    kv_list: list[KLine],
    query: int,
    depth: int = 1,
) -> list[KLine]:
    """Query a list of KLines by ANDing Significance with a query.

    Algorithm:
    - If query not matched → return empty list
    - If query matches a kline with only embeddings → return just that kline
    - If query matches a kline with kline nodes → return kline + all klines
      its nodes represent (recursive to depth n)
    - If any nodes already in result → stop and return list so far

    Only NODE type KLines (high bit = 0) are traversed.
    EMBEDDING nodes (high bit = 1) are leaves and not in kv_list.

    Args:
        kv_list: List of KLines to search (must contain all NODE type KLines)
        query: The query value to match
        depth: Maximum recursion depth for expanding child nodes:
            - depth=0: return empty
            - depth=1: return matching kline only (no child expansion)
            - depth=2: return matching kline + its direct children
            - depth=N: expand N levels of children

    Returns:
        List containing the matching KLine and its descendants.
        Empty list if no match is found.
    """
    if depth <= 0:
        return []

    def find_kline_by_key(key: int) -> KLine | None:
        """Find a KLine in kv_list by its s_key."""
        for kline in kv_list:
            if kline.s_key == key:
                return kline
        return None

    def get_node_klines(nodes: list[KNode]) -> list[KLine]:
        """Get all NODE type KLines from a list of node keys."""
        result = []
        for node_key in nodes:
            if get_node_type(node_key) == KLineType.NODE:
                kline = find_kline_by_key(node_key)
                if kline is not None:
                    result.append(kline)
        return result

    def expand_kline(kline: KLine, result: list[KLine], current_depth: int) -> bool:
        """Expand a KLine and its children into result.

        Returns True if expansion was stopped due to cycle detection.
        """
        # Check for cycle - if kline already in result, stop
        if kline in result:
            return True

        result.append(kline)

        # Stop if we've reached max depth
        if current_depth >= depth:
            return False

        # Expand child NODE klines
        for child in get_node_klines(kline.nodes):
            if expand_kline(child, result, current_depth + 1):
                return True

        return False

    # Find first matching kline
    for kline in kv_list:
        if kline.signifies(query):
            result: list[KLine] = []
            expand_kline(kline, result, 1)
            return result

    return []
