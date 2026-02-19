from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias


# High bit mask (bit 63)
HIGH_BIT_MASK = 0x8000_0000_0000_0000


class KLineType(IntEnum):
    """Type based on the high bit of a key."""
    NODE = 1       # high bit = 1 (branch)
    EMBEDDING = 0  # high bit = 0 (leaf)


# Type alias for a KNode (64-bit int with high bit reserved for type)
KNode: TypeAlias = int


def get_node_type(node: KNode) -> KLineType:
    """Get the type of a KNode based on its high bit."""
    return KLineType.NODE if (node & HIGH_BIT_MASK) else KLineType.EMBEDDING


def create_node_key(key: int) -> KNode:
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
        return KLineType.NODE if (self.s_key & HIGH_BIT_MASK) else KLineType.EMBEDDING

    @classmethod
    def create_node(cls, s_key: int, nodes: list[KNode]) -> "KLine":
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


def add_kline(kv_list: list[KLine], kline: KLine) -> bool:
    """Add a KLine to a list, enforcing the duplicate key invariant.

    Invariant: Duplicate keys are allowed, but nodes must differ.
    - If s_key is new: add the kline
    - If s_key exists with different nodes: add the kline (allowed)
    - If s_key exists with same nodes: reject (would be exact duplicate)

    Args:
        kv_list: List to add to (modified in place)
        kline: KLine to add

    Returns:
        True if added, False if rejected (exact duplicate)
    """
    for existing in kv_list:
        if existing.s_key == kline.s_key:
            if nodes_equal(existing.nodes, kline.nodes):
                return False  # Exact duplicate, reject
    kv_list.append(kline)
    return True


def query_significance(
    kv_list: list[KLine],
    query: int,
    depth: int = 1,
    cap: int = 0,
) -> list[KLine]:
    """Query a list of KLines by ANDing Significance with a query.

    Algorithm:
    - Find all KLines that match the query
    - For each match, expand its children up to depth N
    - Cap applies per level: max N entries at each depth level
    - Cycle detection: don't revisit s_keys already in result

    Only NODE type KLines (high bit = 1) are traversed.
    EMBEDDING nodes (high bit = 0) are leaves and not in kv_list.

    Args:
        kv_list: List of KLines to search (may contain duplicates)
        query: The query value to match
        depth: Maximum recursion depth for expanding child nodes:
            - depth=0: return empty
            - depth=1: return matching klines only (no child expansion)
            - depth=2: return matching klines + their direct children
            - depth=N: expand N levels of children
        cap: Maximum number of children per level (0 = no limit)
            - cap=2: return max 2 top-level matches, max 2 children per match, etc.

    Returns:
        List containing all matching KLines and their descendants.
        Empty list if no match is found.
    """
    if depth <= 0:
        return []

    result: list[KLine] = []
    visited: set[int] = set()  # Track visited s_keys for cycle detection

    def find_kline_by_key(kv_list: list[KLine], key: int) -> KLine | None:
        """Find a KLine in kv_list by its s_key."""
        for kline in kv_list:
            if kline.s_key == key:
                return kline
        return None

    def get_node_klines(nodes: list[KNode]) -> list[KLine]:
        """Get all NODE type KLines from a list of node keys."""
        found = []
        for node_key in nodes:
            if get_node_type(node_key) == KLineType.NODE:
                kline = find_kline_by_key(kv_list, node_key)
                if kline is not None:
                    found.append(kline)
        return found

    def expand_kline(kline: KLine, current_depth: int) -> None:
        """Expand a KLine and its children into result."""
        # Check for cycle - if s_key already visited, check that it is
        # the same kline and stop this branch
        if kline.s_key in visited:
            v_kline = find_kline_by_key(result, kline.s_key)
            if v_kline and nodes_equal(v_kline.nodes, kline.nodes):
                return
        else:
          visited.add(kline.s_key)

        result.append(kline)

        # Stop if we've reached max depth
        if current_depth >= depth:
            return

        # Expand child NODE klines, respecting cap per level
        children = get_node_klines(kline.nodes)
        if cap > 0:
            children = children[:cap]

        for child in children:
            expand_kline(child, current_depth + 1)

    # Find all matching klines and expand them, respecting cap at top level
    match_count = 0
    for kline in kv_list:
        if kline.signifies(query):
            expand_kline(kline, 1)
            match_count += 1
            if cap > 0 and match_count >= cap:
                break

    return result
