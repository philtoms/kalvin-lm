from dataclasses import dataclass
from enum import IntEnum


# High bit mask (bit 63)
HIGH_BIT_MASK = 0x8000_0000_0000_0000


class KLineType(IntEnum):
    """Type of KLine based on the high bit of s_key."""
    NODE = 0       # high bit = 0
    EMBEDDING = 1  # high bit = 1


@dataclass
class KLine:
    """A structure with a 64-bit significance s_key and list of child KLines.

    The high bit of s_key indicates the type:
    - 0: NODE
    - 1: EMBEDDING

    Attributes:
        s_key: 64-bit integer s_key (high bit reserved for type)
        nodes: List of child KLine objects
    """
    s_key: int              # 64-bit s_key
    nodes: list["KLine"]    # list of child KLines

    @property
    def type(self) -> KLineType:
        """Return the type based on the high bit of s_key."""
        return KLineType.EMBEDDING if (self.s_key & HIGH_BIT_MASK) else KLineType.NODE

    @classmethod
    def create_node(cls, s_key: int, nodes: list["KLine"]) -> "KLine":
        """Create a NODE KLine (ensures high bit is 0)."""
        return cls(s_key=s_key & ~HIGH_BIT_MASK, nodes=nodes)

    @classmethod
    def create_embedding(cls, s_key: int, nodes: list["KLine"]) -> "KLine":
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

    Searches recursively up to the specified depth, detecting and halting
    circular dependencies.

    Args:
        kv_list: List of KLines to search
        query: The query value to match
        depth: Maximum recursion depth:
            - depth=0: search nothing
            - depth=1: search only top-level items (default)
            - depth=2: search top-level + children
            - depth=N: search N levels deep

    Returns:
        List of KLines where (s_key & query) != 0, including matches found
        in child nodes up to the specified depth
    """
    results: list[KLine] = []
    visited: set[int] = set()

    def search(kline: KLine, current_depth: int) -> None:
        """Recursively search KLine and its children."""
        # Stop if we've exceeded max depth
        if current_depth >= depth:
            return

        # Use id() to detect circular references (same object)
        kline_id = id(kline)
        if kline_id in visited:
            return
        visited.add(kline_id)

        # Check if this KLine matches
        if kline.signifies(query):
            results.append(kline)

        # Search children
        for child in kline.nodes:
            search(child, current_depth + 1)

    for kline in kv_list:
        search(kline, 0)

    return results
