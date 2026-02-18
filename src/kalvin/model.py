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
    """A structure with a 64-bit significance s_key and list of 64-bit nodes.

    The high bit of s_key indicates the type:
    - 0: NODE
    - 1: EMBEDDING

    Attributes:
        s_key: 64-bit integer s_key (high bit reserved for type)
        nodes: List of integer nodes
    """
    s_key: int          # 64-bit s_key
    nodes: list[int]    # list of nodes

    @property
    def type(self) -> KLineType:
        """Return the type based on the high bit of s_key."""
        return KLineType.EMBEDDING if (self.s_key & HIGH_BIT_MASK) else KLineType.NODE

    @classmethod
    def create_node(cls, s_key: int, nodes: list[int]) -> "KLine":
        """Create a NODE KLine (ensures high bit is 0)."""
        return cls(s_key=s_key & ~HIGH_BIT_MASK, nodes=nodes)

    @classmethod
    def create_embedding(cls, s_key: int, nodes: list[int]) -> "KLine":
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


def query_significance(kv_list: list[KLine], query: int) -> list[KLine]:
    """Query a list of KLines by ANDing Significance with a query.

    Args:
        kv_list: List of Klines to search
        query: The query nodes to match

    Returns:
        List of KLines where (s_key & query) != 0
    """
    return [kv for kv in kv_list if kv.signifies(query)]
