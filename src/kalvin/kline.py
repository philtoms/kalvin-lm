"""KLine - Fundamental unit for knowledge graph operations."""

from __future__ import annotations

from typing import TypeAlias

# === Core Types ===

# Type alias for a KNode (64-bit int)
KNode: TypeAlias = int

# Type alias for KNodes
KNodes: TypeAlias = int | None | list[int]

# Type alias for Significance (64-bit int with S1/S2/S3/S4 encoding)
KSig: TypeAlias = int


class KLine:
    """A structure with a 64-bit significance signature and list of child KNodes.

    Attributes:
        signature: 64-bit integer signature
        nodes: List of child KNode integers
    """

    def __init__(self, signature: KSig, nodes: KNodes | KNode | None, dbg_text: str = ""):
        self.signature = signature
        self.nodes = nodes
        self.dbg_text = dbg_text

    # === Subtype helpers ===

    def is_unsigned(self) -> bool:
        """Check if this is an unsigned KLine (nodes is None)."""
        return self.nodes is None

    def is_signed(self) -> bool:
        """Check if this is a signed KLine (nodes is a single KNode)."""
        return isinstance(self.nodes, int)

    def is_canonized(self) -> bool:
        """Check if this is a canonized KLine (nodes is a list of KNodes)."""
        return isinstance(self.nodes, list)

    def as_node_list(self) -> list[KNode]:
        """Get nodes as a list, handling all three subtypes.

        Returns:
        - Empty list if nodes is None
        - Single-element list if nodes is int
        - The list itself if nodes is list
        """
        if self.nodes is None:
            return []
        if isinstance(self.nodes, int):
            return [self.nodes]
        return self.nodes
    
    def signifies(self, query: KSig) -> bool:
        """Check if this KLine signifies a query via AND operation.

        Args:
            query: The query signature to signify

        Returns:
            True if (self & query) != 0
        """
        return  (self.signature & query) != 0

    def filter(self, signature: KSig) -> list[KSig]:
        """Return a list of nodes with signature removed

        Args:
            signature: The signature value to remove

        Returns:
            nodes with signature removed
        """
        return [node for node in self.as_node_list() if node != signature]

    def mask(self, keep: set) -> list[KSig]:
        """Return a list of nodes with mask removed

        Args:
            keep: The set of signatures to (index) preserve

        Returns:
            nodes with mask applied
        """
        masked = []
        for node in self.as_node_list():
            if node in keep:
                masked.append(node)
            else:
                masked.append(0)
        return masked

    @classmethod
    def create(cls, significance: KSig, token: KNode, nodes: KNodes | KNode | None, dbg_text: str = "") -> "KLine":
        """Create a KLine from significance, token, and nodes.

        The signature is constructed from significance | token.

        Args:
            significance: Significance value to OR with token
            token: Token value to OR with significance
            nodes: List of child KNode integers

        Returns:
            KLine with signature = significance | token
        """
        return cls(signature=significance | token, nodes=nodes, dbg_text=dbg_text)

    def equals(self, other: KLine) -> bool:
        if isinstance(self.nodes, list) and isinstance(other.nodes, list):
            if len(self.nodes) == len(other.nodes):
                for i in range(len(self.nodes)):
                    if self.nodes[i] != other.nodes[i]:
                        return False
                return True
        elif self.nodes == other.nodes:
            return True
        return False


# Singleton for null/empty KLine
KNone = KLine(signature=0, nodes=None, dbg_text="")
