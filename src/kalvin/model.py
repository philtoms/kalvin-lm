from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from struct import pack, unpack
from typing import TypeAlias, Iterator, Literal
import json


# High bit mask (bit 63)
HIGH_BIT_MASK = 0x8000_0000_0000_0000


# === Significance Bit Constants ===
# Layout: S1(bit56) | S1%(bits57-63) | S2(bits40-55) | S3(bits16-39) | Reserved(bits0-15)
# Higher bits = more significant: S1 > S2 > S3 > S4

# S1: single bit (bit 56) - prefix match indicator
S1_BIT = 1 << 56

# S1%: 7 bits (bits 57-63) for degree/percentage
S1_PCT_SHIFT = 57
S1_PCT_MASK = 0x7F << S1_PCT_SHIFT

# S2: 16 bits (bits 40-55)
S2_SHIFT = 40
S2_MASK = 0xFFFF << S2_SHIFT
S2_S1_PCT_SHIFT = 40   # S1 percentage within S2
S2_S2_PCT_SHIFT = 48   # S2 percentage within S2

# S3: 24 bits (bits 16-39)
S3_SHIFT = 16
S3_MASK = 0xFFFFFF << S3_SHIFT
S3_S1_PCT_SHIFT = 16   # S1% for unordered matches (bits 16-23)
S3_S2_PCT_SHIFT = 24   # S2% for unordered matches (bits 24-31)
S3_GEN_PCT_SHIFT = 32  # Generational S1% (bits 32-39)

# S4: no significance
S4_VALUE = 0


class KLineType(IntEnum):
    """Type based on the high bit of a key."""
    NODE = 1       # high bit = 1 (branch)
    EMBEDDING = 0  # high bit = 0 (leaf)


# Type alias for a KNode (64-bit int with high bit reserved for type)
KNode: TypeAlias = int

# Type alias for Significance (64-bit int with S1/S2/S3/S4 encoding)
Significance: TypeAlias = int


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


# === Significance Helper Functions ===

def has_s1(sig: Significance) -> bool:
    """Check if S1 bit is set (prefix match)."""
    return bool(sig & S1_BIT)


def get_s1_percentage(sig: Significance) -> int:
    """Extract S1 percentage (0-127)."""
    return (sig >> S1_PCT_SHIFT) & 0x7F


def get_s2(sig: Significance) -> int:
    """Extract full S2 value (0-65535)."""
    return (sig >> S2_SHIFT) & 0xFFFF


def get_s2_s1_percentage(sig: Significance) -> int:
    """Extract S2's S1 percentage (0-255)."""
    return (sig >> S2_S1_PCT_SHIFT) & 0xFF


def get_s2_s2_percentage(sig: Significance) -> int:
    """Extract S2's S2 percentage (0-255)."""
    return (sig >> S2_S2_PCT_SHIFT) & 0xFF


def build_s1(percentage: int = 100) -> Significance:
    """Build S1 significance with optional percentage.

    Args:
        percentage: Match percentage (0-100), default 100
    """
    scaled = max(0, min(100, percentage)) * 127 // 100  # Scale to 7 bits
    return S1_BIT | (scaled << S1_PCT_SHIFT)


def build_s2(s1_pct: int, s2_pct: int) -> Significance:
    """Build S2 significance.

    Args:
        s1_pct: Positional match percentage (0-100)
        s2_pct: Non-positional match percentage (0-100)
    """
    s1_scaled = max(0, min(100, s1_pct)) * 255 // 100
    s2_scaled = max(0, min(100, s2_pct)) * 255 // 100
    return (s1_scaled << S2_S1_PCT_SHIFT) | (s2_scaled << S2_S2_PCT_SHIFT)


def get_s3(sig: Significance) -> int:
    """Extract full S3 value (0-16777215)."""
    return (sig >> S3_SHIFT) & 0xFFFFFF


def get_s3_s1_percentage(sig: Significance) -> int:
    """Extract S3's S1 percentage for unordered matches (0-255)."""
    return (sig >> S3_S1_PCT_SHIFT) & 0xFF


def get_s3_s2_percentage(sig: Significance) -> int:
    """Extract S3's S2 percentage for unordered matches (0-255)."""
    return (sig >> S3_S2_PCT_SHIFT) & 0xFF


def get_s3_gen_percentage(sig: Significance) -> int:
    """Extract S3's generational S1 percentage (0-255)."""
    return (sig >> S3_GEN_PCT_SHIFT) & 0xFF


def build_s3(s1_pct: int, s2_pct: int, gen_pct: int) -> Significance:
    """Build S3 significance.

    Args:
        s1_pct: Unordered S1 match percentage (0-100)
        s2_pct: Unordered S2 match percentage (0-100)
        gen_pct: Generational S1 match percentage (0-100)
    """
    s1_scaled = max(0, min(100, s1_pct)) * 255 // 100
    s2_scaled = max(0, min(100, s2_pct)) * 255 // 100
    gen_scaled = max(0, min(100, gen_pct)) * 255 // 100
    return (
        (s1_scaled << S3_S1_PCT_SHIFT)
        | (s2_scaled << S3_S2_PCT_SHIFT)
        | (gen_scaled << S3_GEN_PCT_SHIFT)
    )


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

    def add(self, kline: KLine) -> bool:
        """Add a KLine, enforcing the duplicate key invariant.

        Invariant: Duplicate keys are allowed, but nodes must differ.
        - If s_key is new: add the kline
        - If s_key exists with different nodes: add the kline (allowed)
        - If s_key exists with same nodes: reject (would be exact duplicate)

        Args:
            kline: KLine to add

        Returns:
            True if added, False if rejected (exact duplicate)
        """
        for existing in self._klines:
            if existing.s_key == kline.s_key:
                if nodes_equal(existing.nodes, kline.nodes):
                    return False  # Exact duplicate, reject
        self._klines.append(kline)
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

    def calculate_significance(self, query: KLine, model: KLine) -> Significance:
        """Calculate significance between query and model KLines.

        Significance is comparable as integers - higher = more significant.
        S1 > S2 > S3 > S4.

        Returns:
            64-bit significance value
        """
        # Handle empty node lists
        if not query.nodes and not model.nodes:
            return build_s1(100)  # Perfect match
        if not query.nodes or not model.nodes:
            return S4_VALUE

        min_len = min(len(query.nodes), len(model.nodes))

        # Count S1 matches: positional equality (up to min length)
        s1_match_positions = set(
            i for i in range(min_len) if query.nodes[i] == model.nodes[i]
        )
        s1_matches = len(s1_match_positions)

        # S1: All prefix nodes match
        if s1_matches == min_len:
            percentage = 100  # All matched
            return build_s1(percentage)

        # S2: Partial match (some positional matches exist)
        if s1_matches > 0:
            s1_pct = (s1_matches * 100) // min_len

            # S2 matches: nodes at different positions
            model_set = set(model.nodes)
            s2_matches = 0
            for i, node in enumerate(query.nodes):
                if i in s1_match_positions:
                    continue  # Already counted as S1
                if node in model_set:
                    s2_matches += 1

            s2_pct = (s2_matches * 100) // len(query.nodes) if query.nodes else 0
            return build_s2(s1_pct, s2_pct)

        # S3: No positional matches, check unordered and generational
        model_set = set(model.nodes)
        query_set = set(query.nodes)

        # S3-Unordered S1: query nodes that exist in model (any position)
        unordered_s1_matches = query_set & model_set
        s3_s1_pct = (
            (len(unordered_s1_matches) * 100) // len(query_set) if query_set else 0
        )

        # S3-Unordered S2: query nodes whose children match model nodes
        s3_s2_matches = 0
        for node in query.nodes:
            if node in model_set:
                continue  # Already S1 match
            # Check if node's children intersect with model
            node_kline = self.find_by_key(node)
            if node_kline:
                node_children = set(node_kline.nodes)
                if node_children & model_set:
                    s3_s2_matches += 1

        s3_s2_pct = (
            (s3_s2_matches * 100) // len(query.nodes) if query.nodes else 0
        )

        # S3-Generational: query nodes whose descendants (at any depth) match model nodes
        gen_matches = 0
        for node in query.nodes:
            if node in model_set:
                continue  # Already S1 match
            # Collect all descendants of this node
            descendants = self._get_all_descendants(node, visited=set())
            if descendants & model_set:
                gen_matches += 1

        gen_pct = (gen_matches * 100) // len(query.nodes) if query.nodes else 0

        if s3_s1_pct > 0 or s3_s2_pct > 0 or gen_pct > 0:
            return build_s3(s3_s1_pct, s3_s2_pct, gen_pct)

        # S4: No match
        return S4_VALUE

    def _get_all_descendants(self, node_key: int, visited: set[int]) -> set[int]:
        """Recursively collect all descendant nodes.

        Args:
            node_key: The node to start from
            visited: Set of already visited nodes (cycle detection)

        Returns:
            Set of all descendant node keys
        """
        if node_key in visited:
            return set()
        visited.add(node_key)

        descendants: set[int] = set()
        kline = self.find_by_key(node_key)

        if kline is None:
            return descendants

        for child in kline.nodes:
            descendants.add(child)
            # Recursively get child's descendants
            child_descendants = self._get_all_descendants(child, visited.copy())
            descendants.update(child_descendants)

        return descendants

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
                if get_node_type(node_key) == KLineType.NODE:
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

    # === Serialization ===

    def to_bytes(self) -> bytes:
        """Serialize model to binary format.

        Binary layout:
        - 4 bytes: number of klines (uint32)
        - For each kline:
          - 8 bytes: s_key (uint64)
          - 4 bytes: node count (uint32)
          - N * 8 bytes: nodes (uint64 each)
        """
        parts = [pack("<I", len(self._klines))]
        for kline in self._klines:
            parts.append(pack("<Q", kline.s_key))
            parts.append(pack("<I", len(kline.nodes)))
            for node in kline.nodes:
                parts.append(pack("<Q", node))
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Model":
        """Deserialize model from binary format."""
        offset = 0
        kline_count = unpack("<I", data[offset:offset + 4])[0]
        offset += 4

        klines: list[KLine] = []
        for _ in range(kline_count):
            s_key = unpack("<Q", data[offset:offset + 8])[0]
            offset += 8
            node_count = unpack("<I", data[offset:offset + 4])[0]
            offset += 4
            nodes: list[KNode] = []
            for _ in range(node_count):
                node = unpack("<Q", data[offset:offset + 8])[0]
                offset += 8
                nodes.append(node)
            klines.append(KLine(s_key=s_key, nodes=nodes))

        return cls(klines=klines)

    def to_dict(self) -> dict:
        """Serialize model to dict (for JSON)."""
        return {
            "klines": [
                {"s_key": kline.s_key, "nodes": kline.nodes}
                for kline in self._klines
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        """Deserialize model from dict."""
        klines = [
            KLine(s_key=item["s_key"], nodes=item["nodes"])
            for item in data["klines"]
        ]
        return cls(klines=klines)

    def save(
        self,
        path: str | Path,
        format: Literal["binary", "json"] = "binary",
    ) -> None:
        """Save model to file.

        Args:
            path: File path to save to
            format: 'binary' (default, compact) or 'json' (human-readable)
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "binary"

        if format == "binary":
            path.write_bytes(self.to_bytes())
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path,
        format: Literal["binary", "json"] | None = None,
    ) -> "Model":
        """Load model from file.

        Args:
            path: File path to load from
            format: 'binary', 'json', or None (auto-detect from extension)

        Returns:
            Loaded Model instance
        """
        path = Path(path)

        # Auto-detect format from extension if not specified
        if format is None:
            format = "json" if path.suffix.lower() == ".json" else "binary"

        if format == "binary":
            return cls.from_bytes(path.read_bytes())
        else:
            return cls.from_dict(json.loads(path.read_text()))
