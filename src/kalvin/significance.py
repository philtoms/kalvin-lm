"""Significance calculation for KLine matching."""

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from kalvin.model import KLine, Model


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


# Type alias for Significance (64-bit int with S1/S2/S3/S4 encoding)
Significance: TypeAlias = int


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


# === Significance Calculation ===

def _get_all_descendants(model: "Model", node_key: int, visited: set[int]) -> set[int]:
    """Recursively collect all descendant nodes.

    Args:
        model: The Model to search for KLines
        node_key: The node to start from
        visited: Set of already visited nodes (cycle detection)

    Returns:
        Set of all descendant node keys
    """
    if node_key in visited:
        return set()
    visited.add(node_key)

    descendants: set[int] = set()
    kline = model.find_by_key(node_key)

    if kline is None:
        return descendants

    for child in kline.nodes:
        descendants.add(child)
        # Recursively get child's descendants
        child_descendants = _get_all_descendants(model, child, visited.copy())
        descendants.update(child_descendants)

    return descendants


def calculate_significance(model: "Model", query: "KLine", target: "KLine") -> Significance:
    """Calculate significance between query and target KLines.

    Significance is comparable as integers - higher = more significant.
    S1 > S2 > S3 > S4.

    Args:
        model: The Model containing the KLines (for descendant lookup)
        query: The query KLine
        target: The target KLine to compare against

    Returns:
        64-bit significance value
    """
    # Handle empty node lists
    if not query.nodes and not target.nodes:
        return build_s1(100)  # Perfect match
    if not query.nodes or not target.nodes:
        return S4_VALUE

    min_len = min(len(query.nodes), len(target.nodes))

    # Count S1 matches: positional equality (up to min length)
    s1_match_positions = set(
        i for i in range(min_len) if query.nodes[i] == target.nodes[i]
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
        target_set = set(target.nodes)
        s2_matches = 0
        for i, node in enumerate(query.nodes):
            if i in s1_match_positions:
                continue  # Already counted as S1
            if node in target_set:
                s2_matches += 1

        s2_pct = (s2_matches * 100) // len(query.nodes) if query.nodes else 0
        return build_s2(s1_pct, s2_pct)

    # S3: No positional matches, check unordered and generational
    target_set = set(target.nodes)
    query_set = set(query.nodes)

    # S3-Unordered S1: query nodes that exist in target (any position)
    unordered_s1_matches = query_set & target_set
    s3_s1_pct = (
        (len(unordered_s1_matches) * 100) // len(query_set) if query_set else 0
    )

    # S3-Unordered S2: query nodes whose children match target nodes
    s3_s2_matches = 0
    for node in query.nodes:
        if node in target_set:
            continue  # Already S1 match
        # Check if node's children intersect with target
        node_kline = model.find_by_key(node)
        if node_kline:
            node_children = set(node_kline.nodes)
            if node_children & target_set:
                s3_s2_matches += 1

    s3_s2_pct = (
        (s3_s2_matches * 100) // len(query.nodes) if query.nodes else 0
    )

    # S3-Generational: query nodes whose descendants (at any depth) match target nodes
    gen_matches = 0
    for node in query.nodes:
        if node in target_set:
            continue  # Already S1 match
        # Collect all descendants of this node
        descendants = _get_all_descendants(model, node, visited=set())
        if descendants & target_set:
            gen_matches += 1

    gen_pct = (gen_matches * 100) // len(query.nodes) if query.nodes else 0

    if s3_s1_pct > 0 or s3_s2_pct > 0 or gen_pct > 0:
        return build_s3(s3_s1_pct, s3_s2_pct, gen_pct)

    # S4: No match
    return S4_VALUE
