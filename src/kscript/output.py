"""Output module for JSON/JSONL/binary I/O."""

import json
import struct
from pathlib import Path

from kalvin.mod_tokenizer import ModTokenizer

from .compiler import CompiledEntry


# =============================================================================
# Write functions
# =============================================================================

def write_json(entries: list[CompiledEntry], path: Path, tokenizer: ModTokenizer) -> None:
    """Write entries to JSON array file.

    Each entry becomes an object in a JSON array:
    [{"A": "B"}, {"B": "A"}, {"C": ["D", "E"]}]

    Args:
        entries: List of CompiledEntry objects
        path: Output file path
        tokenizer: Tokenizer for decoding
    """
    data = []
    for entry in entries:
        sig, nodes = entry.decode(tokenizer)
        data.append({sig: nodes})

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_jsonl(entries: list[CompiledEntry], path: Path, tokenizer: ModTokenizer) -> None:
    """Write entries to JSONL (line-delimited JSON) file.

    Each entry becomes a separate line:
    {"A": "B"}
    {"B": "A"}
    {"C": ["D", "E"]}

    Args:
        entries: List of CompiledEntry objects
        path: Output file path
        tokenizer: Tokenizer for decoding
    """
    with open(path, "w") as f:
        for entry in entries:
            sig, nodes = entry.decode(tokenizer)
            f.write(json.dumps({sig: nodes}) + "\n")


def write_bin(entries: list[CompiledEntry], path: Path) -> None:
    """Write entries to binary format file.

    Binary format (KSC1):
    - Header: 4 bytes magic "KSC1"
    - Entry count: 4 bytes (uint32, little-endian)
    - Per entry:
      - signature: 8 bytes (uint64, little-endian)
      - node_type: 1 byte (0=None, 1=int, 2=list)
      - if int: 8 bytes (uint64)
      - if list: 4 bytes count + N*8 bytes (uint64 each)

    Args:
        entries: List of CompiledEntry objects
        path: Output file path
    """
    with open(path, "wb") as f:
        # Header
        f.write(b"KSC1")
        f.write(struct.pack("<I", len(entries)))

        for entry in entries:
            # Signature
            f.write(struct.pack("<Q", entry.signature))

            # Node type and value
            if entry.nodes is None:
                f.write(struct.pack("<B", 0))
            elif isinstance(entry.nodes, int):
                f.write(struct.pack("<B", 1))
                f.write(struct.pack("<Q", entry.nodes))
            else:
                f.write(struct.pack("<B", 2))
                f.write(struct.pack("<I", len(entry.nodes)))
                for node_id in entry.nodes:
                    f.write(struct.pack("<Q", node_id))


# =============================================================================
# Read functions
# =============================================================================

def read_json(path: Path, tokenizer: ModTokenizer) -> list[CompiledEntry]:
    """Read entries from JSON or JSONL file.

    Args:
        path: Input file path (.json or .jsonl)
        tokenizer: Tokenizer for encoding strings to token IDs

    Returns:
        List of CompiledEntry objects
    """
    content = path.read_text()

    # Try JSON array first
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            data = [data]
    except json.JSONDecodeError:
        # Fall back to JSONL
        data = [json.loads(line) for line in content.strip().split("\n") if line.strip()]

    entries: list[CompiledEntry] = []
    for obj in data:
        for sig, nodes in obj.items():
            entry = CompiledEntry.encode(sig, nodes, tokenizer)
            entries.append(entry)

    return entries


def read_bin(path: Path) -> list[CompiledEntry]:
    """Read entries from binary format file.

    Args:
        path: Input file path (.bin)

    Returns:
        List of CompiledEntry objects

    Raises:
        ValueError: If file is not valid KSC1 format
    """
    entries: list[CompiledEntry] = []

    with open(path, "rb") as f:
        # Header
        magic = f.read(4)
        if magic != b"KSC1":
            raise ValueError(f"Invalid magic: {magic!r}, expected b'KSC1'")

        count = struct.unpack("<I", f.read(4))[0]

        for _ in range(count):
            sig = struct.unpack("<Q", f.read(8))[0]
            node_type = struct.unpack("<B", f.read(1))[0]

            if node_type == 0:
                nodes = None
            elif node_type == 1:
                nodes = struct.unpack("<Q", f.read(8))[0]
            else:
                list_len = struct.unpack("<I", f.read(4))[0]
                nodes = [struct.unpack("<Q", f.read(8))[0] for _ in range(list_len)]

            entries.append(CompiledEntry(signature=sig, nodes=nodes))

    return entries
