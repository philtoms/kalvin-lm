"""Output writers for compiled KScript entries."""

import json
import struct
from pathlib import Path

from kalvin.mod_tokenizer import ModTokenizer

from .compiler import CompiledEntry


def write_json(
    entries: list[CompiledEntry], path: Path, tokenizer: ModTokenizer
) -> None:
    """Write entries to JSON file as a list.

    Output format:
    [{"L": "M"}, {"L": "O"}]
    """
    model = []
    for e in entries:
        sig, nodes = e.decode(tokenizer)
        model.append({sig: nodes})
    with open(path, "w") as f:
        json.dump(model, f, indent=2)


def write_jsonl(
    entries: list[CompiledEntry], path: Path, tokenizer: ModTokenizer
) -> None:
    """Write entries to JSONL file.

    Each entry becomes a separate line:
    {"L": "M"}
    {"L": "O"}
    """
    with open(path, "w") as f:
        for entry in entries:
            sig, nodes = entry.decode(tokenizer)
            f.write(json.dumps({sig: nodes}) + "\n")


def write_bin(entries: list[CompiledEntry], path: Path) -> None:
    """Write entries to binary file.

    Binary format:
    - Header: 4 bytes magic "KSC1"
    - Entry count: 4 bytes (uint32, little-endian)
    - Per entry:
      - signature: 8 bytes (uint64, little-endian)
      - node_type: 1 byte (0=None, 1=int, 2=list)
      - if int: 8 bytes (uint64)
      - if list: 4 bytes count + N*8 bytes (uint64 each)
    """
    with open(path, "wb") as f:
        f.write(b"KSC1")
        f.write(struct.pack("<I", len(entries)))
        for entry in entries:
            f.write(struct.pack("<Q", entry.signature))
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


def read_bin(path: Path) -> list[CompiledEntry]:
    """Read entries from binary file."""
    entries = []
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"KSC1":
            raise ValueError(f"Invalid magic: {magic}")
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


def read_json(path: Path, tokenizer: ModTokenizer) -> list[CompiledEntry]:
    """Read entries from JSON or JSONL file based on suffix.

    JSON format: [{"A": "B"}, {"B": "A"}]
    JSONL format: {"A": "B"}\\n{"B": "A"}\\n
    """
    suffix = path.suffix.lower()
    entries = []
    with open(path) as f:
        if suffix == ".json":
            model = json.load(f)
            for item in model:
                for sig, nodes in item.items():
                    entries.append(CompiledEntry.encode(sig, nodes, tokenizer))
        else:  # .jsonl or any other extension
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    for sig, nodes in item.items():
                        entries.append(CompiledEntry.encode(sig, nodes, tokenizer))
    return entries
