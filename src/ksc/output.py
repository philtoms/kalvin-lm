"""Output writers for compiled KScript entries."""

import json
from pathlib import Path

from .compiler import CompiledEntry


def write_json(entries: list[CompiledEntry], path: Path) -> None:
    """Write entries to JSON file as a list.

    Output format:
    [{"L": "M"}, {"L": "O"}]
    """
    model = [{e.signature: e.nodes} for e in entries]
    with open(path, "w") as f:
        json.dump(model, f, indent=2)


def write_jsonl(entries: list[CompiledEntry], path: Path) -> None:
    """Write entries to JSONL file.

    Each entry becomes a separate line:
    {"L": "M"}
    {"L": "O"}
    """
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps({entry.signature: entry.nodes}) + "\n")


def read_model(path: Path) -> list[CompiledEntry]:
    """Read entries from JSON or JSONL file based on suffix.

    JSON format: [{"A": "B"}, {"B": "A"}]
    JSONL format: {"A": "B"}\\n{"B": "A"}\\n
    """
    suffix = path.suffix.lower()
    with open(path) as f:
        if suffix == ".json":
            model = json.load(f)
            entries = []
            for item in model:
                for sig, nodes in item.items():
                    entries.append(CompiledEntry(sig, nodes))
            return entries
        else:  # .jsonl or any other extension
            entries = []
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    for sig, nodes in item.items():
                        entries.append(CompiledEntry(sig, nodes))
            return entries
