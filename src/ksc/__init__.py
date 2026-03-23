"""KScript compiler module.

Provides both CLI and Python API for compiling .ks files to JSON/JSONL format.
"""

from pathlib import Path

from .ast import KScriptFile
from .compiler import CompiledEntry, Compiler
from .lexer import Lexer
from .output import read_model, write_json, write_jsonl
from .parser import Parser

__all__ = [
    "KScript",
    "CompiledEntry",
    "Lexer",
    "Parser",
    "Compiler",
    "KScriptFile",
    "write_json",
    "write_jsonl",
    "read_model",
]


class KScript:
    """Compiled KScript model supporting incremental construction.

    Usage:
        # Inline model generation from string
        model1 = KScript("A == B")
        # -> KScript object containing: [{A: "B"}, {B: "A"}]

        # Load from JSON/JSONL file
        model2 = KScript("model.json")  # or .jsonl

        # Extend model with script
        model3 = KScript("script.ks", base=model1)

        # Save to JSON or JSONL (based on suffix)
        model3.output("output.json")
    """

    def __init__(
        self, source: str | Path, base: "KScript | None" = None
    ):
        """Compile KScript from string or file path.

        Args:
            source: KScript source string or path to file.
                    Supported file types:
                    - .ks: KScript source file
                    - .json: JSON model file
                    - .jsonl: JSONL model file
                    If Path(source).exists() returns True, treats as file path.
            base: Optional existing KScript to extend.
        """
        # Start with base entries if provided
        self._entries: list[CompiledEntry] = list(base._entries) if base else []

        # Determine if source is a file path or inline string
        source_str = str(source)
        source_path = Path(source_str)

        if source_path.exists():
            suffix = source_path.suffix.lower()
            if suffix in (".json", ".jsonl"):
                # Load from model file
                self._entries.extend(read_model(source_path))
            else:
                # Compile from source file
                source_code = source_path.read_text()
                self._compile_source(source_code)
        else:
            # Treat as inline source string
            self._compile_source(source_str)

    def _compile_source(self, source: str) -> None:
        """Compile source code and append entries."""
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()
        new_entries = Compiler().compile(kscript_file)
        self._entries.extend(new_entries)

    @property
    def entries(self) -> list[CompiledEntry]:
        """Return all compiled entries."""
        return self._entries

    def output(self, path: str | Path) -> None:
        """Write entries to JSON or JSONL file based on suffix."""
        output_path = Path(path)
        suffix = output_path.suffix.lower()
        if suffix == ".json":
            write_json(self._entries, output_path)
        else:
            write_jsonl(self._entries, output_path)

    def to_model(self) -> list[dict[str, str | None | list[str]]]:
        """Return entries as list of dicts (preserves duplicate signatures)."""
        return [{e.signature: e.nodes} for e in self._entries]

    def to_dict(self) -> dict[str, str | None | list[str]]:
        """Merge entries into dict format (later entries override earlier)."""
        result: dict[str, str | None | list[str]] = {}
        for entry in self._entries:
            result[entry.signature] = entry.nodes
        return result
