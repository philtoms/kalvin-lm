"""KScript compiler module.

Provides both CLI and Python API for compiling .ks files to JSON/JSONL/binary format.
"""

from pathlib import Path

from kalvin.mod_tokenizer import ModTokenizer, Mod32Tokenizer

from .ast import KScriptFile
from .compiler import CompiledEntry, Compiler
from .lexer import Lexer
from .output import read_bin, read_json, write_bin, write_json, write_jsonl
from .parser import Parser
import json

__all__ = [
    "KScript",
    "CompiledEntry",
    "Lexer",
    "Parser",
    "Compiler",
    "KScriptFile",
    "write_json",
    "write_jsonl",
    "write_bin",
    "read_json",
    "read_bin",
]


class KScript:
    """Compiled KScript model supporting incremental construction.

    Usage:
        # Inline model generation from string
        model1 = KScript("A == B")
        # -> KScript object containing: [{sig: token_id_A, nodes: token_id_B}, ...]

        # Load from JSON/JSONL file
        model2 = KScript("model.json")  # or .jsonl

        # Load from binary file (tokenized)
        model3 = KScript("model.bin")

        # Extend model with script
        model4 = KScript("script.ks", base=model1)

        # Save to JSON, JSONL or binary (based on suffix)
        model4.output("output.json")
        model4.output("output.bin")
    """

    def __init__(
        self,
        source: str | Path,
        base: "KScript | None" = None,
        tokenizer: ModTokenizer | None = None,
    ):
        """Compile KScript from string or file path.

        Args:
            source: KScript source string or path to file.
                    Supported file types:
                    - .ks: KScript source file
                    - .json: JSON model file
                    - .jsonl: JSONL model file
                    - .bin: Binary model file (tokenized)
                    If Path(source).exists() returns True, treats as file path.
            base: Optional existing KScript to extend.
            tokenizer: Tokenizer for encoding (default: Mod32Tokenizer).
        """
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self._entries: list[CompiledEntry] = list(base._entries) if base else []

        source_path = Path(source)
        if source_path.exists():
            suffix = source_path.suffix.lower()
            if suffix == ".bin":
                self._entries.extend(read_bin(source_path))
            elif suffix in (".json", ".jsonl"):
                self._entries.extend(read_json(source_path, self.tokenizer))
            else:
                source_code = source_path.read_text()
                self._compile_source(source_code)
        else:
            # Treat as inline source string
            self._compile_source(str(source))

    def _compile_source(self, source: str) -> None:
        """Compile source code and append entries."""
        tokens = Lexer(source).tokenize()
        kscript_file = Parser(tokens).parse()
        new_entries = Compiler(self.tokenizer).compile(kscript_file)
        self._entries.extend(new_entries)

    @property
    def entries(self) -> list[CompiledEntry]:
        """Return all compiled entries."""
        return self._entries

    def output(self, path: str | Path) -> None:
        """Write entries to file based on suffix.

        Suffix determines format:
        - .json: JSON array
        - .jsonl: JSONL (line-delimited)
        - .bin: Binary format (tokenized)
        """
        output_path = Path(path)
        suffix = output_path.suffix.lower()
        if suffix == ".bin":
            write_bin(self._entries, output_path)
        elif suffix == ".json":
            write_json(self._entries, output_path, self.tokenizer)
        else:
            write_jsonl(self._entries, output_path, self.tokenizer)

    def to_jsonl(self) -> list[str]:
        """Return JSONL list of entries."""
        result = []
        for entry in self._entries:
            sig, nodes = entry.decode(self.tokenizer)
            result.append(json.dumps({sig: nodes}))
        return result
