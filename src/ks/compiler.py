"""KScript v3 compiler orchestrator — wires the four-stage pipeline.

The Compiler class is a pure orchestrator: it creates and connects the
pipeline stages but contains no encoding logic of its own.

Pipeline::

    Source → Lexer → Parser → ASTEmitter (+ BindingScope) → TokenEncoder
                                                        ↓
                                                  list[KValue]

Design:
  - A BindingScope is always created for every compilation. When the
    tokenizer has no word lists, resolve() returns None and all characters
    pass through raw.
  - compound expansion always runs in the ASTEmitter. The TokenEncoder handles
    the actual encoding via the tokenizer.

The ``compile_source`` convenience function creates a Lexer, Parser, and
Compiler in sequence — the typical one-shot usage for compiling a source
string.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from kalvin.abstract import KSignifier, KTokenizer
from kalvin.kvalue import KValue
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.signifier import NLPSignifier

from .ast import Import, KScriptFile
from .ast_emitter import ASTEmitter
from .binding_scope import BindingScope
from .lexer import Lexer
from .parser import Parser
from .token_encoder import TokenEncoder

__all__ = ["Compiler", "CompileError", "compile_source", "path_resolver"]


class CompileError(Exception):
    """Import resolution failure — see the Compiler's import walk.

    Attributes:
        node: The failing Import statement, when known.
    """

    def __init__(self, message: str, node: Import | None = None) -> None:
        prefix = f"Line {node.line}, column {node.column}: " if node else ""
        super().__init__(prefix + message)
        self.message = message
        self.node = node


def path_resolver(base: str | Path) -> Callable[[str], str]:
    """Resolve module names as ``<base>/<name>.ks`` source files."""
    base_dir = Path(base)

    def resolve(module: str) -> str:
        path = base_dir / f"{module}.ks"
        if not path.is_file():
            raise FileNotFoundError(f"no script file {path}")
        return path.read_text(encoding="utf-8")

    return resolve


class Compiler:
    """Compiles a KScriptFile AST into a list of KValue objects.

    Pure orchestrator — wires together BindingScope, ASTEmitter, and
    TokenEncoder.  No encoding logic lives here.

    Args:
        tokenizer: Tokenizer for encoding strings to uint64 values.
            Defaults to BPETokenizer() (tokenizer data is mandatory).
        dev: Enable development/diagnostic mode (populates dbg).
        word_bits: Shared word→bit table (see TokenEncoder).
        known_words: Prior-state words seeding the root binding scope.
        resolver: Module name → source, for ``import`` statements
            (see path_resolver).
    """

    def __init__(
        self,
        tokenizer: KTokenizer | None = None,
        signifier: KSignifier | None = None,
        dev: bool = False,
        word_bits: dict[str, int] | None = None,
        known_words: list[str] | None = None,
        resolver: Callable[[str], str] | None = None,
    ) -> None:
        self.tokenizer: KTokenizer = tokenizer or BPETokenizer()
        self._signifier: KSignifier = signifier or NLPSignifier()
        self.dev = dev
        self._word_bits = word_bits
        self._known_words = known_words
        self._resolver = resolver
        self.entries: list[KValue] = []

    def compile(self, file: KScriptFile) -> list[KValue]:
        """Compile a KScriptFile AST into encoded entries.

        Pipeline:
          1. Create BindingScope, push root scope (always, no mode switch).
          2. Create ASTEmitter with scope; emit imported modules (import
             order, depth-first) then the file's own constructs — one shared
             emitter and root binding frame, so a module's word lists seed
             the importing script exactly as known_words do, and compound
             registries / canon dedup span the boundary (value continuity).
          3. Create TokenEncoder, encode symbolic entries to KValue objects.

        Args:
            file: Parsed KScriptFile AST.

        Returns:
            Ordered list of KValue objects (each wrapping a KLine with
            populated dbg and a band-representative significance).
        """
        scope = BindingScope()
        scope.push_scope()  # root scope
        if self._known_words:
            # The prior state's words, in acquisition order — the outermost
            # word list. A char the script cannot bind itself resolves here,
            # continuing the earlier scripts' word binding; the script's own
            # lists are searched most-recent-first and always win.
            scope.add_words(self._known_words)

        emitter = ASTEmitter(scope=scope, dev=self.dev)
        self._imported: set[str] = set()
        self._emit_with_imports(file, scope, emitter, chain=())

        encoder = TokenEncoder(tokenizer=self.tokenizer, signifier=self._signifier, dev=self.dev, word_bits=self._word_bits)
        self.entries = encoder.encode_entries(emitter.entries)
        self.node_labels: dict[int, str] = dict(encoder.node_labels)
        return self.entries

    def _emit_with_imports(
        self, file: KScriptFile, scope: BindingScope, emitter: ASTEmitter,
        chain: tuple[str, ...],
    ) -> None:
        """Emit a file's imports (depth-first) before its own constructs.

        ``import mhall`` makes the module's encounter precede the script's:
        its entries prepend in import order, and — because it emits through
        this same emitter and root binding frame — its annotations feed the
        root scope's word lists (the script's own lists, added later, are
        searched first and always win) while its compound canons register
        once (a reference in the importing script reuses the module's
        decomposition and signature value).

        A file boundary resets the occurrence counters (a module's
        resolutions must not consume the counters its words owe the
        importing script) and closes any dangling annotation context. A
        module already imported anywhere in this compile never emits twice
        (diamond imports collapse); a module importing itself, directly or
        transitively, is a CompileError.
        """
        imports = [c for c in file.constructs if isinstance(c, Import)]
        body = [c for c in file.constructs if not isinstance(c, Import)]
        for imp in imports:
            if imp.module in chain:
                raise CompileError(
                    f"circular import: {' -> '.join((*chain, imp.module))}", imp
                )
            if imp.module in self._imported:
                continue
            self._imported.add(imp.module)
            source = self._resolve_import(imp)
            imported = Parser(Lexer(source).tokenize()).parse()
            self._emit_with_imports(imported, scope, emitter, (*chain, imp.module))
        scope.reset_counters()
        emitter.emit(KScriptFile(constructs=body))

    def _resolve_import(self, imp: Import) -> str:
        """The module's source text, via the configured resolver."""
        if self._resolver is None:
            raise CompileError(
                f"import '{imp.module}' requires a resolver "
                "(e.g. ks.compiler.path_resolver)", imp
            )
        try:
            return self._resolver(imp.module)
        except Exception as exc:
            raise CompileError(
                f"cannot resolve import '{imp.module}': {exc}", imp
            ) from exc


def compile_source(
    source: str,
    tokenizer: KTokenizer | None = None,
    signifier: KSignifier | None = None,
    dev: bool = False,
    word_bits: dict[str, int] | None = None,
    known_words: list[str] | None = None,
    resolver: Callable[[str], str] | None = None,
) -> list[KValue]:
    """Compile a KScript source string into encoded entries.

    Convenience function that creates Lexer, Parser, and Compiler
    in sequence.

    Args:
        source: KScript source code string.
        tokenizer: Tokenizer for encoding strings to uint64 values.
            Defaults to BPETokenizer() (tokenizer data is mandatory).
        dev: Enable development/diagnostic mode.
        resolver: Module name → source, for ``import`` statements
            (see path_resolver).

    Returns:
        Ordered list of KValue objects (each wrapping a KLine with
        populated dbg and a band-representative significance).
    """
    tokens = Lexer(source).tokenize()
    kfile = Parser(tokens).parse()
    return Compiler(
        tokenizer, signifier=signifier, dev=dev, word_bits=word_bits,
        known_words=known_words, resolver=resolver,
    ).compile(kfile)
