"""KScript compiler with construct binding semantics.

This module is a thin orchestrator that wires together:
  - BindingResolver (NLP mode only): walks AST comments → NLPSymbolTable
  - ASTEmitter: walks AST and yields symbolic entries (sig_str, nodes_strs, op)
  - TokenEncoder: converts symbolic entries to tokenized CompiledEntry objects

In Mod32/Mod64 mode (``supports_mcs is True``), the binding resolver is
completely skipped — no imports, no symbol table construction.  In NLP mode
(``supports_mcs is False``), the resolver runs before emission and stores
its ``NLPSymbolTable`` on the Compiler instance for downstream consumers.

Spec ref: @kscript-nlp-binding §1.1 (pipeline diagram), §1.2 (mode selection)

Op mappings (using abbreviated property chains):
  COUNTERSIGN  -> {sig: node}, {node: sig}
  CANONIZE     -> {p[-1].(node or sig): p.sig for p in r.primaries}
  CONNOTATE    -> {sig: node}
  UNDERSIGN    -> {node: sig}
  UNSIGNED     -> sig | S4 (S4=0, no bits)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from kalvin.kline import KLine, KNodes, KSig
from kalvin.abstract import KTokenizer
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.expand import D_MAX
from kalvin.signature import is_literal_node

if TYPE_CHECKING:
    from .symbol_table import NLPSymbolTable

from .ast import (
    Block,
    Construct,
    ConstructItem,
    KScriptFile,
    Literal,
    Node,
    PrimaryConstruct,
    Script,
    Signature,
)
from .parser import Parser
from .token import TokenType

# Re-export from split modules
from .ast_emitter import ASTEmitter, SymbolicEntry
from .token_encoder import CompiledEntry, TokenEncoder

__all__ = [
    "CompiledEntry",
    "ASTEmitter",
    "SymbolicEntry",
    "TokenEncoder",
    "Compiler",
    "compile_source",
]


class Compiler:
    """Compiles KScript AST to CompiledEntry list.

    Thin orchestrator: ASTEmitter walks the AST, TokenEncoder converts
    symbolic entries to token IDs.  In NLP mode (``supports_mcs is False``),
    the BindingResolver runs first to build an NLPSymbolTable from comment
    word lists.
    """

    def __init__(self, tokenizer: KTokenizer | None = None, dev: bool = False):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev
        self._symbol_table: NLPSymbolTable | None = None

    @property
    def symbol_table(self) -> NLPSymbolTable | None:
        """The NLP symbol table populated during compilation.

        ``None`` in Mod32/Mod64 mode; populated after ``compile()`` in NLP mode.
        """
        return self._symbol_table

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile to entries."""
        # NLP mode: run binding resolver to build symbol table from comments.
        # Mod32/Mod64 mode: skip entirely — no resolver import, no symbol table.
        if not self.tokenizer.supports_mcs:
            from .binding_resolver import BindingResolver

            resolver = BindingResolver()
            self._symbol_table = resolver.resolve(file)
        else:
            self._symbol_table = None

        emitter = ASTEmitter(
            dev=self.dev,
            skip_mcs=not self.tokenizer.supports_mcs,
            symbol_table=self._symbol_table,
        )
        if self._symbol_table is not None:
            self._symbol_table.rewind()
        symbolic = emitter.emit(file)

        encoder = TokenEncoder(tokenizer=self.tokenizer, dev=self.dev)
        self.entries = encoder.encode_entries(symbolic)
        return self.entries


def compile_source(source: str, tokenizer: KTokenizer | None = None, dev: bool = False) -> list[CompiledEntry]:
    """Compile KScript source string to entries."""
    from .lexer import Lexer

    tokens = Lexer(source).tokenize()
    kscript_file = Parser(tokens).parse()
    return Compiler(tokenizer, dev=dev).compile(kscript_file)
