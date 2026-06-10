"""KScript compiler with construct binding semantics.

This module is a thin orchestrator that wires together:
  - BindingScope (NLP mode only): created with a root scope, passed to
    ASTEmitter which populates it during the walk
  - ASTEmitter: walks AST and yields symbolic entries (sig_str, nodes_strs, op)
  - TokenEncoder: converts symbolic entries to tokenized CompiledEntry objects

In Mod32/Mod64 mode (``supports_mcs is True``), no BindingScope is created.
In NLP mode (``supports_mcs is False``), a BindingScope with a root scope is
created and passed to the ASTEmitter, which resolves single-character bindings
inline during its single AST walk.

Spec ref: @kscript-nlp-binding §1.1 (pipeline diagram), §1.2 (mode selection)

Op mappings (using abbreviated property chains):
  COUNTERSIGN  -> {sig: node}, {node: sig}
  CANONIZE     -> {p[-1].(node or sig): p.sig for p in r.primaries}
  CONNOTATE    -> {sig: node}
  UNDERSIGN    -> {node: sig}
  UNSIGNED     -> sig | S4 (S4=0, no bits)
"""

from __future__ import annotations

from typing import TypeAlias

from kalvin.kline import KLine, KNodes, KSig
from kalvin.abstract import KTokenizer
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.expand import D_MAX
from kalvin.signature import is_literal_node

from .binding_scope import BindingScope

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
    a BindingScope is created with a root scope and passed to the ASTEmitter,
    which resolves single-character bindings inline during its walk.
    """

    def __init__(self, tokenizer: KTokenizer | None = None, dev: bool = False):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev
        self._scope: BindingScope | None = None

    @property
    def scope(self) -> BindingScope | None:
        """The BindingScope used during compilation.

        ``None`` in Mod32/Mod64 mode; populated with a root scope after
        ``compile()`` in NLP mode.
        """
        return self._scope

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile to entries."""
        # NLP mode: create BindingScope with root scope for the emitter.
        # Mod32/Mod64 mode: no scope needed.
        if not self.tokenizer.supports_mcs:
            self._scope = BindingScope()
            self._scope.push_scope()  # root scope for the emitter to populate
        else:
            self._scope = None

        emitter = ASTEmitter(
            dev=self.dev,
            skip_mcs=not self.tokenizer.supports_mcs,
            scope=self._scope,
        )
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
