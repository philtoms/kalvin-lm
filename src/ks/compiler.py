"""KScript v3 compiler orchestrator — wires the four-stage pipeline.

The Compiler class is a pure orchestrator: it creates and connects the
pipeline stages but contains no encoding logic of its own.

Pipeline (spec §1.1)::

    Source → Lexer → Parser → ASTEmitter (+ BindingScope) → TokenEncoder
                                                        ↓
                                                  list[KLine]

Design:
  - A BindingScope is always created for every compilation. When the
    tokenizer has no word lists, resolve() returns None and all characters
    pass through raw.
  - MTS expansion always runs in the ASTEmitter. The TokenEncoder handles
    the actual encoding via the tokenizer.

The ``compile_source`` convenience function creates a Lexer, Parser, and
Compiler in sequence — the typical one-shot usage for compiling a source
string.

Spec ref: @specs/kscript.md §1.1 (pipeline), §12.2 (compiler orchestrator),
          §13 (public API).
"""

from __future__ import annotations

from kalvin.abstract import KTokenizer
from kalvin.kline import KLine
from kalvin.nlp_tokenizer import NLPTokenizer

from .ast import KScriptFile
from .ast_emitter import ASTEmitter, SymbolicEntry
from .binding_scope import BindingScope
from .token_encoder import TokenEncoder

__all__ = ["Compiler", "compile_source"]


class Compiler:
    """Compiles a KScriptFile AST into a list of KLine objects.

    Pure orchestrator — wires together BindingScope, ASTEmitter, and
    TokenEncoder.  No encoding logic lives here.

    Args:
        tokenizer: Tokenizer for encoding strings to uint64 values.
            Defaults to NLPTokenizer() (tokenizer data is mandatory).
        dev: Enable development/diagnostic mode (populates dbg).
    """

    def __init__(
        self,
        tokenizer: KTokenizer | None = None,
        dev: bool = False,
    ) -> None:
        self.tokenizer: KTokenizer = tokenizer or NLPTokenizer()
        self.dev = dev
        self.entries: list[KLine] = []

    def compile(self, file: KScriptFile) -> list[KLine]:
        """Compile a KScriptFile AST into encoded entries.

        Pipeline:
          1. Create BindingScope, push root scope (always, no mode switch).
          2. Create ASTEmitter with scope, emit symbolic entries.
          3. Create TokenEncoder, encode symbolic entries to KLine objects.

        Args:
            file: Parsed KScriptFile AST.

        Returns:
            Ordered list of KLine objects with populated dbg.
        """
        scope = BindingScope()
        scope.push_scope()  # root scope

        emitter = ASTEmitter(scope=scope, dev=self.dev)
        symbolic: list[SymbolicEntry] = emitter.emit(file)

        encoder = TokenEncoder(tokenizer=self.tokenizer, dev=self.dev)
        self.entries = encoder.encode_entries(symbolic)
        return self.entries


def compile_source(
    source: str,
    tokenizer: KTokenizer | None = None,
    dev: bool = False,
) -> list[KLine]:
    """Compile a KScript source string into encoded entries.

    Convenience function that creates Lexer, Parser, and Compiler
    in sequence.

    Args:
        source: KScript source code string.
        tokenizer: Tokenizer for encoding strings to uint64 values.
            Defaults to NLPTokenizer() (tokenizer data is mandatory).
        dev: Enable development/diagnostic mode.

    Returns:
        Ordered list of KLine objects with populated dbg.
    """
    from .lexer import Lexer
    from .parser import Parser

    tokens = Lexer(source).tokenize()
    kfile = Parser(tokens).parse()
    return Compiler(tokenizer, dev=dev).compile(kfile)
