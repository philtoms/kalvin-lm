"""KScript v3 compiler orchestrator — wires the four-stage pipeline.

The Compiler class is a pure orchestrator: it creates and connects the
pipeline stages but contains no encoding logic of its own.

Pipeline (spec §1.1)::

    Source → Lexer → Parser → ASTEmitter (+ BindingScope) → TokenEncoder
                                                        ↓
                                              list[CompiledEntry]

Key v3 design decisions (vs v2):
  - **BindingScope always created**: No Mod32 mode switch.  A BindingScope
    is always instantiated for every compilation.  When the tokenizer is
    Mod32/Mod64 (supports_mcs=True), the scope simply has no word lists,
    so resolve() always returns None and all characters pass through raw.
  - **No skip_mcs parameter**: MCS expansion always runs in the ASTEmitter.
    The TokenEncoder handles the actual encoding via the tokenizer.

The ``compile_source`` convenience function creates a Lexer, Parser, and
Compiler in sequence — the typical one-shot usage for compiling a source
string.

Spec ref: @specs/kscript.md §1.1 (pipeline), §12.2 (compiler orchestrator),
          §13 (public API).
"""

from __future__ import annotations

from kalvin.abstract import KTokenizer
from kalvin.mod_tokenizer import Mod32Tokenizer

from .ast import KScriptFile
from .ast_emitter import ASTEmitter, SymbolicEntry
from .binding_scope import BindingScope
from .token_encoder import CompiledEntry, TokenEncoder


__all__ = ["Compiler", "compile_source"]


class Compiler:
    """Compiles a KScriptFile AST into a list of CompiledEntry objects.

    Pure orchestrator — wires together BindingScope, ASTEmitter, and
    TokenEncoder.  No encoding logic lives here.

    Args:
        tokenizer: Tokenizer for encoding strings to uint64 values.
            Defaults to Mod32Tokenizer().
        dev: Enable development/diagnostic mode (populates dbg_text).
    """

    def __init__(
        self,
        tokenizer: KTokenizer | None = None,
        dev: bool = False,
    ) -> None:
        self.tokenizer: KTokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev
        self.entries: list[CompiledEntry] = []

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile AST into encoded entries.

        Pipeline:
          1. Create BindingScope, push root scope (always, no mode switch).
          2. Create ASTEmitter with scope, emit symbolic entries.
          3. Create TokenEncoder, encode symbolic entries to CompiledEntry.

        Args:
            file: Parsed KScriptFile AST.

        Returns:
            Ordered list of CompiledEntry objects.
        """
        # 1. Always create BindingScope — no Mod32 mode switch
        scope = BindingScope()
        scope.push_scope()  # Root scope

        # 2. Emit symbolic entries
        emitter = ASTEmitter(scope=scope, dev=self.dev)
        symbolic: list[SymbolicEntry] = emitter.emit(file)

        # 3. Encode to token IDs
        encoder = TokenEncoder(tokenizer=self.tokenizer, dev=self.dev)
        self.entries = encoder.encode_entries(symbolic)
        return self.entries


def compile_source(
    source: str,
    tokenizer: KTokenizer | None = None,
    dev: bool = False,
) -> list[CompiledEntry]:
    """Compile a KScript source string into encoded entries.

    Convenience function that creates Lexer, Parser, and Compiler
    in sequence.

    Args:
        source: KScript source code string.
        tokenizer: Tokenizer for encoding (default: Mod32Tokenizer).
        dev: Enable development/diagnostic mode.

    Returns:
        Ordered list of CompiledEntry objects.
    """
    from .lexer import Lexer
    from .parser import Parser

    tokens = Lexer(source).tokenize()
    kfile = Parser(tokens).parse()
    return Compiler(tokenizer, dev=dev).compile(kfile)
