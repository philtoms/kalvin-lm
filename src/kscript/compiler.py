"""KScript compiler — single-pass orchestrator.

This module wires together three stages in a single pass:

  1. **Parse**: Lexer + Parser produce a KScriptFile AST.
  2. **Emit**: ASTEmitter walks the AST and yields symbolic entries
     (sig_str, nodes_strs, op).  In NLP mode (``supports_mcs is False``),
     a ``BindingScope`` is created inline and passed to the emitter so
     it can resolve single-character bindings during its walk.
  3. **Encode**: TokenEncoder converts symbolic entries to tokenized
     CompiledEntry objects.

In Mod32/Mod64 mode (``supports_mcs is True``), no BindingScope is created
and the emitter operates in pure backward-compatible mode.

Spec ref: @kscript-nlp-binding §1.1 (pipeline diagram), §1.2 (mode selection)

Op mappings (using abbreviated property chains):
  COUNTERSIGN  -> {sig: node}, {node: sig}
  CANONIZE     -> {p[-1].(node or sig): p.sig for p in r.primaries}
  CONNOTATE    -> {sig: node}
  UNDERSIGN    -> {node: sig}
  UNSIGNED     -> sig | S4 (S4=0, no bits)
"""

from __future__ import annotations

from kalvin.kline import KLine, KNodes, KSig
from kalvin.abstract import KTokenizer
from kalvin.mod_tokenizer import Mod32Tokenizer
from kalvin.expand import D_MAX
from kalvin.signature import is_literal_node

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
from .binding_scope import BindingScope
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

    Single-pass orchestrator: ASTEmitter walks the AST (with optional
    BindingScope for NLP mode), TokenEncoder converts symbolic entries
    to token IDs.
    """

    def __init__(self, tokenizer: KTokenizer | None = None, dev: bool = False):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile to entries."""
        # NLP mode: create a BindingScope with a root scope for inline
        # resolution during emission.  Mod32/Mod64 mode: no scope needed.
        scope = None
        if not self.tokenizer.supports_mcs:
            scope = BindingScope()
            scope.push_scope()  # Root scope (was done by BindingResolver)

        emitter = ASTEmitter(
            dev=self.dev,
            skip_mcs=not self.tokenizer.supports_mcs,
            scope=scope,
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
