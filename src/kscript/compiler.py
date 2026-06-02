"""KScript compiler with construct binding semantics.

This module is now a thin orchestrator that wires together:
  - ASTEmitter: walks AST and yields symbolic entries (sig_str, nodes_strs, op)
  - TokenEncoder: converts symbolic entries to tokenized CompiledEntry objects

The Compiler class preserves backward compatibility — same API, same results.

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
from kalvin.mod_tokenizer import Mod32Tokenizer, ModTokenizer
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
    symbolic entries to token IDs.
    """

    def __init__(self, tokenizer: ModTokenizer | None = None, dev: bool = False):
        self.entries: list[CompiledEntry] = []
        self.tokenizer = tokenizer or Mod32Tokenizer()
        self.dev = dev

    def compile(self, file: KScriptFile) -> list[CompiledEntry]:
        """Compile a KScriptFile to entries."""
        emitter = ASTEmitter(dev=self.dev)
        symbolic = emitter.emit(file)

        encoder = TokenEncoder(tokenizer=self.tokenizer, dev=self.dev)
        self.entries = encoder.encode_entries(symbolic)
        return self.entries


def compile_source(source: str, tokenizer: ModTokenizer | None = None, dev: bool = False) -> list[CompiledEntry]:
    """Compile KScript source string to entries."""
    from .lexer import Lexer

    tokens = Lexer(source).tokenize()
    kscript_file = Parser(tokens).parse()
    return Compiler(tokenizer, dev=dev).compile(kscript_file)
