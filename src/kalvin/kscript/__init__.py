"""KScript parser and compiler for Kalvin models."""

from .ast import Identifier, KLineExpr, KNodeRef, KScript, SignificanceType
from .compiler import CompileResult, Compiler
from .lexer import Lexer, LexerError
from .parser import ParseError, Parser
from .tokens import Token, TokenType

__all__ = [
    # AST
    "KScript",
    "KLineExpr",
    "KNodeRef",
    "Identifier",
    "SignificanceType",
    # Compiler
    "Compiler",
    "CompileResult",
    # Lexer
    "Lexer",
    "LexerError",
    "Token",
    "TokenType",
    # Parser
    "Parser",
    "ParseError",
]


def parse(source: str) -> KScript:
    """Parse KScript source to AST."""
    return Parser.from_source(source).parse()


def compile_script(source: str, tokenizer=None) -> CompileResult:
    """Parse and compile KScript source to Model."""
    script = parse(source)
    compiler = Compiler(tokenizer)
    return compiler.compile(script)
