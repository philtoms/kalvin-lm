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
    # Convenience functions
    "parse",
    "compile_script",
]


def parse(source: str) -> KScript:
    """Parse KScript source to AST."""
    return Parser.from_source(source).parse()


def compile_script(source: str, tokenizer=None, model=None) -> CompileResult:
    """Parse and compile KScript source to Model.

    Args:
        source: KScript source code
        tokenizer: Optional tokenizer for string-to-token conversion
        model: Optional existing Model to add KLines to

    Returns:
        CompileResult with model, symbol_table, load_paths, save_path
    """
    script = parse(source)
    compiler = Compiler(tokenizer, model)
    return compiler.compile(script)
