"""KScript parser and interpreter for Kalvin models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ast import Identifier, KLineExpr, KNodeRef, KScriptAst, SignificanceType
from .interpreter import InterpretError, InterpretResult, Interpreter
from .lexer import Lexer, LexerError
from .parser import ParseError, Parser
from .tokens import Token, TokenType
from .mod_tokenizer import Mod32Tokenizer, Mod64Tokenizer, Mod128Tokenizer


if TYPE_CHECKING:
    from kalvin.abstract import KAgent

__all__ = [
    # AST
    "KScriptAst",
    "KLineExpr",
    "KNodeRef",
    "Identifier",
    "SignificanceType",
    # Interpreter
    "Interpreter",
    "InterpretResult",
    "InterpretError",
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
    "interpret_script",
    # Tokenizer
    "Mod32Tokenizer",
    "Mod64Tokenizer",
    "Mod128Tokenizer",
]


def parse(source: str) -> KScriptAst:
    """Parse KScript source to AST."""
    return Parser.from_source(source).parse()


def interpret_script(source: str, agent: KAgent | None = None) -> InterpretResult:
    """Parse and interpret KScript source using identity/compound semantics.

    - Single-char identifiers become Identity KLines (S1 | token, nodes=[token])
    - Multi-char identifiers become Compound KLines (S1 | S2 | all_tokens)
    - Operators call agent.signify() to establish relationships

    Args:
        source: KScript source code
        agent: Optional KAgent instance. If not provided,
               creates a new Kalvin agent with default settings.

    Returns:
        InterpretResult with model, symbol_table, load_paths, save_path
    """
    from kalvin import Kalvin as KalvinClass

    agent_instance = agent if agent is not None else KalvinClass()
    script = parse(source)
    interpreter = Interpreter(agent=agent_instance)
    return interpreter.interpret(script)
