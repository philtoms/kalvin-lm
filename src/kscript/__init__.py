"""KScript parser and compiler for Kalvin models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ast import Identifier, KLineExpr, KNodeRef, KScript, SignificanceType
from .compiler import CompileResult, Compiler
from .interpreter import InterpretError, InterpretResult, Interpreter
from .lexer import Lexer, LexerError
from .parser import ParseError, Parser
from .tokens import Token, TokenType

if TYPE_CHECKING:
    from kalvin.agent import KAgent

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
    "compile_script",
    "interpret_script",
]


def parse(source: str) -> KScript:
    """Parse KScript source to AST."""
    return Parser.from_source(source).parse()


def compile_script(source: str, agent: KAgent | None = None) -> CompileResult:
    """Parse and compile KScript source using a KAgent.

    Args:
        source: KScript source code
        agent: Optional KAgent instance. If not provided,
               creates a new Kalvin agent with default settings.

    Returns:
        CompileResult with model, symbol_table, load_paths, save_path
    """
    from kalvin import Kalvin as KalvinClass

    agent_instance = agent if agent is not None else KalvinClass()
    script = parse(source)
    compiler = Compiler(agent=agent_instance)
    return compiler.compile(script)


def interpret_script(source: str, agent: KAgent | None = None) -> InterpretResult:
    """Parse and interpret KScript source using new identity/compound semantics.

    Key differences from compile_script():
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
