"""KScript v3 package — public API.

Provides the ``KScript`` class for one-shot compilation of KScript source
strings, plus re-exports of key pipeline types.

Usage::

    from ks import KScript

    model = KScript("A == B")
    for entry in model.entries:
        print(entry)

The v3 API is intentionally minimal: no file I/O, no ``base`` parameter,
no ``output()`` method, no ``to_jsonl()``.
"""

from __future__ import annotations

from collections.abc import Callable

from kalvin.abstract import KSignifier, KTokenizer
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.signifier import NLPSignifier

from .ast_emitter import SymbolicEntry
from .compiler import Compiler, CompileError, compile_source, path_resolver

__all__ = [
    "KScript",
    "KLine",
    "KValue",
    "Compiler",
    "CompileError",
    "compile_source",
    "path_resolver",
    "SymbolicEntry",
]


class KScript:
    """One-shot compilation of a KScript source string.

    Compiles the source immediately upon construction using the full
    pipeline (Lexer → Parser → Compiler).

    Args:
        source: KScript source code string.
        tokenizer: Tokenizer for encoding (default: BPETokenizer; tokenizer data is mandatory).
        dev: Enable development/diagnostic mode.
        resolver: Module name → source, for ``import`` statements.

    Example::

        model = KScript("A == B")
        print(model.entries)  # list[KValue]
    """

    def __init__(
        self,
        source: str,
        tokenizer: KTokenizer | None = None,
        signifier: KSignifier | None = None,
        dev: bool = False,
        resolver: Callable[[str], str] | None = None,
    ) -> None:
        self._tokenizer: KTokenizer = tokenizer or BPETokenizer()
        self._signifier: KSignifier = signifier or NLPSignifier()
        self._dev = dev
        self._entries: list[KValue] = compile_source(
            source,
            tokenizer=self._tokenizer,
            signifier=self._signifier,
            dev=self._dev,
            resolver=resolver,
        )

    @property
    def entries(self) -> list[KValue]:
        """Return the compiled entries."""
        return self._entries
