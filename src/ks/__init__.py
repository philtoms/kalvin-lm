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

Spec ref: @specs/kscript.md §13 (Public API).
"""

from __future__ import annotations

from kalvin.abstract import KTokenizer
from kalvin.kline import KLine
from kalvin.nlp_tokenizer import NLPTokenizer

from .ast_emitter import SymbolicEntry
from .compiler import Compiler, compile_source

__all__ = [
    "KScript",
    "KLine",
    "Compiler",
    "compile_source",
    "SymbolicEntry",
]


class KScript:
    """One-shot compilation of a KScript source string.

    Compiles the source immediately upon construction using the full
    pipeline (Lexer → Parser → Compiler).

    Args:
        source: KScript source code string.
        tokenizer: Tokenizer for encoding (default: NLPTokenizer; tokenizer data is mandatory).
        dev: Enable development/diagnostic mode.

    Example::

        model = KScript("A == B")
        print(model.entries)  # list[KLine]
    """

    def __init__(
        self,
        source: str,
        tokenizer: KTokenizer | None = None,
        dev: bool = False,
    ) -> None:
        self._tokenizer: KTokenizer = tokenizer or NLPTokenizer()
        self._dev = dev
        self._entries: list[KLine] = compile_source(
            source,
            tokenizer=self._tokenizer,
            dev=self._dev,
        )

    @property
    def entries(self) -> list[KLine]:
        """Return the compiled entries."""
        return self._entries
