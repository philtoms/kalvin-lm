"""Compiler for KScript AST to Kalvin Model."""

from dataclasses import dataclass, field

from kalvin.model import KLine, Model
from kalvin.significance import S4_VALUE, build_s1, build_s2, build_s3

from .ast import (
    Identifier,
    KLineExpr,
    KScript,
    KNodeRef,
    LoadStatement,
    SaveStatement,
    SignificanceType,
)

# Type for the symbol table (maps names to s_keys)
SymbolTable = dict[str, int]


class CompileError(Exception):
    """Compiler error with context."""

    pass


@dataclass
class CompileResult:
    """Result of compilation."""

    model: Model
    symbol_table: SymbolTable
    load_paths: list[str] = field(default_factory=list)
    save_path: str | None = None
    attention_klines: list[int] = field(default_factory=list)  # s_keys with attention


class Compiler:
    """
    Compiles KScript AST to Kalvin Model with KLines.

    Two-phase compilation:
    1. First pass: Collect all KLine definitions and assign token IDs
    2. Second pass: Build KLines with resolved references
    """

    def __init__(self, tokenizer=None):
        """
        Initialize compiler.

        Args:
            tokenizer: Optional tokenizer for string-to-token conversion.
                      If None, identifiers are hashed to create token IDs.
        """
        self.tokenizer = tokenizer

    def compile(self, script: KScript) -> CompileResult:
        """Compile a KScript AST to a Model."""
        model = Model()
        symbol_table: SymbolTable = {}
        load_paths: list[str] = []
        save_path: str | None = None
        attention_klines: list[int] = []

        # First pass: collect all identifiers and assign s_keys
        token_map = self._collect_identifiers(script)

        # Second pass: compile statements
        for stmt in script.statements:
            if isinstance(stmt, LoadStatement):
                load_paths.append(stmt.path.name)
            elif isinstance(stmt, SaveStatement):
                save_path = stmt.path.name if stmt.path else None
            elif isinstance(stmt, KLineExpr):
                kline = self._compile_kline_expr(stmt, token_map, symbol_table, model)
                if kline:
                    if stmt.attention:
                        attention_klines.append(kline.s_key)

        return CompileResult(
            model=model,
            symbol_table=symbol_table,
            load_paths=load_paths,
            save_path=save_path,
            attention_klines=attention_klines,
        )

    def _collect_identifiers(self, script: KScript) -> dict[str, int]:
        """
        First pass: collect all identifiers and assign unique token IDs.

        This ensures forward references work correctly.
        """
        token_map: dict[str, int] = {}
        next_token = 1  # Start from 1, reserve 0 for special use

        for stmt in script.statements:
            if isinstance(stmt, KLineExpr):
                # Add the KSig identifier
                if isinstance(stmt.sig, Identifier):
                    name = stmt.sig.name
                    if name not in token_map:
                        token_map[name] = self._name_to_token(name, next_token)
                        next_token += 1

                # Add all node identifiers
                for node_ref in stmt.nodes:
                    name = node_ref.identifier.name
                    if name not in token_map:
                        token_map[name] = self._name_to_token(name, next_token)
                        next_token += 1

        return token_map

    def _name_to_token(self, name: str, fallback: int) -> int:
        """
        Convert a name to a token ID.

        If a tokenizer is available, use it. Otherwise, use a stable hash.
        """
        if self.tokenizer:
            tokens = self.tokenizer.encode(name)
            return tokens[0] if tokens else fallback
        else:
            # Use stable hash for deterministic token IDs
            # Mask to fit in lower bits (preserve upper bits for significance)
            return hash(name) & 0xFFFF

    def _compile_kline_expr(
        self,
        expr: KLineExpr,
        token_map: dict[str, int],
        symbol_table: SymbolTable,
        model: Model,
    ) -> KLine | None:
        """
        Compile a KLine expression to a KLine and add to model.

        Returns the created KLine, or None if duplicate.
        """
        # Get the token for this KLine's sig
        if isinstance(expr.sig, Identifier):
            name = expr.sig.name
            token = token_map.get(name, self._name_to_token(name, 0))
        else:
            # Nested KLineExpr - compile recursively
            nested = self._compile_kline_expr(expr.sig, token_map, symbol_table, model)
            if nested is None:
                return None
            token = nested.s_key

        # Build significance value
        sig_value = self._build_significance(expr.significance)

        # Build node list (convert identifiers to s_keys)
        nodes: list[int] = []
        for node_ref in expr.nodes:
            node_name = node_ref.identifier.name
            node_token = token_map.get(node_name, self._name_to_token(node_name, 0))
            # Each node is a simple KLine with token as s_key
            node_s_key = node_token  # Simple: just the token
            nodes.append(node_s_key)

        # Create KLine: s_key = significance | token
        s_key = sig_value | token
        kline = KLine(s_key=s_key, nodes=nodes)

        # Add to model
        if model.add(kline):
            # Record in symbol table
            if isinstance(expr.sig, Identifier):
                symbol_table[expr.sig.name] = s_key
            return kline

        return None

    def _build_significance(self, sig_type: SignificanceType | None) -> int:
        """Convert SignificanceType to the corresponding significance value."""
        if sig_type is None:
            return 0

        if sig_type == SignificanceType.S1:
            return build_s1(100)  # Full prefix match
        elif sig_type == SignificanceType.S2:
            return build_s2(100, 100)  # Full positional + non-positional
        elif sig_type in (SignificanceType.S3_FORWARD, SignificanceType.S3_BACKWARD):
            return build_s3(100, 100, 100)  # Full unordered
        elif sig_type == SignificanceType.S4:
            return S4_VALUE

        return 0
