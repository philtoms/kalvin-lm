"""Compiler for KScript AST to Kalvin Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kalvin.model import KLine, Model
from kalvin.significance import S4_VALUE, build_s1, build_s2, build_s3

from .ast import (
    Identifier,
    KLineExpr,
    KLineRelationship,
    KScript,
    KNodeRef,
    LoadStatement,
    SaveStatement,
    SignificanceType,
)

if TYPE_CHECKING:
    from kalvin.agent import KAgent

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

    Requires a KAgent instance for tokenization and model storage.
    """

    def __init__(self, agent: KAgent):
        """
        Initialize compiler with a KAgent.

        Args:
            agent: KAgent instance providing tokenizer and model.
        """
        self._agent = agent
        self.tokenizer = agent.tokenizer
        self.model = agent.model

    @property
    def agent(self) -> KAgent:
        """Get the KAgent."""
        return self._agent

    def compile(self, script: KScript) -> CompileResult:
        """Compile a KScript AST to a Model."""
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
                kline = self._compile_kline_expr(stmt, token_map, symbol_table)
                if kline:
                    if stmt.attention:
                        attention_klines.append(kline.signature)

        return CompileResult(
            model=self.model,
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
                self._collect_kline_identifiers(stmt, token_map, next_token)
                # Update next_token based on new entries
                next_token = len(token_map) + 1

        return token_map

    def _collect_kline_identifiers(
        self,
        expr: KLineExpr,
        token_map: dict[str, int],
        next_token: int,
    ) -> None:
        """Recursively collect identifiers from a KLine expression."""
        # Add the KSig identifier
        if isinstance(expr.sig, Identifier):
            name = expr.sig.name
            if name not in token_map:
                token_map[name] = self._name_to_token(name, next_token + len(token_map))

        # Add all node identifiers from all relationships (including nested KLines)
        for relationship in expr.relationships:
            for node_ref in relationship.nodes:
                name = node_ref.identifier.name
                if name and name not in token_map:
                    token_map[name] = self._name_to_token(name, next_token + len(token_map))

                # Recursively collect from nested KLines
                if node_ref.nested_kline:
                    self._collect_kline_identifiers(node_ref.nested_kline, token_map, next_token)

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
    ) -> KLine | None:
        """
        Compile a KLine expression to KLine(s) and add to model.

        For KLines with multiple relationships, creates one KLine per relationship.
        Returns the first created KLine, or None if all are duplicates.
        """
        # Get the token for this KLine's sig
        if isinstance(expr.sig, Identifier):
            name = expr.sig.name
            token = token_map.get(name, self._name_to_token(name, 0))
        else:
            # Nested KLineExpr as sig - compile recursively
            nested = self._compile_kline_expr(expr.sig, token_map, symbol_table)
            if nested is None:
                return None
            token = nested.signature

        first_kline: KLine | None = None

        # Compile each relationship as a separate KLine
        for relationship in expr.relationships:
            # Build significance value
            sig_value = self._build_significance(relationship.significance)

            # Build node list (convert identifiers to s_keys)
            nodes: list[int] = []
            for node_ref in relationship.nodes:
                # If there's a nested KLine, compile it first
                if node_ref.nested_kline:
                    nested_kline = self._compile_kline_expr(
                        node_ref.nested_kline, token_map, symbol_table
                    )
                    if nested_kline:
                        nodes.append(nested_kline.signature)
                else:
                    # Simple node reference
                    node_name = node_ref.identifier.name
                    node_token = token_map.get(node_name, self._name_to_token(node_name, 0))
                    # Each node is a simple KLine with token as signature
                    node_s_key = node_token  # Simple: just the token
                    nodes.append(node_s_key)

            # Create KLine: signature = significance | token
            signature = sig_value | token
            kline = KLine(signature=signature, nodes=nodes)

            # Add to model
            if self.model.add(kline):
                # Record in symbol table (use first relationship's signature)
                if first_kline is None:
                    first_kline = kline
                    if isinstance(expr.sig, Identifier):
                        symbol_table[expr.sig.name] = signature

        # If no relationships, create a simple KLine with just the sig
        if not expr.relationships:
            signature = token  # No significance, just the token
            kline = KLine(signature=signature, nodes=[])
            if self.model.add(kline):
                first_kline = kline
                if isinstance(expr.sig, Identifier):
                    symbol_table[expr.sig.name] = signature

        return first_kline

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
