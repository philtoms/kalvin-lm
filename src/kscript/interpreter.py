"""Interpreter for KScript AST with new identity/compound KLine semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kalvin.model import KLine
from kalvin.significance import S1_BIT, S4_VALUE, build_s1, build_s2, build_s3

from .ast import (
    Identifier,
    KLineExpr,
    KScript,
    KNodeRef,
    LoadStatement,
    SaveStatement,
    SignificanceType,
)

if TYPE_CHECKING:
    from kalvin.agent import KAgent

# Type for the symbol table (maps names to signatures)
SymbolTable = dict[str, int]


class InterpretError(Exception):
    """Interpreter error with context."""

    pass


@dataclass
class InterpretResult:
    """Result of interpretation."""

    model: "Model"
    symbol_table: SymbolTable
    load_paths: list[str] = field(default_factory=list)
    save_path: str | None = None
    attention_klines: list[int] = field(default_factory=list)


class Interpreter:
    """
    Interprets KScript AST with identity/compound KLine semantics.

    - Single-char identifiers become Identity KLines (S1 | token, nodes=[token])
    - Multi-char identifiers become Compound KLines (S1 | S2 | all_tokens, nodes=identity_sigs)
    - Operators call agent.signify() to establish relationships
    - Attention (?) is a yield point that processes what has been seen so far

    Requires a KAgent instance for tokenization, model storage, and signify().
    """

    def __init__(self, agent: KAgent):
        """
        Initialize interpreter with a KAgent.

        Args:
            agent: KAgent instance providing tokenizer, model, and signify().
        """
        self._agent = agent
        self.tokenizer = agent.tokenizer
        self.model = agent.model

    @property
    def agent(self) -> KAgent:
        """Get the KAgent."""
        return self._agent

    def interpret(self, script: KScript) -> InterpretResult:
        """Interpret a KScript AST."""
        symbol_table: SymbolTable = {}
        load_paths: list[str] = []
        save_path: str | None = None
        attention_points: list[int] = []  # Signatures at each attention/yield point

        for stmt in script.statements:
            if isinstance(stmt, LoadStatement):
                load_paths.append(stmt.path.name)
            elif isinstance(stmt, SaveStatement):
                save_path = stmt.path.name if stmt.path else None
            elif isinstance(stmt, KLineExpr):
                self._interpret_kline_expr(stmt, symbol_table, attention_points)

        return InterpretResult(
            model=self.model,
            symbol_table=symbol_table,
            load_paths=load_paths,
            save_path=save_path,
            attention_klines=attention_points,
        )

    def create_kline(self, name: str) -> KLine:
        """
        Create an identity or compound KLine for an identifier.

        Identity KLine (single char like 'M'):
            signature = S1_BIT | token(M)
            nodes = [token(M)]

        Compound KLine (multi-char like 'ALL'):
            signature = S1_BIT | S2_BITS | token(A) | token(L) | token(L)
            nodes = [identity_sig(A), identity_sig(L), identity_sig(L)]

        Args:
            name: Identifier name (1+ characters)

        Returns:
            KLine with appropriate signature and nodes
        """
        # Tokenize each character
        tokens = []
        for char in name:
            encoded = self.tokenizer.encode(char)
            tokens.append(encoded[0] if encoded else hash(char) & 0xFFFF)

        if len(name) == 1:
            # Identity KLine: S1 | token, nodes = [token]
            token = tokens[0]
            signature = S1_BIT | token
            nodes = [token]
        else:
            # Compound KLine: S1 | S2 | all tokens
            # First ensure identity klines exist for each char
            for token in tokens:
                identity_sig = S1_BIT | token
                identity_kline = KLine(signature=identity_sig, nodes=[token])
                self.model.add(identity_kline)

            # Build compound signature: S1 | S2 | all tokens
            signature = S1_BIT | build_s2(100, 100)
            for token in tokens:
                signature |= token

            # Nodes are identity signatures for each character
            nodes = [S1_BIT | t for t in tokens]

        kline = KLine(signature=signature, nodes=nodes)
        self.model.add(kline)
        return kline

    def _interpret_kline_expr(
        self,
        expr: KLineExpr,
        symbol_table: SymbolTable,
        attention_points: list[int],
    ) -> KLine | None:
        """
        Interpret a KLine expression with new semantics.

        Creates the sig KLine (identity or compound) and establishes
        relationships via agent.signify().

        Attention (?) is a yield point:
        - `A? => B` - attend to A first, then process relationship to B
        - `A? => B?` - attend to A, then attend to B (two yield points)
        """
        # Step 1: Create the sig KLine
        if isinstance(expr.sig, Identifier):
            name = expr.sig.name
            # Check if already exists in symbol table
            if name in symbol_table:
                existing_sig = symbol_table[name]
                sig_kline = self.model.find_by_key(existing_sig)
                if sig_kline is None:
                    sig_kline = self.create_kline(name)
            else:
                sig_kline = self.create_kline(name)
                symbol_table[name] = sig_kline.signature
        else:
            # Nested KLineExpr as sig - interpret recursively
            sig_kline = self._interpret_kline_expr(expr.sig, symbol_table, attention_points)
            if sig_kline is None:
                return None

        # Step 2: If attention marker on sig, yield/attend before relationships
        if expr.attention:
            attention_points.append(sig_kline.signature)
            # This is a yield point - model has been updated, Kalvin can query

        # Step 3: Process relationships using signify()
        for relationship in expr.relationships:
            # Get significance level for this relationship
            s = self._significance_type_to_value(relationship.significance)

            # Interpret nodes and create their KLines
            node_klines = []
            for node_ref in relationship.nodes:
                if node_ref.nested_kline:
                    # Nested KLine expression - may have its own attention
                    node_kline = self._interpret_kline_expr(
                        node_ref.nested_kline, symbol_table, attention_points
                    )
                else:
                    # Simple identifier
                    node_name = node_ref.identifier.name
                    if node_name in symbol_table:
                        existing_sig = symbol_table[node_name]
                        node_kline = self.model.find_by_key(existing_sig)
                        if node_kline is None:
                            node_kline = self.create_kline(node_name)
                    else:
                        node_kline = self.create_kline(node_name)
                        symbol_table[node_name] = node_kline.signature

                if node_kline:
                    node_klines.append(node_kline)

            # Establish significance relationships
            for node_kline in node_klines:
                self.agent.signify(sig_kline, node_kline, s)

        return sig_kline

    def _significance_type_to_value(self, sig_type: SignificanceType | None) -> int:
        """Convert SignificanceType enum to significance value."""
        if sig_type is None:
            return 0

        if sig_type == SignificanceType.S1:
            return build_s1(100)
        elif sig_type == SignificanceType.S2:
            return build_s2(100, 100)
        elif sig_type in (SignificanceType.S3_FORWARD, SignificanceType.S3_BACKWARD):
            return build_s3(100, 100, 100)
        elif sig_type == SignificanceType.S4:
            return S4_VALUE

        return 0
