"""Interpreter for KScript AST with new identity/compound KLine semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kalvin.model import KLine, Model
from kalvin.significance import S1, S2, S3, S4

from .ast import (
    Identifier,
    KLineExpr,
    KScriptAst,
    KNodeRef,
    LoadStatement,
    SaveStatement,
    SignificanceType,
)

from .tokens import (
    encode_mod,
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

    model: Model
    symbol_table: SymbolTable
    load_paths: list[str] = field(default_factory=list)
    save_path: str | None = None


class Interpreter:
    """
    Interprets KScript AST with identity/compound KLine semantics.

    - Single-char identifiers become Identity KLines (S1 | token, nodes=[token])
    - Multi-char identifiers become Compound KLines (S1 | S2 | all_tokens, nodes=identity_sigs)
    - Operators call agent.signify() to establish relationships

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

    def interpret(self, ast: KScriptAst) -> InterpretResult:
        """Interpret a KScript AST."""
        symbol_table: SymbolTable = {}
        load_paths: list[str] = []
        save_path: str | None = None

        for stmt in ast.statements:
            if isinstance(stmt, LoadStatement):
                load_paths.append(stmt.path.name)
            elif isinstance(stmt, SaveStatement):
                save_path = stmt.path.name if stmt.path else None
            elif isinstance(stmt, KLineExpr):
                self._interpret_kline_expr(stmt, symbol_table)

        return InterpretResult(
            model=self.model,
            symbol_table=symbol_table,
            load_paths=load_paths,
            save_path=save_path,
        )

    def create_kline(self, name: str) -> KLine:
        """
        Create an identity or compound KLine for an identifier.

        Identity KLine (single char like 'M'):
            signature = S1_BIT | identity_sig(M)
            nodes = [token(M)]

        Compound KLine (multi-char like 'ALL'):
            signature = S1_BIT | S2_BITS | identity_sig(A) | identity_sig(L) | identity_sig(L)
            nodes = [identity_sig(A), identity_sig(L), identity_sig(L)]

        Args:
            name: Identifier name (1+ characters)

        Returns:
            KLine with appropriate signature and nodes
        """
        # Tokenize each character
        tokens = []
        sigs = []
        for char in name:
            sigs.append(encode_mod(char))
            encoded = self.tokenizer.encode(char)
            tokens.append(encoded[0])

        signature = S1
        nodes = []
        # First ensure identity klines exist for each char
        for idx, token in enumerate(tokens):
            identity_sig = S1 | sigs[idx]
            identity_kline = KLine(signature=identity_sig, nodes=[token])
            self.model.add(identity_kline)

            signature |= identity_sig
            nodes.append(identity_sig)

        if len(name) > 1:
            signature |= S2

        # Build compound signature: S1 | S2 | all sig tokens
        kline = KLine(signature=signature, nodes=nodes)
        self.model.add(kline)
        return kline

    def _interpret_kline_expr(
        self,
        expr: KLineExpr,
        symbol_table: SymbolTable,
    ) -> KLine:
        """
        Interpret a KLine expression with new semantics.

        Creates the sig KLine (identity or compound) and establishes
        relationships via agent.signify().
        """
        # Step 1: Create the sig KLine
        if isinstance(expr.sig, Identifier):
            name = expr.sig.name
            sig_kline = self.create_kline(name)
            symbol_table[name] = sig_kline.signature
        else:
            # Nested KLineExpr as sig - interpret recursively
            sig_kline = self._interpret_kline_expr(expr.sig, symbol_table)

        # Step 2: Process relationships using signify()
        # Relationships chain: MHALL = SVO => [S, V, O] means:
        #   MHALL -> SVO (S1)
        #   SVO -> [S, V, O] (S2)
        current_subject = sig_kline

        for relationship in expr.relationships:
            # Get significance level for this relationship
            S = self._significance_type_to_value(relationship.significance)

            # Interpret nodes and create their KLines
            node_klines = []
            for node_ref in relationship.nodes:
                if node_ref.nested_kline:
                    # Nested KLine expression
                    node_kline = self._interpret_kline_expr(
                        node_ref.nested_kline, symbol_table
                    )
                else:
                    # Simple identifier
                    node_name = node_ref.identifier.name
                    node_kline = self.create_kline(node_name)
                    symbol_table[node_name] = node_kline.signature

                node_klines.append(node_kline)

            # Establish significance relationships
            for node_kline in node_klines:
                self.agent.signify(current_subject, node_kline, S)

            # Chain: last node becomes subject for next relationship
            if node_klines:
                current_subject = node_klines[-1]

        return sig_kline

    def _significance_type_to_value(self, sig_type: SignificanceType | None) -> int:
        """Convert SignificanceType enum to significance value."""
        if sig_type is None:
            return 0

        if sig_type == SignificanceType.S1:
            return S1
        elif sig_type == SignificanceType.S2:
            return S2
        elif sig_type in (SignificanceType.S3_FORWARD, SignificanceType.S3_BACKWARD):
            return S3
        elif sig_type == SignificanceType.S4:
            return S4

        return 0
