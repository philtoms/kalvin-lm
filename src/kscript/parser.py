"""Recursive descent parser for KScript language.

Grammar:
    script ::= construct+
    construct ::=
      | sig                              -- identity
      | sig == node                      -- countersign
      | sig > node                       -- connotate fwd
      | sig = node                       -- undersign
      | sig => construct                 -- canonize fwd (right-assoc)
      | construct <= construct           -- canonize bwd
      | construct < construct            -- connotate bwd
      | construct construct*             -- sequence

Key insight: BWD operators bind CONSTRUCTS, not nodes.
The signature after BWD becomes the owner of a new construct.

Another key insight: Only SIGNATURE tokens can be construct owners.
LITERAL tokens can only appear in node positions.
"""

from .ast import (
    Construct,
    ConstructType,
    KScriptFile,
    Literal,
    Node,
    Script,
    Signature,
)
from .token import Token, TokenType


class ParseError(Exception):
    """Error during parsing."""

    def __init__(self, message: str, token: Token):
        super().__init__(f"Line {token.line}, column {token.column}: {message}")
        self.token = token


class Parser:
    """Recursive descent parser for KScript with construct-level BWD binding.

    Handles:
    - Multiple top-level scripts (column 1 signatures)
    - BWD operators binding constructs (signature on RIGHT becomes owner)
    - Subscript normalization (INDENT/DEDENT → inline sequence)
    - Script boundary detection at column 1
    - Error recovery (literals at start, empty constructs)

    Key insight: _is_literal() = not _check(TokenType.SIGNATURE)
    """

    def __init__(self, tokens: list[Token]):
        """Initialize parser with token list.

        Args:
            tokens: List of tokens from lexer (ending with EOF)
        """
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> KScriptFile:
        """Parse tokens into a KScriptFile AST.

        Returns:
            KScriptFile containing all top-level scripts
        """
        scripts: list[Script] = []

        while not self._at_end():
            # Skip newlines and comments at top level
            if self._check(TokenType.NEWLINE) or self._check(TokenType.COMMENT):
                self._advance()
                continue

            # Skip literals at top level until we find a signature
            # (error recovery for literal at script start)
            if self._is_literal():
                self._advance()
                continue

            # Parse top-level script (starts at column 1 with signature)
            if self._check(TokenType.SIGNATURE):
                script = self._parse_script()
                scripts.append(script)
            else:
                # Skip unexpected tokens at top level
                self._advance()

        return KScriptFile(scripts)

    def _parse_script(self) -> Script:
        """Parse a script starting with a signature.

        A script consists of constructs, where each construct has an owner.
        The primary signature is the first owner.

        Returns:
            Script AST node with all constructs
        """
        start_token = self._expect(TokenType.SIGNATURE)
        primary_sig = self._make_signature(start_token)

        # Collect all constructs in this script
        constructs: list[Construct] = []

        # Current owner starts as primary signature
        current_owner = primary_sig

        # Track accumulated nodes for BWD binding
        accumulated_nodes: list[Node] = []

        # Track pending operator for subscript (when => has no inline nodes)
        pending_operator: ConstructType | None = None

        # Parse inline constructs until newline or end
        while not self._at_end() and not self._check(TokenType.NEWLINE):
            # Check for BWD operators - these signal NEW owner
            if self._check(TokenType.CANONIZE_BWD) or self._check(TokenType.CONNOTATE_BWD):
                bwd_type = ConstructType.CANONIZE_BWD if self._check(TokenType.CANONIZE_BWD) else ConstructType.CONNOTATE_BWD
                self._advance()  # consume operator

                # Next SIGNATURE becomes the NEW owner
                if self._check(TokenType.SIGNATURE):
                    new_owner_token = self._advance()
                    new_owner = self._make_signature(new_owner_token)

                    # Determine target nodes based on BWD type
                    # If no accumulated nodes, use current_owner as target
                    if bwd_type == ConstructType.CANONIZE_BWD:
                        # <= binds ALL accumulated nodes (or current owner if empty)
                        target_nodes: list[Node] = list(accumulated_nodes) if accumulated_nodes else [current_owner]
                    else:
                        # < binds only CLOSEST (last) node (or current owner if empty)
                        target_nodes = [accumulated_nodes[-1]] if accumulated_nodes else [current_owner]

                    # Create BWD construct with new owner
                    if target_nodes:
                        constructs.append(Construct(
                            owner=new_owner,
                            type=bwd_type,
                            nodes=target_nodes,
                            line=new_owner.line
                        ))

                    # New owner becomes current
                    current_owner = new_owner
                    accumulated_nodes = []

                    # Continue parsing - new owner may have FWD constructs
                    continue
                else:
                    # BWD without following signature - error recovery, skip
                    break

            # Check for FWD operators
            op_type = self._try_parse_operator()
            if op_type:
                # Parse nodes for this construct
                nodes = self._parse_nodes(multi=(op_type == ConstructType.CANONIZE_FWD))

                if nodes:
                    # Create construct with current owner
                    constructs.append(Construct(
                        owner=current_owner,
                        type=op_type,
                        nodes=nodes,
                        line=current_owner.line
                    ))

                    # Accumulate nodes for potential BWD
                    accumulated_nodes = nodes

                    # For => operator, the last node (if signature) becomes potential new owner
                    if op_type == ConstructType.CANONIZE_FWD and nodes:
                        last_node = nodes[-1]
                        if isinstance(last_node, Signature):
                            current_owner = last_node
                    # Clear any pending operator since we handled it inline
                    pending_operator = None
                else:
                    # No inline nodes - track pending operator for subscript
                    pending_operator = op_type

                continue

            # Check for sequence (signature without operator starts new construct)
            if self._check(TokenType.SIGNATURE):
                sig_token = self._advance()
                sig = self._make_signature(sig_token)
                # This signature becomes the new current owner (sequence)
                current_owner = sig
                accumulated_nodes = [sig]
                continue

            # Check for literal in node position (add to accumulated nodes)
            if self._is_literal():
                node = self._parse_literal_node()
                if node:
                    accumulated_nodes.append(node)
                continue

            # Nothing recognized - break
            break

        # Consume newline if present
        if self._check(TokenType.NEWLINE):
            self._advance()

        # Parse subscripts and normalize to inline constructs
        self._parse_subscripts(constructs, current_owner, accumulated_nodes, pending_operator)

        return Script(primary_sig, constructs, primary_sig.line)

    def _parse_subscripts(
        self,
        constructs: list[Construct],
        current_owner: Signature,
        accumulated_nodes: list[Node],
        pending_operator: ConstructType | None = None
    ) -> None:
        """Parse subscripts and append constructs inline.

        Subscripts are layout sugar - they're normalized to the same
        construct list as inline constructs.

        Args:
            constructs: List to append new constructs to
            current_owner: Current owner signature (modified in place)
            accumulated_nodes: Current accumulated nodes (modified in place)
            pending_operator: Operator waiting for subscript nodes (e.g., => with no inline nodes)
        """
        while self._check(TokenType.INDENT):
            self._advance()  # consume INDENT

            # If we have a pending operator, collect subscript content as nodes
            if pending_operator:
                subscript_nodes: list[Node] = []

                # Collect nodes until DEDENT
                while (
                    not self._at_end()
                    and not self._check(TokenType.DEDENT)
                    and not self._check(TokenType.EOF)
                ):
                    # Skip newlines and comments
                    if self._check(TokenType.NEWLINE) or self._check(TokenType.COMMENT):
                        self._advance()
                        continue

                    # Collect signatures and literals as nodes
                    if self._check(TokenType.SIGNATURE):
                        sig_token = self._advance()
                        subscript_nodes.append(self._make_signature(sig_token))
                    elif self._is_literal():
                        node = self._parse_literal_node()
                        if node:
                            subscript_nodes.append(node)
                    else:
                        # Skip unexpected tokens
                        self._advance()

                # Create the pending construct with collected nodes
                if subscript_nodes:
                    constructs.append(Construct(
                        owner=current_owner,
                        type=pending_operator,
                        nodes=subscript_nodes,
                        line=current_owner.line
                    ))
                    accumulated_nodes = subscript_nodes

                    # For => operator, last signature becomes potential new owner
                    if pending_operator == ConstructType.CANONIZE_FWD:
                        last_node = subscript_nodes[-1]
                        if isinstance(last_node, Signature):
                            current_owner = last_node

                # Consume DEDENT
                if self._check(TokenType.DEDENT):
                    self._advance()

                # Pending operator handled, clear it
                pending_operator = None
                continue

            # Normal subscript parsing (no pending operator)
            # Parse subscript content
            while (
                not self._at_end()
                and not self._check(TokenType.DEDENT)
                and not self._check(TokenType.EOF)
            ):
                # Skip newlines and comments
                if self._check(TokenType.NEWLINE) or self._check(TokenType.COMMENT):
                    self._advance()
                    continue

                # Check for BWD operators in subscript
                if self._check(TokenType.CANONIZE_BWD) or self._check(TokenType.CONNOTATE_BWD):
                    bwd_type = ConstructType.CANONIZE_BWD if self._check(TokenType.CANONIZE_BWD) else ConstructType.CONNOTATE_BWD
                    self._advance()

                    if self._check(TokenType.SIGNATURE):
                        new_owner_token = self._advance()
                        new_owner = self._make_signature(new_owner_token)

                        if bwd_type == ConstructType.CANONIZE_BWD:
                            target_nodes = list(accumulated_nodes)
                        else:
                            target_nodes = [accumulated_nodes[-1]] if accumulated_nodes else []

                        if target_nodes:
                            constructs.append(Construct(
                                owner=new_owner,
                                type=bwd_type,
                                nodes=target_nodes,
                                line=new_owner.line
                            ))

                        current_owner = new_owner
                        accumulated_nodes = []
                    continue

                # Check for signature - sequence or construct start
                if self._check(TokenType.SIGNATURE):
                    sig_token = self._advance()
                    sig = self._make_signature(sig_token)

                    # Check if followed by operator
                    op_type = self._try_parse_operator()
                    if op_type:
                        nodes = self._parse_nodes(multi=(op_type == ConstructType.CANONIZE_FWD))
                        if nodes:
                            constructs.append(Construct(
                                owner=current_owner,
                                type=op_type,
                                nodes=nodes,
                                line=current_owner.line
                            ))
                            accumulated_nodes = nodes
                            if op_type == ConstructType.CANONIZE_FWD and nodes:
                                last_node = nodes[-1]
                                if isinstance(last_node, Signature):
                                    current_owner = last_node
                    else:
                        # Sequence - signature becomes new owner
                        current_owner = sig
                        accumulated_nodes = [sig]
                    continue

                # Check for literal node
                if self._is_literal():
                    node = self._parse_literal_node()
                    if node:
                        accumulated_nodes.append(node)
                    continue

                # Skip unexpected tokens
                self._advance()

            # Consume DEDENT
            if self._check(TokenType.DEDENT):
                self._advance()

    def _try_parse_operator(self) -> ConstructType | None:
        """Try to parse a FWD construct operator. Returns None if not found."""
        if self._check(TokenType.COUNTERSIGN):
            self._advance()
            return ConstructType.COUNTERSIGN

        if self._check(TokenType.CANONIZE_FWD):
            self._advance()
            return ConstructType.CANONIZE_FWD

        if self._check(TokenType.CONNOTATE_FWD):
            self._advance()
            return ConstructType.CONNOTATE_FWD

        if self._check(TokenType.UNDERSIGN):
            self._advance()
            return ConstructType.UNDERSIGN

        return None

    def _parse_nodes(self, multi: bool) -> list[Node]:
        """Parse one or more nodes.

        Args:
            multi: If True, parse multiple nodes; if False, parse single node
        """
        nodes: list[Node] = []

        while not self._at_end():
            node = self._try_parse_node()
            if node:
                nodes.append(node)
                if not multi:
                    break
            else:
                break

        return nodes

    def _try_parse_node(self) -> Node | None:
        """Try to parse a single node (signature or literal)."""
        if self._check(TokenType.SIGNATURE):
            token = self._advance()
            return self._make_signature(token)

        if self._is_literal():
            return self._parse_literal_node()

        return None

    def _parse_literal_node(self) -> Literal | None:
        """Parse a literal token as a Literal node."""
        if self._check(TokenType.LITERAL):
            token = self._advance()
            return Literal(token.value, token.line, token.column)
        return None

    def _is_literal(self) -> bool:
        """Check if current token is a literal (not a signature).

        Key insight: Any token in node position that is NOT a SIGNATURE is a LITERAL.
        """
        return self._check(TokenType.LITERAL)

    def _make_signature(self, token: Token) -> Signature:
        """Create a Signature from a token."""
        return Signature(token.value, None, token.line, token.column)

    def _peek(self) -> Token:
        """Return current token without advancing."""
        return self.tokens[self.pos]

    def _check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type."""
        if self._at_end():
            return False
        return self._peek().type == token_type

    def _advance(self) -> Token:
        """Advance and return current token."""
        if not self._at_end():
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return self.tokens[-1]  # EOF

    def _expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type, raise error if not found."""
        if self._check(token_type):
            return self._advance()
        raise ParseError(f"Expected {token_type.name}", self._peek())

    def _at_end(self) -> bool:
        """Check if at end of tokens."""
        return self._peek().type == TokenType.EOF
