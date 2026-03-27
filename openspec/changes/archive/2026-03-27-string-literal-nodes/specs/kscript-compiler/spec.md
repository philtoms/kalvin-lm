## MODIFIED Requirements

### Requirement: Token Types
The system SHALL define the following token types:

| Type | Pattern | Description |
|------|---------|-------------|
| COUNTERSIGN | `==` | Bidirectional link |
| CANONIZE_FWD | `=>` | Forward canonization |
| CANONIZE_BWD | `<=` | Backward canonization |
| CONNOTATE_FWD | `>` | Forward connotation |
| CONNOTATE_BWD | `<` | Backward connotation |
| UNDERSIGN | `=` | Undersign link |
| SIGNATURE | `[A-Z]+` | Uppercase-only identifier |
| STRING_LITERAL | `[a-zA-Z0-9]+` (not all uppercase) | Unquoted string literal |
| STRING | `"..."` | Double-quoted string |
| NUMBER | `[0-9]+` | Numeric literal |
| COMMENT | `(...)` | Parenthesized comment |
| NEWLINE | `\n` | Line ending |
| INDENT | - | Increased indentation |
| DEDENT | - | Decreased indentation |
| EOF | - | End of file |

#### Scenario: Token dataclass
- **WHEN** a token is created
- **THEN** it has `type` (TokenType), `value` (str), `line` (int), `column` (int)

---

### Requirement: Lexer - Signatures
The lexer SHALL tokenize uppercase-only letter sequences as SIGNATURE tokens.

#### Scenario: Single-letter signature
- **WHEN** lexer encounters `A`
- **THEN** it produces SIGNATURE token with value `"A"`

#### Scenario: Multi-letter signature
- **WHEN** lexer encounters `MHALL`
- **THEN** it produces SIGNATURE token with value `"MHALL"`

#### Scenario: Signature with inline comment
- **WHEN** lexer encounters `S(ubject)`
- **THEN** it produces SIGNATURE token with value `"S"`
- **AND** the comment `(ubject)` is consumed but not attached

---

## ADDED Requirements

### Requirement: Lexer - Unquoted String Literals
The lexer SHALL tokenize identifiers containing lowercase letters, digits, or mixed case as STRING_LITERAL tokens.

#### Scenario: Lowercase identifier
- **WHEN** lexer encounters `zed`
- **THEN** it produces STRING_LITERAL token with value `"zed"`

#### Scenario: Mixed case identifier
- **WHEN** lexer encounters `Hello`
- **THEN** it produces STRING_LITERAL token with value `"Hello"`

#### Scenario: Alphanumeric identifier
- **WHEN** lexer encounters `item123`
- **THEN** it produces STRING_LITERAL token with value `"item123"`

#### Scenario: Uppercase identifier remains signature
- **WHEN** lexer encounters `FOO`
- **THEN** it produces SIGNATURE token with value `"FOO"` (not STRING_LITERAL)

---

### Requirement: Parser - STRING_LITERAL as Node
The parser SHALL accept STRING_LITERAL tokens as nodes in any construct position.

#### Scenario: String literal as node
- **WHEN** parser processes `X => Y zed`
- **THEN** it creates a construct with nodes `[Signature("Y"), StringLiteral("zed")]`

#### Scenario: Multiple string literals
- **WHEN** parser processes `A => foo bar BAZ`
- **THEN** it creates a construct with nodes `[StringLiteral("foo"), StringLiteral("bar"), Signature("BAZ")]`

#### Scenario: String literal in countersign
- **WHEN** parser processes `A == hello`
- **THEN** it creates a countersign construct with node `StringLiteral("hello")`
