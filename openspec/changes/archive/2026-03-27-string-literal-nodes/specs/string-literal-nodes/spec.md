## ADDED Requirements

### Requirement: Lexer recognizes unquoted string literals

The lexer SHALL tokenize unquoted ASCII identifiers containing lowercase letters, digits, or mixed case as `STRING_LITERAL` tokens.

#### Scenario: Lowercase identifier
- **WHEN** lexer processes `zed`
- **THEN** it returns a `STRING_LITERAL` token with value `"zed"`

#### Scenario: Mixed case identifier
- **WHEN** lexer processes `Hello`
- **THEN** it returns a `STRING_LITERAL` token with value `"Hello"`

#### Scenario: Alphanumeric identifier
- **WHEN** lexer processes `item123`
- **THEN** it returns a `STRING_LITERAL` token with value `"item123"`

#### Scenario: Uppercase identifier remains signature
- **WHEN** lexer processes `FOO`
- **THEN** it returns a `SIGNATURE` token with value `"FOO"`

### Requirement: Parser accepts STRING_LITERAL tokens as nodes

The parser SHALL accept `STRING_LITERAL` tokens in any position where a node is expected (after construct operators).

#### Scenario: String literal as node in construct
- **WHEN** parser processes `X => Y zed`
- **THEN** it creates a construct with signature `X`, operator `=>`, and nodes `[Signature(Y), StringLiteral("zed")]`

#### Scenario: Multiple string literals
- **WHEN** parser processes `A => foo bar BAZ`
- **THEN** it creates a construct with nodes `[StringLiteral("foo"), StringLiteral("bar"), Signature("BAZ")]`

### Requirement: Compiler encodes string literal nodes as unpacked character sequences

The compiler SHALL encode `StringLiteral` nodes from `STRING_LITERAL` tokens the same way as quoted strings - as unpacked character sequences (each char as `ord(c) << 1`).

#### Scenario: String literal compiles to unpacked chars
- **WHEN** compiler processes construct `A => zed` where `zed` is a StringLiteral
- **THEN** the nodes are encoded as `[ord('z')<<1, ord('e')<<1, ord('d')<<1]` = `[244, 202, 204]`

#### Scenario: String literal gets no reverse entry
- **WHEN** compiler processes countersign `A == zed` where `zed` is a StringLiteral
- **THEN** only one entry is created `{A: zed}` without reverse `{zed: A}`
