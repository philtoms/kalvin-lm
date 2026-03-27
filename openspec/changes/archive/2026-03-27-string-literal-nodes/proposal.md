## Why

KScript currently requires all nodes to be either uppercase signatures (`[A-Z]+`) or double-quoted strings. This forces verbose syntax for simple lowercase identifiers. Users should be able to write `X => Y zed` instead of `X => Y "zed"`, making scripts more readable and natural.

## What Changes

- Add support for **unquoted string literals**: ASCII sequences that are not exclusively uppercase and contain no whitespace
- Lexer distinguishes between:
  - `SIGNATURE`: Exclusively uppercase (`[A-Z]+`)
  - `STRING_LITERAL`: Contains lowercase letters, numbers, or mixed case (`[a-zA-Z0-9]+` with at least one non-uppercase)

**Examples:**

```
X => Y zed          # X, Y = signatures; zed = string literal
FOO => bar BAZ      # FOO, BAZ = signatures; bar = string literal
Hello => World123   # Hello, World123 = string literals (mixed case/digits)
```

## Capabilities

### New Capabilities

- `string-literal-nodes`: Lexer and parser support for unquoted string literals as node values

### Modified Capabilities

- `kscript-compiler`: Extends lexer token recognition to distinguish signatures from string literals

## Impact

- **Lexer** (`src/kscript/lexer.py`): Add `STRING_LITERAL` token type and recognition logic
- **Token** (`src/kscript/token.py`): Add `STRING_LITERAL` to `TokenType` enum
- **Parser** (`src/kscript/parser.py`): Handle `STRING_LITERAL` in node position
- **Compiler** (`src/kscript/compiler.py`): Convert string literals to appropriate node representation
