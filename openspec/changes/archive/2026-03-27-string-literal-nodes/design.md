## Context

KScript currently supports two node types in constructs:
1. **Signatures**: `[A-Z]+` - uppercase identifiers that reference other KLines
2. **Quoted strings**: `"..."` - double-quoted string literals

The lexer (`lexer.py`) tokenizes signatures as `SIGNATURE` tokens. The parser (`parser.py`) distinguishes `Signature` vs `StringLiteral` nodes based on token type. The compiler (`compiler.py`) encodes signatures as packed tokens (PACKED_BIT set) and literals as unpacked character sequences.

## Goals / Non-Goals

**Goals:**
- Support unquoted string literals: `[a-zA-Z0-9]+` sequences that are not exclusively uppercase
- Maintain backward compatibility: all existing KScript files parse identically
- Preserve the clear distinction: uppercase = signature, lowercase/mixed = literal

**Non-Goals:**
- Support for whitespace in unquoted literals (must use quotes)
- Support for special characters in unquoted literals (must use quotes)
- Changing how quoted strings work

## Decisions

### D1: Add STRING_LITERAL token type (not reuse SIGNATURE)

**Rationale**: Keeping SIGNATURE as exclusively uppercase preserves the semantic clarity. A new `STRING_LITERAL` token type makes the distinction explicit at the lexer level.

**Alternative considered**: Reuse SIGNATURE with a flag - rejected because it conflates two distinct concepts and complicates downstream processing.

### D2: Lexer recognizes identifiers that contain lowercase or digits

**Rule**: After checking for uppercase (SIGNATURE), check for identifier pattern `[a-zA-Z0-9]+`:
- If all uppercase → SIGNATURE
- If contains lowercase or digits → STRING_LITERAL

**Rationale**: This allows `Hello`, `zed`, `World123` as literals while keeping `FOO`, `BAR` as signatures.

### D3: Parser creates StringLiteral from STRING_LITERAL tokens

**Rationale**: Consistent with existing pattern - the parser already creates `StringLiteral` nodes from quoted STRING tokens. The compiler already handles `StringLiteral` vs `Signature` distinction.

### D4: Compiler treats STRING_LITERAL nodes as literals (no reverse entries)

**Rationale**: Already implemented - the compiler checks `isinstance(node, Signature)` to decide whether to create reverse entries. `StringLiteral` nodes naturally skip reverse entries.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Ambiguity with mixed-case identifiers like `Hello` | Clear rule: anything not exclusively uppercase is a literal |
| Breaking existing scripts using lowercase as identifiers | Unlikely - current lexer skips lowercase entirely |
| Decompiler may not output unquoted form | Add logic to detect safe unquoted output |
