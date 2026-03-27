## Why

Multi-character signatures (MCS) like `ABC` currently encode as opaque identifiers with no internal structure. This loses the semantic relationship between a concept and its constituent parts. When users write `ABC`, they're expressing a composite concept that inherently relates to `A`, `B`, and `C`. The compiler should recognize and preserve this hierarchical relationship automatically.

## What Changes

- **MCS Implicit Canonization**: When the compiler encounters a multi-character signature (e.g., `ABC`), it will implicitly emit a canonization entry `{ABC: [A, B, C]}` with S2 significance
- **Component Identity Entries**: Each component character gets an implicit identity entry (e.g., `{A: null}`, `{B: null}`, `{C: null}`) with S4 significance
- **Single-Character Bypass**: Single-character signatures (e.g., `A`) are not expanded - they're already atomic
- **Compiler Inference**: The compiler infers MCS expansion without requiring source AST modification

## Capabilities

### New Capabilities

- `multi-character-signatures`: Implicit hierarchical canonization for multi-character uppercase identifiers

### Modified Capabilities

- `kscript-compiler`: Compiler emits additional entries for MCS expansion (canonization + identities) when processing signatures with length > 1

## Impact

- **Compiler** (`src/kscript/compiler.py`): MCS expansion logic during signature processing
- **No lexer/parser changes**: MCS is a compiler-level concern
- **No breaking changes**: Existing scripts continue to work; MCS adds entries, doesn't remove
