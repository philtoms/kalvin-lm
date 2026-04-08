## Why

The current parser treats backward (BWD) signatures as nodes belonging to the previous construct. This is fundamentally wrong - BWD operators (`<=`, `<`) signal NEW constructs that "overlap" the previous one. For example, `A => B <= C` should produce `{A: [B]}` AND `{C: [B]}` (C points back to B), not treat C as a node of A's construct.

Additionally, multi-character signature (MCS) expansion is currently applied inconsistently - only to script signatures, not to signatures that become construct owners via BWD operators.

## What Changes

**BREAKING** - Complete parser and compiler rebuild with corrected grammar:

- **BWD operators bind constructs, not nodes**: `construct <= construct` means the RIGHT construct's signature points back to the LEFT construct's nodes
- **`<=` (canonize bwd) binds ALL nodes** from the left construct
- **`<` (connotate bwd) binds only the CLOSEST node** from the left construct
- **MCS expansion for all signatures in signature position**: Any multi-char signature that owns a construct gets MCS expansion (canonization + identities)
- **Subscripts are layout sugar**: `A => B C = D` is semantically equivalent to indented form
- **Literals never in signature position**: Literals (anything not `[A-Z]+`) can only be nodes, never construct owners
- **Script boundaries at column 1**: Multiple scripts per file, each starting at column 1
- **Empty construct recovery**: `A =>` with no nodes recovers as identity

## Capabilities

### New Capabilities

- `kscript-grammar`: Formal grammar definition with construct-level BWD binding, entity emission rules, and script boundary semantics

### Modified Capabilities

- `kscript-compiler`: Complete rebuild to implement new grammar - lexer unchanged, parser and compiler rewritten

## Impact

- **BREAKING**: Parser (`src/kscript/parser.py`) - complete rewrite with new grammar
- **BREAKING**: Compiler (`src/kscript/compiler.py`) - rewritten with new construct semantics
- **Tests** (`tests/test_kscript.py`) - updated to match new behavior
- **Unchanged**: Lexer (`src/kscript/lexer.py`) - tokenization unchanged
- **Unchanged**: Output module (`src/kscript/output.py`) - binary/JSON formats unchanged
- **Out of scope**: Decompiler - handled in separate change
