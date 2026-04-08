## Why

The current KScript compiler in `src/kscript/` evolved incrementally without a formal specification, making it difficult to verify correctness, onboard new developers, or confidently refactor. A comprehensive spec now exists at `openspec/specs/kscript-compiler/spec.md` that defines all language constructs, semantics, and output formats.

This change rebuilds the compiler from scratch using the spec as the source of truth, ensuring the implementation is verifiable, maintainable, and fully documented.

## What Changes

- **BREAKING**: Complete rewrite of `src/kscript/` package
- Lexer tokenization with Python-style INDENT/DEDENT handling
- Recursive descent parser with immediate binding semantics
- Compiler with all construct semantics (countersign, canonize, connotate, undersign)
- Literal encoding with PACKED_BIT distinction
- Output formats: JSON, JSONL, binary (KSC1)
- KScript API class for programmatic use
- CLI entry point for command-line compilation

## Capabilities

### New Capabilities

- `kscript-compiler`: Full compiler implementation from spec - lexer, parser, AST, compiler, output formats, API, and CLI

### Modified Capabilities

(None - this is a clean-room rebuild, not a modification of existing specs)

## Impact

**Files affected:**
- `src/kscript/` - complete rewrite of all modules
- `tests/test_kscript.py` - updated to match spec scenarios

**Dependencies:**
- `kalvin.mod_tokenizer` - ModTokenizer for encoding signatures
- `kalvin.abstract` - KLine base class for CompiledEntry

**API stability:**
- `KScript` class API remains compatible
- `CompiledEntry` extends `KLine` (unchanged interface)
- CLI arguments unchanged

**Integration points:**
- `ui/kscript/` TUI app uses `KScript` API (no changes needed)
- `kalvin.rationalise()` receives `CompiledEntry` as `KLine` (unchanged)
