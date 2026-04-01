# KScript Compiler v2

## Why

The current KScript compiler has accumulated complexity:
(~480 lines of state tracking for "current owner" semantics, subscript normalization during parsing). This has made the implementation difficult to maintain and extend. A simpler mental model is needed that:
The model:

- **CLNs (Construct Level Nodes)** are nodes collected between construct operators types, not signatures themselves. This makes the parser easier to reason about.
- **Eager emit** - no buffering, emit immediately as each construct is completed
- **Two-step compilation** - collect CLNs, emit, then handle BWD if present

- **Cleaner semantics** - CLN collection is construct-type agnostic, BWD is just an extra emission using collected CLNs
- **Subscripts** are constructs themselves, processed recursively in step 2
- **MCS (Multi-character Signature)** expansion happens for any signature in construct owner position

## What Changes

- **BREAKING**: Complete rewrite of `parser.py` and `compiler.py` modules
- New CLN-based node collection model with cleaner semantics
- Grammar explicitly defined:
- `node ::= sig | literal | construct.sig` where `construct.sig` is a signature or CLN, or literal
- `sig ::= [A-Z]+`
- `literal ::= ![A-Z]+
- `construct ::= sig -- identity (S4)
  | sig == node -- countersign (S1)
  | sig > node -- connotate fwd (S3)
  | sig => node+ -- undersign (S1)
  | sig => node+ -- canonize fwd (S2 right-associative)
  | construct <= construct -- canonize bwd (S2 left associative)
  | construct < construct -- connotate bwd (S3, CLOSEST left node)
  - **Subscripts**: `<INDENT> construct+ <DEDENT>` normalized to inline constructs during parsing
- **Eager emit**: No buffering, emit immediately per construct is completed
  - **Two-step compilation**: Collect CLNs, emit main entry, then handle BWD if present

  - **Output**: JSON, JSONL, binary formats

## Capabilities

### New Capabilities

- `kscript-compilation`: A KScript language grammar and compilation semantics

### Modified Capabilities

- None (new capability, no requirement changes to existing specs)

## Impact

- **Parser**: `src/kscript/parser.py` - complete rewrite
- **Compiler**: `src/kscript/compiler.py` - minor updates to `lexer.py`, `ast.py`, `output.py`
- **Tests**: `tests/test_kscript.py`
- **Decompiler**: `src/kscript/decompiler.py` (minor updates)

- **CLI**: `src/kscript/__main__.py`

- **API**: `src/kscript/__init__.py`, `src/kscript/api.py`
