# Design: KScript Compiler v2

## Context

The current KScript compiler in `src/kscript/` consists of:
- **Lexer** (`token.py`, `lexer.py`): 14 token types, Python-style INDENT/DEDENT
- **Parser** (`parser.py`): ~480 lines with complex state tracking for "current owner" and BWD binding
- **AST** (`ast.py`): Dataclasses for Signature, Literal, Construct, Script, KScriptFile
- **Compiler** (`compiler.py`): Compiles AST to CompiledEntry list
- **Output** (`output.py`): JSON, JSONL, binary formats
- **Decompiler** (`decompiler.py`): Reverse compilation for debugging

**Current problems:**
1. Parser complexity from tracking "current owner" across constructs
2. BWD operators cause control flow changes rather than simple emissions
3. Subscripts are normalized during parsing, mixing concerns
4. MCS expansion logic scattered across compiler

**Constraints:**
- Must maintain backward compatibility with compiled output format
- Must pass existing test suite
- Significance encoding (S1-S4) must be preserved

## Goals / Non-Goals
**Goals:**
- Simplify parser to ~200 lines with CLN-based semantics
- Unify compilation with eager emit pattern
- Clear separation of parsing and compilation phases
- Make MCS expansion consistent and explicit

**Non-Goals:**
- Changing the output format (JSON/JSONL/binary remain same)
- Modifying the decompiler
- Adding new operators

## Decisions
### Decision 1: CLN-Based Node Collection
**Choice**: CLNs are collected between operators, not "owned by" constructs.
**Rationale**: This eliminates the need for "current owner" tracking. CLNs are just nodes that appear between construct operators, making parsing stateless and straightforward.
**Alternative considered**: Owner tracking (current approach) - rejected due to complexity.

### Decision 2: Eager Emit Compilation
**Choice**: Emit CompiledEntry immediately when each construct is parsed.
**Rationale**: Eliminates buffering and makes the compiler stateless. BWD operators simply trigger an additional emission using already-collected CLNs.
**Alternative considered**: Two-pass compilation - rejected as unnecessary complexity.

### Decision 3: Two-Step Compilation Per Construct
**Choice**: Each construct comp compiled in two steps:
1. **Step 1**: Collect sig and CLNs, emit MCS (if applicable), emit main entry, handle BWD if present
2. **Step 2**: Process subscripts recursively for each CLN
**Rationale**: Clear separation of concerns. Step 1 is inline processing, Step 2 is recursive subscript handling.
**Alternative considered**: Single-pass with buffering - rejected due to memory overhead.

### Decision 4: BWD Semantics
**Choice**: BWD operators use CLNs from the current construct (not accumulated across sequence).
- `<=` (S2): binds ALL CLNs from current construct
- `<` (S3): binds only `CLNs[-1]` (closest)
**Rationale**: Simplifies BWD handling - no cross-construct state needed.
**Alternative considered**: Accumulating CLNs across sequence - rejected as it creates "sequence_clns" state.

## Risks / Trade-offs
| Risk | Mitigation |
|------|-----------|
| Duplicate entity emissions | By design - duplicates are acceptable for graph completeness |
| Literal in BWD sig position | Parser skips BWD, treats left side as identity |
| Complex subscript nesting | Tested with 3+ levels of nesting |

## Migration Plan
1. Create new `parser_v2.py` and `compiler_v2.py` alongside existing modules
2. Implement new parser with CLN semantics
3. Implement new compiler with eager emit
4. Port tests to new test file
5. Run tests in parallel with old tests
6. Once passing, deprecate old modules
7. Update imports in `__init__.py`

## Open Questions
- Should we support mixed signature/literal CLNs in the same construct?
- How to handle error recovery for malformed input?
