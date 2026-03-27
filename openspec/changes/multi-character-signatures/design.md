## Context

The KScript compiler currently treats multi-character signatures (MCS) as opaque identifiers. A signature like `ABC` is encoded as a single packed token with no relationship to its component characters `A`, `B`, `C`. This loses semantic structure that users implicitly express when writing composite identifiers.

The current compiler flow:

```
Source → Lexer → Parser → Compiler → Entries
                                    ↓
                            Each signature → single token
```

The new flow adds MCS expansion at the compiler level:

```
Source → Lexer → Parser → Compiler → Entries
                                    ↓
                            Each signature → if len > 1:
                                emit canonization + identities
```

## Goals / Non-Goals

**Goals:**

- Emit implicit canonization entries for multi-character signatures
- Emit identity entries for each component character
- Preserve ordering: MCS entries before constructs using them

**Non-Goals:**

- No lexer/parser changes (MCS is a compiler concern)
- No AST modification (inference at emission time)
- No source-level visibility of MCS expansion
- No change to single-character signature handling

## Decisions

### Decision 1: Compiler-Level Inference

**Choice**: Infer MCS expansion during compilation, not at parse time.

**Alternatives considered:**

1. **AST transformation**: Add implicit `ABC => A B C` constructs to the AST
   - Rejected: Adds complexity to parser, violates "source is truth"
2. **Post-processing pass**: Add a separate MCS expansion pass after compilation
   - Rejected: Requires two passes, more complex
3. **Compiler inference (chosen)**: During signature processing, check length and emit
   - Chosen: Single pass, minimal change, local to compiler

**Rationale**: Compiler already iterates over signatures. Adding MCS check is a simple conditional at the point of signature emission.

### Decision 2: Entry Ordering

**Choice**: MCS entries emitted immediately before the construct that uses them.

**Rationale**: Ensures MCS canonization and identities exist before any construct references them. Natural ordering from single-pass compilation.

## Risks / Trade-offs

| Risk                                                     | Mitigation                                                    |
| -------------------------------------------------------- | ------------------------------------------------------------- |
| Output bloat for many MCS                                | Acceptable - semantic completeness is worth the size increase |
| Duplicate component identities (A appears in ABC and AD) | Acceptable - Kalvin model handles duplicates internally       |
| Ordering guarantees                                      | Single-pass ensures MCS emitted before constructs using them  |
