## Context

The KScript compiler transforms `.ks` source files into compiled entries (KLines). The current implementation has a fundamental flaw: backward (BWD) operators treat signatures as nodes of the previous construct rather than as owners of new constructs.

**Current Architecture:**
```
source.ks → Lexer → Parser → Compiler → CompiledEntry[]
```

- Lexer tokenizes with INDENT/DEDENT support (unchanged)
- Parser produces AST with Script/Construct nodes
- Compiler emits CompiledEntry objects with significance encoding

**Problem:** Parser produces flat construct lists where BWD signatures become nodes. Compiler then incorrectly binds them to previous constructs.

## Goals / Non-Goals

**Goals:**
- Implement correct grammar where BWD operators bind constructs
- MCS expansion for all signatures in signature position
- Entity emission for all signatures that don't continue with FWD
- Script boundary detection at column 1
- Empty construct recovery

**Non-Goals:**
- Lexer changes (tokenization unchanged)
- Output format changes (JSON/JSONL/binary unchanged)
- Decompiler updates (separate change)
- Performance optimization

## Decisions

### D1: Grammar Structure

**Decision:** Use recursive grammar where BWD operators bind constructs, not nodes.

```
script ::= construct+

construct ::=
  | sig                              -- identity
  | sig == node                      -- countersign
  | sig > node                       -- connotate fwd
  | sig = node                       -- undersign
  | sig => construct                 -- canonize fwd (right-assoc)
  | construct <= construct           -- canonize bwd: ALL left nodes
  | construct < construct            -- connotate bwd: CLOSEST left node
  | construct construct*             -- sequence

sig ::= [A-Z]+
node ::= sig | literal
literal ::= ![A-Z]+
```

**Rationale:** This grammar ensures every construct has a signature owner. BWD operators explicitly bind two constructs, making the "overlap" semantics clear.

**Alternative considered:** Keep flat construct list with look-ahead. Rejected because it complicates the compiler and doesn't express the construct ownership clearly.

### D2: Parser Architecture

**Decision:** Single-pass recursive descent parser with construct accumulation.

- Track `current_construct` as we parse
- On FWD operator (`=>`): continue building current construct
- On BWD operator (`<=`, `<`): emit current construct, start new one with BWD binding
- On sequence (sig after sig): emit entity for previous, start new construct

**Rationale:** Single-pass is simpler than building intermediate AST then compiling. The grammar is naturally recursive, matching recursive descent.

### D3: Entity Emission Rules

**Decision:** Emit entity (`{sig: None}`) when:
1. Signature doesn't continue with FWD operator (`=>`, `>`)
2. Signature appears as node in any construct
3. MCS character components (always S4)

**Rationale:** Entities ensure all referenced signatures exist in the graph. Duplicates are allowed (different cascades).

### D4: MCS Expansion Scope

**Decision:** Expand MCS ONLY when signature appears in signature position (construct owner), not in node position.

```
ABC => X    →  MCS for ABC (sig position)
A => ABC    →  NO MCS for ABC (node position)
ABC <= D    →  NO MCS for ABC (node position on left of BWD)
D <= ABC    →  MCS for ABC (sig position on right of BWD)
```

**Rationale:** MCS represents hierarchical decomposition of a concept. Only the construct owner represents a "concept" - nodes are just references. In BWD constructs, the RIGHT side owns the construct.

### D5: Subscript Layout Normalization

**Decision:** Parser normalizes INDENT/DEDENT to inline sequence during parsing.

```
A =>
  B
  C = D
```
Parsed as: `A => B C = D`

**Rationale:** Simplifies compiler - no special subscript handling needed. Layout is purely syntactic sugar.

### D6: Script Boundary Detection

**Decision:** New script starts when signature appears at column 1 (indent level 0).

- Ignore literals at column 1 until first signature
- Multiple scripts per file supported
- Each script is independently compiled

**Rationale:** Matches user mental model of "top-level definitions". Column 1 is unambiguous.

### D7: Error Recovery

**Decision:** Recover gracefully from common errors:
- Literal at script start: ignore until first signature
- Empty construct (`A =>`): emit identity
- Literal in sig position mid-script: treat as node

**Rationale:** Partial compilation is better than complete failure. Matches REPL-friendly design.

## Risks / Trade-offs

**Risk:** Breaking change for existing KScript files
→ **Mitigation:** This is a rebuild; existing files may need updates. Document migration.

**Risk:** Parser complexity with recursive grammar
→ **Mitigation:** Extensive test coverage for all operator combinations.

**Risk:** Performance with deep nesting
→ **Mitigation:** Python recursion limit is sufficient for typical scripts. Document if needed.

**Trade-off:** Duplicate entity entries increase output size
→ **Acceptance:** Duplicates represent different cascades and are semantically correct.

## Open Questions

1. **Literal entity emission:** Confirmed - literals do NOT get entity entries (only signatures do).
2. **MCS duplicate tracking:** Confirmed - do NOT deduplicate MCS entries (each occurrence emits fresh).
