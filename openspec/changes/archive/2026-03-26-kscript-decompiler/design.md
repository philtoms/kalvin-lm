## Context

Kalvin outputs ordered KLines where:
- The **first KLine** is the primary statement (everything else satisfies it)
- **Significance is always set** - every signature has a significance level:
  - S1 (bit 63): countersign (`==`) - bidirectional signature link
  - S2 (bit 62): canonize (`=>`) - multi-node composition
  - S3 (bit 61): connotate (`>`) - single-node annotation
  - S4 (no bits): identity - signature exists with no special relationship
- **Missing entries** = identity (node appears but has no KLine)

The decompiler reverses the compilation process: ordered KLines → KScript source.

## Goals / Non-Goals

**Goals:**
- Convert ordered KLines with significance levels to valid KScript source
- Reconstruct subscript structure for multi-node canonize constructs
- Surface anomalies/invalid KLines with clear markers (development tool)
- Support semantic decompilation (output may differ from original but produces same KLines)

**Non-Goals:**
- Lossless decompilation (comments, formatting, exact original structure)
- Handling truly invalid KLines (MVP assumes significance is always set - S4 is valid)
- Optimizing output for human readability beyond basic formatting

## Decisions

### D1: Primary Statement Model
**Decision:** The first KLine is the primary statement. All subsequent KLines exist to satisfy it.

**Rationale:**
- Kalvin outputs KLines in dependency order
- Decompiler follows the chain from primary through nodes
- Missing entries are identity (just output signature name)

**Algorithm:**
```
1. Build lookup: signature → KLine
2. Process primary signature recursively
3. For each node, check if it has an entry:
   - Yes → recurse to emit its constructs
   - No → emit as identity (just the signature)
```

### D2: Significance Level Detection
**Decision:** Use significance bits to determine construct operator. S4 (no bits) is a valid level meaning identity.

| Bits | Level | Operator | Pattern |
|------|-------|----------|---------|
| bit 63 set | S1 | `==` | `{AB: CD}` → `AB == CD` |
| bit 62 set | S2 | `=>` | `{CD: [C, D]}` → `CD => C D` |
| bit 61 set | S3 | `>` | `{C: 1}` → `C > 1` |
| no bits | S4 | identity | `{X: None}` → `X` |

**Rationale:**
- Every signature has significance (it's the essence of KLine)
- S4 = "no special relationship" = identity
- This is not an error case, it's a valid level

### D3: Countersign Pair Handling
**Decision:** Countersign produces TWO KLines but ONE construct in output.

**Example:**
```
{S1|AB: CD}    ← emit "AB == CD"
{S1|CD: AB}    ← skip (already processed)
```

**Rationale:**
- Countersign is bidirectional
- Emitting both would be redundant
- Mark both signatures as processed when first is emitted

### D4: Subscript vs Inline
**Decision:** Canonize with multiple nodes always uses subscripts.

**Example:**
```
{S2|CD: [C, D]}
{S3|C: 1}
{S3|D: 2}

Output:
CD =>
  C > 1
  D > 2
```

**Rationale:**
- KScript syntax requires subscripts when nodes have constructs
- Simpler rule: multi-node canonize → subscripts
- Nodes without entries emit as identity

### D5: Error Surfacing
**Decision:** Truly invalid/unexpected KLines surface with `!!!` markers.

**Examples:**
```
!!! ORPHAN: {sig}            ← unreachable from primary
!!! BROKEN: expected {sig}   ← chain references non-existent sig
```

**Rationale:**
- Development tool mentality
- Fail loud, don't hide problems
- User (lead developer) needs visibility into anomalies

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Circular references | Track processed set, skip already-processed |
| Deep nesting | No mitigation needed - just indent |
| Output differs from original | Document: semantic equivalence, not lossless |
