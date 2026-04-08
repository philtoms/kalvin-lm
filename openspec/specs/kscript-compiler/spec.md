# Delta Spec: KScript Compiler

## Purpose

This delta spec modifies the KScript Compiler to implement the new grammar where BWD operators bind constructs, not nodes. See `kscript-grammar` spec for the formal grammar definition.

## MODIFIED Requirements

### Requirement: Parser - Immediate Binding
The parser SHALL implement recursive grammar where BWD operators bind constructs.

#### Scenario: Chained FWD constructs
- **WHEN** parser processes `A => B => C`
- **THEN** first construct is `A => (B => C)`
- **AND** compiled output is `{A: [B]}`, `{B: [C]}`

#### Scenario: BWD binds constructs
- **WHEN** parser processes `A => B <= C`
- **THEN** parse is `(A => B) <= C`
- **AND** C owns a BWD construct pointing back to B's nodes
- **AND** compiled output is `{A: [B]}`, `{B: None}`, `{C: [B]}`, `{B: None}`

#### Scenario: BWD with chain continuation
- **WHEN** parser processes `A => B <= C => D`
- **THEN** C owns both the BWD construct AND the subsequent FWD construct
- **AND** compiled output is `{A: [B]}`, `{B: None}`, `{C: [B]}`, `{B: None}`, `{C: [D]}`, `{D: None}`

---

### Requirement: Parser - Subscripts
The parser SHALL treat INDENT/DEDENT as layout sugar for inline constructs.

#### Scenario: Subscript normalized to inline
- **WHEN** parser processes:
  ```
  A =>
    B
    C
  ```
- **THEN** it is semantically equivalent to `A => B C`
- **AND** no nested Script nodes are created

#### Scenario: Nested subscript normalized
- **WHEN** parser processes:
  ```
  A =>
    B =>
      C
  ```
- **THEN** it is semantically equivalent to `A => B => C`

---

### Requirement: Compiler - Canonize Semantics
The compiler SHALL emit entries based on construct ownership.

#### Scenario: Forward canonize
- **WHEN** compiler processes `AB => C D`
- **THEN** output contains `{AB: [C, D]}`
- **AND** entity entries for C and D

#### Scenario: Backward canonize binds ALL left nodes
- **WHEN** compiler processes `A B <= C`
- **THEN** output contains `{C: [A, B]}`
- **AND** C is the signature owner (RIGHT side of BWD)

#### Scenario: BWD in chain
- **WHEN** compiler processes `A => B <= C`
- **THEN** output contains `{A: [B]}`, `{B: None}`, `{C: [B]}`, `{B: None}`
- **AND** C points back to B (the last node of the left construct)

---

### Requirement: Compiler - Connotate Semantics
The compiler SHALL emit entries with identity for the node.

#### Scenario: Forward connotate
- **WHEN** compiler processes `A > B`
- **THEN** output contains `{A: [B]}` AND `{B: None}`

#### Scenario: Backward connotate binds CLOSEST only
- **WHEN** compiler processes `A B < C`
- **THEN** output contains `{C: [B]}` AND `{B: None}`
- **AND** C points back to B only (the CLOSEST/last node), not A

---

### Requirement: Error Recovery
The compiler SHALL recover from syntax errors and produce valid output.

#### Scenario: Literal at script start
- **WHEN** compiler processes `hello world\nA => B`
- **THEN** literals at start are ignored
- **AND** script begins at signature A

#### Scenario: Empty construct recovery
- **WHEN** compiler processes `A =>` with no nodes
- **THEN** output contains `{A: None}` (identity recovery)
- **AND** if A is multi-char, MCS expansion is applied first

#### Scenario: Literal in signature position mid-script
- **WHEN** compiler processes `A => hello B`
- **THEN** `hello` is treated as a literal node
- **AND** output contains `{A: [hello, B]}`

## REMOVED Requirements

### Requirement: Parser - Backward Canonize with Leading Nodes
**Reason:** Replaced by new grammar where BWD operators bind constructs. The pattern `B C D <= A` is now parsed as sequence `(B C D) <= A` where LEFT construct provides nodes.

**Migration:** Use the new BWD construct semantics. `B C D <= A` now means A points back to B, C, D.

---

### Requirement: Compiler - Subscripts as Nodes
**Reason:** Subscripts are now layout sugar, not a separate semantic concept. The parser normalizes subscripts to inline sequences before compilation.

**Migration:** Subscript syntax continues to work but is normalized to inline. `A =>\n  B\n  C` is equivalent to `A => B C`.

## ADDED Requirements

### Requirement: Compiler - MCS for BWD Signatures
The compiler SHALL expand MCS for signatures in signature position on BWD right side.

#### Scenario: MCS on BWD right side
- **WHEN** compiler processes `A <= BCD`
- **THEN** output contains MCS expansion `{BCD: [B, C, D]}`, `{B: None}`, `{C: None}`, `{D: None}`
- **AND** BWD construct `{BCD: [A]}`

#### Scenario: No MCS on BWD left side
- **WHEN** compiler processes `ABC <= D`
- **THEN** output does NOT contain MCS for ABC
- **AND** ABC is treated as nodes in the left construct

---

### Requirement: Compiler - Entity Emission Rules
The compiler SHALL emit entity entries for all signatures that don't continue with FWD operators.

#### Scenario: Entity at end of chain
- **WHEN** compiler processes `A => B`
- **THEN** output contains `{B: None}` (B doesn't continue)

#### Scenario: Entity before BWD
- **WHEN** compiler processes `A => B <= C`
- **THEN** output contains `{B: None}` (B is followed by BWD, not FWD)

#### Scenario: No entity before FWD
- **WHEN** compiler processes `A => B => C`
- **THEN** output does NOT contain `{B: None}` between constructs (B continues with FWD)
