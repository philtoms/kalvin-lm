# Spec: KScript Grammar

## Purpose

Formal grammar definition for KScript, a domain-specific language for defining knowledge graph relationships. This spec defines the syntax, semantics, and compilation rules for the language.

## Requirements

### Requirement: Grammar Structure
The KScript language SHALL use the following grammar:

```
script ::= construct+

construct ::=
  | sig                              -- identity
  | sig == node                      -- countersign
  | sig > node                       -- connotate fwd
  | sig = node                       -- undersign
  | sig => construct                 -- canonize fwd (right-associative)
  | construct <= construct           -- canonize bwd: ALL left nodes
  | construct < construct            -- connotate bwd: CLOSEST left node
  | construct construct*             -- sequence

sig ::= [A-Z]+
node ::= sig | literal
literal ::= ![A-Z]+
```

#### Scenario: Identity construct
- **WHEN** parser processes `A`
- **THEN** a single identity construct is created with signature A

#### Scenario: Forward chain
- **WHEN** parser processes `A => B => C`
- **THEN** constructs are nested: `A => (B => C)`

#### Scenario: Backward binding
- **WHEN** parser processes `A <= B`
- **THEN** BWD operator binds two constructs where RIGHT points back to LEFT

---

### Requirement: BWD Operator Semantics
Backward operators SHALL bind constructs, not nodes.

#### Scenario: Canonize BWD binds ALL left nodes
- **WHEN** compiler processes `A B <= C`
- **THEN** output contains `{C: [A, B]}` (C points back to ALL left nodes)

#### Scenario: Connotate BWD binds CLOSEST left node
- **WHEN** compiler processes `A B < C`
- **THEN** output contains `{C: [B]}` (C points back to CLOSEST/B last node only)

#### Scenario: BWD with chain
- **WHEN** compiler processes `A => B <= C => D`
- **THEN** output contains `{A: [B]}`, `{C: [B]}`, `{C: [D]}`
- **AND** C is the signature owner of both BWD and subsequent FWD constructs

---

### Requirement: Signature Position
A signature SHALL be in "signature position" when it owns a construct.

#### Scenario: Script signature position
- **WHEN** `ABC` appears as the first token of a script
- **THEN** ABC is in signature position

#### Scenario: FWD operator right side
- **WHEN** compiler processes `A => BC`
- **THEN** BC is in signature position (right side of FWD continues as construct owner)

#### Scenario: BWD operator right side
- **WHEN** compiler processes `A <= BC`
- **THEN** BC is in signature position (right side of BWD owns the backward construct)

#### Scenario: Node position
- **WHEN** compiler processes `A => B C`
- **THEN** B and C are in NODE position (not signature position)

---

### Requirement: Entity Emission
The compiler SHALL emit entity entries (`{sig: None}`) for signatures.

#### Scenario: Entity for identity
- **WHEN** compiler processes `A` (identity)
- **THEN** output contains `{A: None}`

#### Scenario: Entity for nodes not continuing with FWD
- **WHEN** compiler processes `A => B <= C`
- **THEN** output contains `{B: None}` (B doesn't continue with FWD)

#### Scenario: Entity for nodes in constructs
- **WHEN** compiler processes `A => B C`
- **THEN** output contains `{B: None}` and `{C: None}` (nodes get entities)

#### Scenario: No entity for literals
- **WHEN** compiler processes `A => hello`
- **THEN** output does NOT contain `{hello: None}` (literals get no entities)

#### Scenario: Duplicate entities allowed
- **WHEN** compiler processes `A => B <= C => B`
- **THEN** output contains two `{B: None}` entries (different cascades)

---

### Requirement: MCS Expansion
Multi-character signatures (MCS) in signature position SHALL be expanded.

#### Scenario: MCS canonization
- **WHEN** compiler processes `ABC` in signature position
- **THEN** output contains `{ABC: [A, B, C]}` with S2 significance

#### Scenario: MCS component identities
- **WHEN** compiler processes `ABC` in signature position
- **THEN** output contains `{A: None}`, `{B: None}`, `{C: None}` with S4 significance

#### Scenario: MCS in node position - no expansion
- **WHEN** compiler processes `A => ABC`
- **THEN** output does NOT contain `{ABC: [A, B, C]}` (ABC is in node position)

#### Scenario: MCS on BWD right side
- **WHEN** compiler processes `A <= BCD`
- **THEN** output contains `{BCD: [B, C, D]}` (BCD is in signature position on right of BWD)

#### Scenario: MCS on BWD left side - no expansion
- **WHEN** compiler processes `ABC <= D`
- **THEN** output does NOT contain `{ABC: [A, B, C]}` (ABC is in node position on left of BWD)

---

### Requirement: Script Boundaries
Scripts SHALL start at column 1 (indent level 0).

#### Scenario: Multiple scripts
- **WHEN** parser processes `A => B\nC => D` (C at column 1)
- **THEN** two scripts are created: A and C

#### Scenario: Literals at script start ignored
- **WHEN** parser processes `hello\nA => B` (literal at column 1)
- **THEN** literal `hello` is ignored
- **AND** script starts at A

#### Scenario: Indented lines continue script
- **WHEN** parser processes `A =>\n  B\n  C`
- **THEN** single script A with B and C as continuation

---

### Requirement: Subscript Layout
INDENT/DEDENT SHALL be syntactic sugar for inline constructs.

#### Scenario: Subscript equivalent to inline
- **WHEN** parser processes either:
  ```
  A =>
    B
    C = D
  ```
  OR `A => B C = D`
- **THEN** both produce identical compiled output

---

### Requirement: Literal Handling
Literals (anything not `[A-Z]+`) SHALL only appear in node positions.

#### Scenario: Literal as node
- **WHEN** compiler processes `A => hello123`
- **THEN** `hello123` is treated as a literal node

#### Scenario: No literal entities
- **WHEN** compiler processes any construct with literal nodes
- **THEN** no entity entries are emitted for literals

#### Scenario: Literal at script start
- **WHEN** parser encounters literals at column 1 before any signature
- **THEN** literals are ignored until first signature

---

### Requirement: Empty Construct Recovery
Empty constructs SHALL recover as identity.

#### Scenario: Empty FWD construct
- **WHEN** compiler processes `A =>` with no following nodes
- **THEN** output contains `{A: None}` (identity recovery)

#### Scenario: Empty MCS construct
- **WHEN** compiler processes `ABC =>` with no following nodes
- **THEN** output contains MCS expansion for ABC plus identity

---

### Requirement: Operator Precedence
Operators SHALL have the following precedence (highest to lowest):

1. Sequence (construct construct*)
2. FWD operators (`=>`, `>`, `==`, `=`)
3. BWD operators (`<=`, `<`)

#### Scenario: FWD right-associative
- **WHEN** parser processes `A => B => C`
- **THEN** parse is `A => (B => C)`

#### Scenario: BWD after FWD
- **WHEN** parser processes `A => B <= C`
- **THEN** parse is `(A => B) <= C`
- **AND** C points back to B (the result of A => B)
