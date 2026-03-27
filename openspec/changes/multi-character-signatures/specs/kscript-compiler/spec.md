# Spec: KScript Compiler (Delta)

## MODIFIED Requirements

### Requirement: Compiler - Canonize Semantics
The compiler SHALL emit multi-node entries for canonize constructs AND implicit MCS canonization entries.

#### Scenario: Forward canonize
- **WHEN** compiler processes `AB => C D`
- **THEN** output contains `{AB: ["A", "B"]}` (implicit MCS)
- **AND** output contains `{A: null}`, `{B: null}` (component identities)
- **AND** output contains `{AB: ["C", "D"]}` (explicit construct)

#### Scenario: Backward canonize with trailing node
- **WHEN** compiler processes `X <= A BC`
- **THEN** output contains `{BC: ["B", "C"]}` (implicit MCS for BC)
- **AND** output contains `{A: ["X"]}` (last node is parent)

#### Scenario: Backward canonize with leading nodes
- **WHEN** compiler processes `B C D <= A`
- **THEN** output contains `{A: ["B", "C", "D"]}` (explicit construct)
- **AND** MCS expansion entries for B, C, D are emitted

#### Scenario: Single-char signature no expansion
- **WHEN** compiler processes `A => B C`
- **THEN** no MCS expansion for A (single char)
- **AND** MCS expansion for BC if used elsewhere
