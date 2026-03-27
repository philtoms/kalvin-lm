# Spec: Multi-Character Signatures

## Purpose

Defines the implicit hierarchical canonization semantics for multi-character uppercase signatures (MCS). When KScript encounters a signature like `ABC`, it represents a composite concept with S2-level relationships to its constituent characters `A`, `B`, and `C`.

## Requirements

### Requirement: MCS Canonization
The compiler SHALL emit an implicit canonization entry for any multi-character signature (length > 1) encountered in the source.

#### Scenario: Simple MCS expansion
- **WHEN** compiler processes signature `ABC`
- **THEN** entry `{ABC: [A, B, C]}` is emitted with S2 significance

#### Scenario: MCS in construct position
- **WHEN** compiler processes `ABC => X`
- **THEN** entry `{ABC: [A, B, C]}` is emitted (implicit MCS expansion)
- **AND** entry `{ABC: [X]}` is emitted (explicit construct)

#### Scenario: Single-character bypass
- **WHEN** compiler processes signature `A`
- **THEN** no MCS expansion entry is emitted (single-char is atomic)

---

### Requirement: MCS Component Identity
The compiler SHALL emit identity entries for each component character of an MCS.

#### Scenario: Component identities
- **WHEN** compiler processes signature `ABC`
- **THEN** entries `{A: null}`, `{B: null}`, `{C: null}` are emitted with S4 significance

#### Scenario: No duplicate identities
- **WHEN** compiler processes `ABC => X` followed by `AB => Y`
- **THEN** identity entries for `A` and `B` are only emitted once

---

### Requirement: MCS Entry Ordering
MCS expansion entries SHALL be emitted before constructs that use the signature.

#### Scenario: Ordering with explicit constructs
- **WHEN** compiler processes `ABC == X`
- **THEN** entries are emitted in order:
  1. `{ABC: [A, B, C]}` (MCS canonization)
  2. `{A: null}`, `{B: null}`, `{C: null}` (identities)
  3. `{ABC: X}`, `{X: ABC}` (countersign construct)

---

### Requirement: MCS Significance Level
MCS canonization entries SHALL use S2 significance (canonize level).

#### Scenario: S2 significance encoding
- **WHEN** MCS canonization entry `{ABC: [A, B, C]}` is emitted
- **THEN** the signature has bit 55 set (S2 indicator)
- **AND** component nodes have no significance bits (S4)

---

### Requirement: MCS in All Positions
MCS expansion SHALL occur regardless of where the signature appears.

#### Scenario: MCS as script signature
- **WHEN** compiler processes `ABC => X`
- **THEN** `{ABC: [A, B, C]}` is emitted

#### Scenario: MCS as construct node
- **WHEN** compiler processes `X => ABC`
- **THEN** `{ABC: [A, B, C]}` is emitted

#### Scenario: MCS in countersign
- **WHEN** compiler processes `ABC == XYZ`
- **THEN** `{ABC: [A, B, C]}` is emitted
- **AND** `{XYZ: [X, Y, Z]}` is emitted
