# Spec: KScript Compilation

## Purpose

Defines the compilation semantics for KScript, including CLN (Construct Level Node) collection, eager emission, MCS expansion, and significance level encoding.

## Requirements

### Requirement: CLN-based Node Collection
The KScript parser SHALL collect CLNs (Construct Level Nodes) as nodes appearing between construct operators tokens, not as nodes "owned by" a construct.
- CLNs SHALL be collected agnostic to construct type
- CLNs SHALL include signatures, literals, and construct signatures (when a construct appears in node position)
- Identity constructs SHALL have implicit CLN equal to [signature]

#### Scenario: CLN collection for canonize
- **WHEN** parser processes `A => B C <= D`
- **THEN** CLNs collected are [B, C] before `<=` operator
- **AND** BWD binds ALL CLNs: {D|S2: [B, C]}

#### Scenario: CLN collection for connotate BWD
- **WHEN** parser processes `A => B C < D`
- **THEN** CLNs collected as [B, C] before `<` operator
- **AND** BWD binds CLNs[-1]: {D|S3: [C]}

#### Scenario: CLN collection with implicit opening
- **WHEN** parser processes `A B <= CD` with no preceding operator
- **THEN** CLNs collected as [A, B] from implicit opening
- **AND** BWD binds ALL CLNs: {CD|S2: [A, B]}

---

### Requirement: Eager Emit Compilation
The compiler SHALL emit CompiledEntry immediately when each construct is completed.
- The compiler SHALL NOT buffer constructs
- BWD operators SHALL trigger additional emission using already-collected CLNs
- Eager emit SHALL maintain correct emission order per construct

#### Scenario: Eager emit for canonize
- **WHEN** compiler processes `A => B C`
- **THEN** {A|S2: [B, C]} emitted immediately
- **AND** no buffering occurs

#### Scenario: BWD triggers additional emit
- **WHEN** compiler processes `A => B C <= D`
- **THEN** {A|S2: [B, C]} emitted first
- **AND** {D|S2: [B, C]} emitted second (additional BWD entry)

---

### Requirement: MCS Expansion for Owner Signatures
Multi-character signatures (MCS) in owner position SHALL emit:
- Canonization entry: `{sig|S2: [char for char in sig]}`
- Identity entries: `{char|S4: None}` for each component char
- MCS expansion SHALL occur before main construct emission
- Single-character signatures SHALL NOT emit MCS expansion

#### Scenario: MCS expansion for multi-char owner
- **WHEN** compiler processes `ABC => X`
- **THEN** MCS canonization `{ABC|S2: [A, B, C]} emitted
- **AND** identity entries {A|S4: None}, {B|S4: None}, {C|S4: None} emitted

#### Scenario: No MCS for single-char owner
- **WHEN** compiler processes `A => X`
- **THEN** no MCS canonization emitted
- **AND** only construct entry {A|S2: [X]} emitted

---

### Requirement: BWD Literal Rejection
BWD operators with a literal in signature position SHALL be invalid.
- The parser SHALL skip BWD emission
- The left side SHALL be treated as identity construct
- Literal nodes SHALL NOT become BWD signature owners

#### Scenario: Literal in BWD sig position
- **WHEN** parser processes `A < 1` where `1` is literal
- **THEN** BWD is rejected
- **AND** A is treated as identity: {A|S4: None}

---

### Requirement: Subscript Processing
Subscripts SHALL be processed as recursive constructs.
- Subscripts SHALL attach to the last CLN of the preceding construct
- INDENT/DEDENT tokens SHALL delineate subscript boundaries
- Nested subscripts SHALL be supported

#### Scenario: Subscript attaches to last CLN
- **WHEN** parser processes:
```
A =>
  B
  C = D
```
- **THEN** subscript `C = D` attaches to B (last CLN of `A => B`)
- **AND** equivalent to inline `A => B C = D`

#### Scenario: Nested subscript
- **WHEN** parser processes:
```
A =>
  B =>
    C
    D
```
- **THEN** subscript `C` attaches to B
- **AND** subscript `D` attaches to C (nested)

---

### Requirement: Significance Level Emission
Compiled entries SHALL include significance level (S1-S4):
- S1 (countersign): bidirectional signature links
- S2 (canonize): multi-node composition
- S3 (connotate): single-node annotation with entity emission
- S4 (undersign/identity): unidirectional with entity emission
- Significance bits SHALL be added to token ID during encoding

#### Scenario: Countersign significance
- **WHEN** compiler processes `A == B`
- **THEN** entries emitted with S1 significance
- **AND** bidirectional entries {A|S1: B}, {B|S1: A} created

#### Scenario: Canonize significance
- **WHEN** compiler processes `AB => C D`
- **THEN** entry {AB|S2: [C, D]} emitted with S2 significance
- **AND** no bidirectional entry created

#### Scenario: Connotate significance
- **WHEN** compiler processes `A > B`
- **THEN** entry {A|S3: [B]} emitted with S3 significance
- **AND** entity entry {B|S4: None} emitted

#### Scenario: Undersign significance
- **WHEN** compiler processes `A = B`
- **THEN** entry {A|S4: B} emitted with S4 significance
- **AND** entity entry {B|S4: None} emitted
