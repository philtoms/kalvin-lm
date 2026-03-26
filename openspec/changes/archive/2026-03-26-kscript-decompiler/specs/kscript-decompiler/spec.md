## ADDED Requirements

### Requirement: Decompiler converts KLines to KScript source
The decompiler SHALL convert an ordered list of KLines with significance bits to valid KScript source code.

#### Scenario: Basic decompilation
- **WHEN** decompiler receives ordered KLines with significance bits
- **THEN** decompiler outputs valid KScript source that represents the same relationships

### Requirement: Significance level detection
The decompiler SHALL use significance bits to determine the construct operator.

#### Scenario: S1 countersign detection
- **WHEN** KLine has bit 63 set (S1)
- **THEN** decompiler emits countersign operator `==`

#### Scenario: S2 canonize detection
- **WHEN** KLine has bit 62 set (S2)
- **THEN** decompiler emits canonize operator `=>`

#### Scenario: S3 connotate detection
- **WHEN** KLine has bit 61 set (S3)
- **THEN** decompiler emits connotate operator `>`

#### Scenario: S4 identity detection
- **WHEN** KLine has no significance bits set (S4)
- **THEN** decompiler emits just the signature (identity)

### Requirement: Primary statement processing
The decompiler SHALL treat the first KLine as the primary statement and process all subsequent KLines as satisfying it.

#### Scenario: Primary statement is starting point
- **WHEN** decompiler processes KLines
- **THEN** first KLine signature becomes the root of the output script

#### Scenario: Missing entry is identity
- **WHEN** node appears in a construct but has no KLine entry
- **THEN** decompiler emits just the signature name (identity)

### Requirement: Countersign pair handling
The decompiler SHALL emit ONE countersign construct for a bidirectional pair.

#### Scenario: Countersign pair deduplication
- **WHEN** KLines `{S1|AB: CD}` and `{S1|CD: AB}` both exist
- **THEN** decompiler emits `AB == CD` (not both directions)

### Requirement: Subscript reconstruction
The decompiler SHALL emit subscripts for canonize constructs with multiple nodes.

#### Scenario: Multi-node canonize with subscripts
- **WHEN** canonize has multiple nodes and nodes have their own constructs
- **THEN** decompiler emits subscript structure:
```
SIG =>
  NODE1 > ...
  NODE2 > ...
```

#### Scenario: Identity in subscripts
- **WHEN** canonize node has no KLine entry
- **THEN** decompiler emits just the node signature in subscript

### Requirement: Error surfacing
The decompiler SHALL surface anomalies with clear markers for developer visibility.

#### Scenario: Orphaned KLine
- **WHEN** KLine is not reachable from primary statement
- **THEN** decompiler emits `!!! ORPHAN: {signature}`

#### Scenario: Broken chain reference
- **WHEN** construct references a signature that doesn't exist
- **THEN** decompiler emits `!!! BROKEN: expected {signature}`
