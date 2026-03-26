## MODIFIED Requirements

### Requirement: Significance encoding uses 32-bit space in upper signature bits
The compiler SHALL encode construct significance in bits 32-63 of the 64-bit signature.

#### Scenario: S1 countersign encoding
- **WHEN** compiler processes a countersign construct (`==`)
- **THEN** signature has bit 63 set (S1 indicator shifted from bit 31)

#### Scenario: S2 canonize encoding
- **WHEN** compiler processes a canonize construct (`=>`)
- **THEN** signature has bit 55 set (S2 indicator) with degree in bits 56-62

#### Scenario: S3 connotate encoding
- **WHEN** compiler processes a connotate construct (`>`)
- **THEN** signature has bit 32 set (S3 indicator) with degree in bits 33-54

#### Scenario: S4 undersign encoding
- **WHEN** compiler processes an undersign construct (`=`)
- **THEN** signature has no significance bits set (bits 32-63 are clear)

### Requirement: Token space remains in lower 32 bits
The compiler SHALL keep token encoding in bits 0-31 unchanged.

#### Scenario: PACKED_BIT unchanged
- **WHEN** compiler encodes any signature
- **THEN** bit 0 indicates packed vs literal (unchanged from current)

#### Scenario: Character tokenization unchanged
- **WHEN** compiler encodes signature characters
- **THEN** bits 1-31 contain character tokens (unchanged from current)
