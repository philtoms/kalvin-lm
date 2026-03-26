## MODIFIED Requirements

### Requirement: Significance detection uses Int32Significance class
The decompiler SHALL use `Int32Significance` for construct type detection instead of ad-hoc bit masks.

#### Scenario: S1 countersign detection
- **WHEN** KLine has bit 63 set
- **THEN** decompiler identifies as S1 countersign and emits `==` operator

#### Scenario: S2 canonize detection
- **WHEN** KLine has any bit in range 55-62 set (and bit 63 clear)
- **THEN** decompiler identifies as S2 canonize and emits `=>` operator

#### Scenario: S3 connotate detection
- **WHEN** KLine has any bit in range 32-54 set (and bits 55-63 clear)
- **THEN** decompiler identifies as S3 connotate and emits `>` operator

#### Scenario: S4 undersign detection
- **WHEN** KLine has no significance bits set (bits 32-63 clear)
- **THEN** decompiler identifies as S4 undersign and emits `=` operator

### Requirement: Token mask updated to 32-bit boundary
The decompiler SHALL use token mask `(1 << 32) - 1` to strip significance bits.

#### Scenario: Strip significance for decoding
- **WHEN** decompiler decodes a signature
- **THEN** significance bits (32-63) are stripped, leaving only token bits (0-31)

#### Scenario: Preserve PACKED_BIT
- **WHEN** decompiler strips significance
- **THEN** bit 0 (PACKED_BIT) is preserved for packed/literal detection
