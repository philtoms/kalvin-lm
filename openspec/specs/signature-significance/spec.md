# Spec: Signature Significance

## Purpose

Defines the bit allocation strategy for 64-bit signatures, reserving upper bits for significance encoding while confining character tokenization to lower bits.

## Requirements

### Requirement: Bit Allocation
The system SHALL allocate the 64-bit signature space as follows:

| Bits | Purpose |
|------|---------|
| 0 | PACKED_BIT flag (1 = packed, 0 = literal) |
| 1-32 | Character tokenization (Mod32Tokenizer) |
| 33-63 | Reserved for significance encoding |

#### Scenario: Character encoding in lower bits
- **WHEN** a signature is tokenized
- **THEN** all character bits are set in positions 1-32
- **AND** bits 33-63 remain clear (zero)

#### Scenario: PACKED_BIT at position 0
- **WHEN** a packed signature is encoded
- **THEN** bit 0 is set to 1
- **AND** character bits occupy positions 1-32

---

### Requirement: Significance Space Reservation
The system SHALL reserve bits 33-63 for future significance encoding.

#### Scenario: Upper bits available
- **WHEN** significance encoding is implemented
- **THEN** bits 33-63 are available for priority levels
- **AND** no character tokenization uses these bits

#### Scenario: Future significance levels
- **WHEN** significance is encoded in the future
- **THEN** S1-S4 levels can be stored in upper 32 bits
- **AND** character encoding remains in lower 32 bits
