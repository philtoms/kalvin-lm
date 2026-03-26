## Context

The 64-bit signature is divided into two regions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         64-bit Signature Layout                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SIGNIFICANCE (bits 32-63, shifted from 32-bit value):                    │
│   ├─ Bit 63       │ S1 indicator (shifted from bit 31)                     │
│   ├─ Bits 56-62   │ S2 degree (7 bits, shifted from 24-30)                 │
│   ├─ Bit 55       │ S2 indicator (shifted from bit 23)                     │
│   ├─ Bits 33-54   │ S3 degree (22 bits, shifted from 1-22)                 │
│   └─ Bit 32       │ S3 indicator (shifted from bit 0)                      │
│                                                                             │
│   TOKEN SPACE (bits 0-31, unchanged):                                       │
│   ├─ Bit 0       │ PACKED_BIT                                              │
│   └─ Bits 1-31   │ Character tokenization (31 char bits)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Goals / Non-Goals

**Goals:**
- Clean separation: tokens in bits 0-31, significance in bits 32-63
- Graded significance with appropriate granularity per level
- Natural mapping to KScript construct types
- Add `Int32Significance` class for decompiler use

**Non-Goals:**
- Changing token encoding (Mod32Tokenizer unchanged)
- Changing KScript syntax or semantics
- Replacing `Int64Significance` (currently unused, left as-is)

## Decisions

### D1: 32-bit Significance Layout

**Decision:** 32-bit significance value shifted left by 32 to occupy bits 32-63.

```
32-bit layout (before shift):
┌────────────────────────────────────────┐
│ Bit 31    │ S1 (single bit)            │
│ Bits 23-30│ S2 range (8 bits)          │
│ Bits 0-22 │ S3 range (22 bits)         │
│ All clear │ S4 (no significance)       │
└────────────────────────────────────────┘
```

**Rationale:**
- S1 is binary (highest significance or not)
- S2 is 8 bits for near neighbors (coarse granularity)
- S3 is 22 bits for far neighbors (fine granularity)
- As distance increases, granularity becomes finer

### D2: Instance-Level Constants (Shifted)

**Decision:** Constants stored as 64-bit values (32-bit significance << 32).

```python
class Int32Significance:
    # 32-bit values (conceptual)
    _S1_32 = 1 << 31
    _S2_IND_32 = 1 << 23
    _S3_IND_32 = 1 << 0

    # Stored as 64-bit (shifted left 32)
    S1 = _S1_32 << 32      # bit 63
    S2_IND = _S2_IND_32 << 32  # bit 55
    S3_IND = _S3_IND_32 << 32  # bit 32
```

**Rationale:**
- Constants are directly usable with 64-bit signatures
- No shifting needed at call sites
- Clear mapping from 32-bit concept to 64-bit storage

### D3: Hierarchical Detection

**Decision:** Check bits top-down to determine significance level.

```python
def get_level(self, sig: int) -> str:
    if sig & self.S1:           # bit 63 set
        return "S1"
    elif sig & self.S2_RANGE:   # any bit 55-62 set
        return "S2"
    elif sig & self.S3_RANGE:   # any bit 32-54 set
        return "S3"
    else:
        return "S4"             # all significance bits clear
```

**Rationale:**
- Higher bits take precedence
- Simpler than checking each range independently
- Naturally ordered: S1 > S2 > S3 > S4

### D4: Construct Type Mapping

**Decision:** Significance levels map directly to construct operators.

| Significance | Construct | Operator | Semantic |
|--------------|-----------|----------|----------|
| S1 | countersign | `==` | Direct bidirectional link |
| S2 | canonize | `=>` | Near neighbor composition |
| S3 | connotate | `>` | Far neighbor annotation |
| S4 | undersign | `=` | Unrecognized but accepted |

**Rationale:**
- Countersign is the strongest relationship (S1)
- Canonize composes near neighbors (S2)
- Connotate annotates at greater distance (S3)
- Undersign accepts existence without recognition (S4)

**Terminology:**
- **Undersign** = the S4 construct (operator `=`)
- **Identity kline** = result of undersign construct (rationalizes to itself)
- **S4 identity semantic**: "I do not recognize this kline - but I accept that it exists!"

### D5: Near Neighbor Semantics

**Decision:** Significance ranges encode mathematical "near neighbor" distance.

```
   S1        S2 (8 bits)           S3 (22 bits)
   │         │                     │
   ▼         ▼                     ▼
┌──────┬────────────┬──────────────────────────────┐
│  ●   │  ░░░░░░░░  │  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │
│      │  coarse    │         fine                 │
└──────┴────────────┴──────────────────────────────┘

   │←─ near ──→│←─────── far ────────────→│
       8 bits           22 bits
```

**Rationale:**
- Near neighbors cluster coarsely (fewer bits needed)
- Far neighbors spread finely (more bits to distinguish)
- As distance increases, you need more granularity

## Implementation Notes

### Token Mask

Old: `TOKEN_MASK = (1 << 61) - 1`
New: `TOKEN_MASK = (1 << 32) - 1`

This strips significance bits and keeps only token bits 0-31.

### Significance Masks

```python
# Range masks (64-bit, shifted)
S2_RANGE = 0x7F00_0000_0000_0000  # bits 56-62 (7 bits for degree)
S3_RANGE = 0x007F_FFFF_0000_0000  # bits 33-54 (22 bits for degree)
SIG_MASK = 0xFFFF_FFFF_0000_0000  # bits 32-63 (all significance)
```

### Extracting Significance Value

```python
sig_value = (signature >> 32) & 0xFFFF_FFFF
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Breaking existing compiled KScripts | Accept as MVP - recompile from source |
| S2 range smaller than old Int64Significance | 8 bits sufficient for near neighbor degree |
| Bit manipulation errors | Comprehensive tests for all edge cases |
