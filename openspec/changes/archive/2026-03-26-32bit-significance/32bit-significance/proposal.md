## Why

The current decompiler significance encoding uses ad-hoc bit positions (63, 62, 61) that conflict with the 64-bit signature layout comments and don't align with the semantic significance model in `Int64Significance`.

We need a unified 32-bit significance system that:
- Lives cleanly in the upper 32 bits of the 64-bit signature (no token conflict)
- Maps naturally to construct types (countersign, canonize, connotate, undersign)
- Encodes graded significance with appropriate granularity per level

## What Changes

- New `Int32Significance` class replaces decompiler's ad-hoc detection (Int64SIgnificance is unchanged and currently unused)
- Module-level constants replaced by instance level constants (32 bits shifted left into 64 bits)
- Significance bits shifted to positions 32-63 of 64-bit signature
- Token space remains bits 0-31 (unchanged)
- Decompiler updated to use new significance detection

## Capabilities

### Modified Capabilities

- `kscript-compiler`: Significance encoding moves to bits 32-63
- `kscript-decompiler`: Uses new `Int32Significance` for construct type detection

## Impact

- `src/kalvin/significance.py`: Add `Int32Significance`
- `src/kscript/decompiler.py`: Update bit masks and detection logic
- `src/kscript/compiler.py`: Update significance bit setting
- Tests: Update all significance-related tests