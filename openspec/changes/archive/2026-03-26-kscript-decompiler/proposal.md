## Why

Kalvin outputs ordered KLines with significance bits encoding construct types. During development, we need to inspect these KLines in human-readable form. Currently there's no way to convert KLines back to KScript source, making debugging difficult.

This change introduces a decompiler that reverses the compilation process: ordered KLines → KScript source.

## What Changes

- New `Decompiler` class that converts ordered KLines to KScript source
- Support for all significance levels (always set, S4 = no bits):
  - S1 (bit 63): countersign (`==`)
  - S2 (bit 62): canonize (`=>`)
  - S3 (bit 61): connotate (`>`)
  - S4 (no bits): undersign identity (`X` for `{X: None}`)
- Subscript reconstruction for multi-node canonize constructs
- Identity recovery for nodes without entries
- Error surfacing for invalid/malformed KLines (development tool mentality)

## Capabilities

### New Capabilities

- `kscript-decompiler`: Decompiler that converts ordered KLines with significance bits back to KScript source code

### Modified Capabilities

(None - this is a new module with no requirement changes to existing specs)

## Impact

- New module: `src/kscript/decompiler.py`
- New API: `Decompiler` class with `decompile(klines, tokenizer) -> str`
- Depends on: existing `CompiledEntry.decode()` for token ID → string conversion
- Depends on: significance bit positions (S1=bit 63, S2=bit 62, S3=bit 61, S4=none)
