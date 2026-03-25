## Context

The Kalvin model uses 64-bit signatures for KLine identification. Currently, `Mod64Tokenizer` maps characters to bit positions using modulo 64 arithmetic, consuming the full 64-bit space (bits 1-64, with bit 0 reserved for PACKED_BIT).

The significance system (defined in `src/kalvin/significance.py`) needs space in the upper bits to encode match priority levels (S1-S4). This requires reserving bits 33-63 for significance, leaving only bits 1-32 for character tokenization.

## Goals / Non-Goals

**Goals:**
- Switch default tokenizer from `Mod64Tokenizer` to `Mod32Tokenizer`
- Reserve upper 32 bits (33-63) for future significance encoding
- Maintain backward compatibility for JSON/JSONL formats

**Non-Goals:**
- Implementing significance encoding itself (future change)
- Migrating existing binary files
- Changing the tokenizer API or interface

## Decisions

### Decision 1: Use Mod32Tokenizer as default

**Rationale:** Mod32Tokenizer uses `(ord(char) % 32) + 1` for bit positions, confining character encoding to bits 1-32. This leaves bits 33-63 available for significance.

**Alternatives considered:**
- **Mod16Tokenizer**: Would leave more room (48 bits) but increases collision rate unacceptably (A collides with Q, etc.)
- **Hybrid approach**: Encode significance in separate field - rejected because it breaks the 64-bit signature contract

**Character collision analysis (mod 32):**
```
A-Z: positions 1-26 (no collisions within alphabet)
0-9: positions 27-31, 1-5 (digits collide with A-E - acceptable)
```

### Decision 2: Update all three tokenizer references

**Rationale:** Consistency requires updating all default tokenizer instantiations:
1. `KScript.__init__` (API usage)
2. `Compiler.__init__` (internal compilation)
3. CLI default in `__main__.py`

**Alternatives considered:**
- **Single source of truth**: Could refactor to use a module-level constant, but over-engineering for a simple swap

### Decision 3: No binary migration tool

**Rationale:** Binary files are an internal format, not a public API. Users should recompile from `.ks` sources. JSON/JSONL files are the interchange format and remain compatible.

**Alternatives considered:**
- **Migration script**: Would add maintenance burden for minimal benefit

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Existing `.bin` files become unreadable | Document breaking change; users recompile from source |
| Digit collisions with letters (0-4 collides with A-E) | Acceptable for signatures (typically uppercase letters) |
| Future code may accidentally use upper bits | Add documentation/comments marking bits 33-63 as reserved |

## Migration Plan

1. Update tokenizer defaults in all three locations
2. Run test suite to verify no regressions
3. Update CLAUDE.md to document bit allocation

**Rollback:** Simple git revert of the three file changes.

## Open Questions

- Should we add a runtime check/warning when decoding with wrong tokenizer? (Deferred - low priority)
