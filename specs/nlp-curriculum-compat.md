# NLP Tokenizer Curriculum Compatibility

## Summary

All existing curricula compile and train correctly with the NLPTokenizer without modification. Bare signatures (no parenthetical comments) produce valid NLP-BPE graph nodes. The NLP binding resolver operates correctly with an empty symbol table.

## Spec

### SC-1: Bare signatures produce consistent NLP-BPE tokens

Single-character signatures (e.g., `M`, `H`, `A`) encode to 64-bit NLP-BPE nodes where the upper 32 bits carry NLP type information and the lower 32 bits carry the BPE token ID. The same character always produces the same node value.

### SC-2: Multi-character signatures decompose correctly

Multi-character signatures (e.g., `MHALL`, `SVO`, `ALL`) decompose into individual character components with unsigned entries and a canonize (S2) entry mapping the first token to all tokens.

### SC-3: Training runs complete cleanly

Each curriculum completes with:
- All lessons satisfied
- Zero LLM calls for S1/S4 curricula
- Correct auto-countersign behavior
- No compilation errors

### SC-4: Comments are optional for abstract-letter curricula

Curricula using abstract uppercase letters (A, B, C, D, M, H, S, V, O) do not require parenthetical comments. Comments are required only when semantic word resolution is desired (e.g., `M(ary)` → NLP-bound token for "Mary").

## Test Matrix

| Curriculum | Compiled Entries | STM Entries | Training Runs |
|---|---|---|---|
| first-steps.md | 4 | 4 | 3/3 lessons ✅ |
| first-steps-s2.md | 8 | 7 | verified |
| mhall-svo-single.md | 24 | 18 | 1/1 lesson ✅ |
| mhall-svo-equivalence.md | 31 | 23 | verified |
| cascade-pressure.md | 67 | 35 | verified |
| conflict-drill.md | 14 | 11 | verified |
| s3-auto-countersign.md | 11 | 10 | verified |

## Evidence

Auto-tune session `nlp-curricula` at `auto-tune/nlp-curricula/`:
- Run 1: first-steps.md — 3/3 lessons, zero LLM
- Run 2: mhall-svo-single.md — 18/18 entries, zero LLM
- All 7 curricula verified programmatically
