# NLP-First Curriculum Annotations — Implementation Plan

**Parent:** NLP tokenisation pipeline
**Status:** ✅ complete
**Spec refs:** `specs/nlp-first-curriculum-annotations.md`
**Evidence:** `auto-tune/nlp-first-annotations/`

## Spec References

- `@specs/nlp-first-curriculum-annotations` — NA-1 through NA-8
- `@specs/kscript-nlp-binding` — word list syntax and claiming rules
- `@specs/nlp-curriculum-compat` — backwards compatibility (bare sigs still work)

## Implementation Tasks

### Task 1: Annotate first-steps.md ✅

- **Spec ref:** @specs/nlp-first-curriculum-annotations NA-1, NA-5
- **Changes:** `M` → `M(ark)`, `H` → `H(alo)`, `M == H` → `M(ark) == H(alo)`
- **Verification:** Training run 3/3 lessons, 4/4 entries, zero LLM

### Task 2: Annotate first-steps-s2.md ✅

- **Spec ref:** @specs/nlp-first-curriculum-annotations NA-1, NA-2, NA-3
- **Changes:** inline on M/H/A, block `(Mark Halo)` on `MH`, inline on H and A in `MH => H A(lpha)`
- **Verification:** test_nlp_curriculum_compat passes

### Task 3: Annotate mhall-svo-single.md ✅

- **Spec ref:** @specs/nlp-first-curriculum-annotations NA-1, NA-2, NA-3, NA-4
- **Changes:** block `(Mary Had A Little Lamb)` on MHALL, inline S(ubject)/V(erb)/O(bject)/M(ary)/H(ad)/A(ll)/D(et)/L/M(od)/L/O on subscript lines
- **Verification:** test_nlp_curriculum_compat passes

### Task 4: Annotate mhall-svo-equivalence.md ✅

- **Spec ref:** @specs/nlp-first-curriculum-annotations NA-1, NA-2, NA-3, NA-4
- **Changes:** inline on all identity atoms (M(ary), H(ad), L(ittle), S(ubject), V(erb), O(bject)), block (A L L) on ALL, inline on M=S/H=V/ALL=O lesson, block + inline on final MHALL lesson
- **Verification:** test_nlp_curriculum_compat passes

### Task 5: Annotate cascade-pressure.md ✅

- **Spec ref:** @specs/nlp-first-curriculum-annotations NA-1, NA-2, NA-3, NA-4
- **Changes:** NATO phonetic inline (Alpha through Juliet), block comments on all multi-char sigs (AB, CD, EF, GH, ABCD, EFGH, ACEGI, ABCDEFGHIJ), inline on all right-side nodes
- **Verification:** test_nlp_curriculum_compat passes

### Task 6: Annotate conflict-drill.md ✅

- **Spec ref:** @specs/nlp-first-curriculum-annotations NA-1, NA-2, NA-3, NA-4
- **Changes:** NATO phonetic inline (Alpha through Echo), block (Alpha Beta) on AB, block (Alpha Charlie Echo) on ACE, inline on right-side nodes
- **Verification:** test_nlp_curriculum_compat passes

### Task 7: Annotate s3-auto-countersign.md ✅

- **Spec ref:** @specs/nlp-first-curriculum-annotations NA-1, NA-2, NA-3, NA-4
- **Changes:** inline M(ark)/H(alo)/A(lpha)/P(apa)/X(ray), block (Hotel Papa Alpha) on HPA, inline A(lpha) on `A == A`, inline P(apa)/X(ray) on `HPA => P X`
- **Verification:** test_nlp_curriculum_compat passes

## Test Mapping

| Spec ID | Test file | Test function | Status |
|---------|-----------|---------------|--------|
| NA-6 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_compiles[first-steps.md] | ✅ |
| NA-6 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_compiles[first-steps-s2.md] | ✅ |
| NA-6 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_compiles[mhall-svo-single.md] | ✅ |
| NA-6 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_compiles[mhall-svo-equivalence.md] | ✅ |
| NA-6 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_compiles[cascade-pressure.md] | ✅ |
| NA-6 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_compiles[conflict-drill.md] | ✅ |
| NA-6 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_compiles[s3-auto-countersign.md] | ✅ |
| NA-7 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_rationalizes[first-steps.md] | ✅ |
| NA-7 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_rationalizes[first-steps-s2.md] | ✅ |
| NA-7 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_rationalizes[mhall-svo-single.md] | ✅ |
| NA-7 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_rationalizes[mhall-svo-equivalence.md] | ✅ |
| NA-7 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_rationalizes[cascade-pressure.md] | ✅ |
| NA-7 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_rationalizes[conflict-drill.md] | ✅ |
| NA-7 | test_nlp_curriculum_compat.py | TestCurriculumCompilation::test_curriculum_rationalizes[s3-auto-countersign.md] | ✅ |
| NA-8 | test_nlp_curriculum_compat.py | All 21 tests | ✅ |

## Design Decisions

### DD-1: NATO phonetic for abstract-sig curricula

**Decision:** Use NATO phonetic alphabet words (Alpha, Beta, Charlie…) for curricula with abstract letter sigs (cascade-pressure, conflict-drill).
**Rationale:** Semantically meaningful nouns that produce clean NLP type bits. Consistent across curricula — A is always Alpha. Alternative was descriptive names (e.g., `A(tom)`, `B(ond)`), but NATO is standard and unambiguous.

### DD-2: Word theme consistency within curricula

**Decision:** Each curriculum uses a single word theme — identity names (first-steps, first-steps-s2), MHALL nursery rhyme (mhall-svo-*), NATO phonetic (cascade-pressure, conflict-drill).
**Rationale:** Mixed themes within a curriculum would be confusing. Consistent themes make the curricula self-documenting.

### DD-3: Block comments only where multi-char sigs are defined

**Decision:** Block comments placed only on the first appearance of a multi-char sig in a code block, not on every reference.
**Rationale:** The binding resolver processes comments per-scope. A block comment before the definition sig is sufficient — subsequent uses of the same sig within that scope inherit the binding via upward traversal.

## Status

| Task | Status | Notes |
|------|--------|-------|
| 1 | ✅ done | first-steps.md |
| 2 | ✅ done | first-steps-s2.md |
| 3 | ✅ done | mhall-svo-single.md |
| 4 | ✅ done | mhall-svo-equivalence.md |
| 5 | ✅ done | cascade-pressure.md |
| 6 | ✅ done | conflict-drill.md |
| 7 | ✅ done | s3-auto-countersign.md |
