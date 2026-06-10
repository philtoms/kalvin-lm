# NLP-First Curriculum Annotations

## Summary

All curriculum files use NLP-annotated signatures: every single-character signature has an inline parenthetical comment, and every multi-character signature has a block comment word list preceding it. This makes curricula self-documenting and ensures NLPTokenizer produces semantically rich graph nodes rather than abstract-letter fallbacks.

## Dependencies

- **@specs/kscript-nlp-binding** — NLP word list syntax, claiming rules, binding mechanisms
- **@specs/nlp-curriculum-compat** — bare signatures still compile (backwards compatibility)

## Annotation Rules

### NA-1: Inline comments on single-character signatures

Every single-character signature in a KScript code block must carry an inline parenthetical comment: `M(ark)`, `H(alo)`, `S(ubject)`, etc. The comment text, combined with the signature character, forms the full NLP word.

### NA-2: Block comments on multi-character signatures

Every multi-character signature must be preceded by a block comment word list where the word count matches the character count. Example: `(Mary Had A Little Lamb)` before `MHALL`.

### NA-3: Inline comments on right-side nodes

Node-position single-character signatures in operators receive inline comments **only when the binding would introduce new information**. If the character is already bound to the same word in the scope chain, the inline annotation is redundant and must be omitted. Example: in `(Mary Had A Little Lamb)` followed by `MHALL == SVO => S(ubject) = M(ary)`, the `M(ary)` is redundant because M is already bound to "Mary" via the block comment. The correct form is `S(ubject) = M` — the node inherits M→"Mary" from the parent scope.

Inline annotations on node-position signatures that **intentionally shadow** a parent binding with a different word are required. Example: `O(bject) = A(ll)` where A is bound to "a" in the parent scope but "All" in the subscript scope.

### NA-4: Semantic word choices

Word choices should be semantically meaningful — proper nouns, common nouns, or NATO phonetic alphabet — so that NLP-BPE tokens carry grammatically useful type bits. Abstract letter names (e.g., `A(lpha)`) are acceptable for curricula that use abstract sigs.

### NA-5: No annotation on prose

NLP annotations appear only inside fenced KScript code blocks. The curriculum prose (lesson descriptions, objective, approach) is not annotated.

## Per-Curriculum Annotation Map

| Curriculum | Sigs | Block comments | Inline comments | Word theme |
|---|---|---|---|---|
| first-steps | M, H | — | M(ark), H(alo) | Identity names |
| first-steps-s2 | M, H, A, MH | (Mark Halo) on MH | M(ark), H(alo), A(lpha) | Identity names |
| mhall-svo-single | MHALL, SVO, M, H, A, L, S, V, O, D | (Mary Had A Little Lamb) on MHALL | All single chars | MHALL nursery rhyme |
| mhall-svo-equivalence | M, H, A, L, S, V, O, ALL, MHALL, SVO | (Mary Had A Little Lamb) on MHALL, (A L L) on ALL | All single chars | MHALL + grammatical roles |
| cascade-pressure | A–J, AB, CD, EF, GH, ABCD, EFGH, ACEGI, ABCDEFGHIJ | (Alpha Beta) on AB, etc. | NATO phonetic on all | NATO phonetic alphabet |
| conflict-drill | A–E, AB, ACE | (Alpha Beta) on AB, (Alpha Charlie Echo) on ACE | NATO phonetic on all | NATO phonetic alphabet |
| s3-auto-countersign | M, H, A, P, X, HPA | (Hotel Papa Alpha) on HPA | Mark, Halo, Alpha, Papa, X-ray | Mixed (NATO + identity) |

## Verification

### NA-6: Compilation

Each annotated curriculum compiles without errors through the KScript compiler with NLPTokenizer active.

### NA-7: Rationalisation

Each annotated curriculum rationalises correctly — the agent produces the expected number of satisfied entries with no regressions compared to bare-sig runs.

### NA-8: Existing tests pass

All 21 tests in `tests/test_nlp_curriculum_compat.py` continue to pass. The annotations must not break bare-sig compatibility.

### NA-9: Redundancy detection

The BindingResolver detects and skips inline annotations that would redundantly re-bind a character to the same word already present in the scope chain. This is implemented via `NLPSymbolTable.is_bound_to(char, word)`. Redundant bindings are silently skipped with a debug-level log message. This applies to both sig-side and node-side inline comments.

## Out of Scope

- Standalone KScript files (`data/example.ks`, `data/scripts/mw.ks`, `data/chats/example.ks`) — already annotated.
- Changes to the KScript compiler, binding resolver, or NLPTokenizer — no code changes required.
- Mod32 compilation — annotations are inert in Mod32 mode.
