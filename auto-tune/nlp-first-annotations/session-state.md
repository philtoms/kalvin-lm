# Auto-Tune Session State

## Goal
Rewrite all 7 curriculum files to be NLP-first: every signature gets a parenthetical comment. Block comments for multi-char sigs, inline for single-char. Standalone scripts already done.

## Done Criteria
- All 7 curricula have NLP annotations on every signature
- Each curriculum compiles and trains cleanly with NLPTokenizer
- All 21 existing tests in `tests/test_nlp_curriculum_compat.py` pass
- No errors or warnings in harness.log

## Session
- **Name:** nlp-first-annotations
- **Curriculum:** curricula/first-steps.md
- **Branch:** auto-tune/nlp-first-annotations
- **Worktree:** .worktrees/auto-tune/nlp-first-annotations
- **Started:** 2026-06-10

## Current Phase
complete

## Next Action
Goal met. Ready for cascade documentation if desired. Changes on branch `auto-tune/nlp-first-annotations` — do NOT merge to main without review.

## Patterns & Notes
- Block comment `(W1 W2 W3)` before multi-char sigs: positional zip binding
- Inline comment `S(ubject)` for single-char sigs
- Reference pattern: `data/example.ks`, `data/scripts/mw.ks`, `data/chats/example.ks`
- Binding resolver: `src/kscript/binding_resolver.py`
- AST emitter: `src/kscript/ast_emitter.py`
- NLPTokenizer: `src/kalvin/nlp_tokenizer.py`
- data/tokenizer is gitignored — need symlink for tests in worktree

## Files Modified
- `curricula/first-steps.md` — inline M(ark), H(alo)
- `curricula/first-steps-s2.md` — inline + block (Mark Halo) on multi-char sig
- `curricula/mhall-svo-single.md` — block (Mary Had A Little Lamb) + inline
- `curricula/mhall-svo-equivalence.md` — block + inline across all 5 lessons
- `curricula/cascade-pressure.md` — NATO phonetic annotations (Alpha–Juliet)
- `curricula/conflict-drill.md` — inline Alpha/Beta/Charlie/Delta/Echo
- `curricula/s3-auto-countersign.md` — inline Mark/Halo/Alpha/Papa/X-ray

## Run Log

### Run 1 (latest) — annotated first-steps ✅
- **Code changes:** All 7 curricula annotated with NLP-first comments
- **Observations:** Training with first-steps: 3/3 lessons, 4/4 entries satisfied, zero LLM, zero escalations. All 21 tests pass. Clean harness.log.
- **Verdict:** goal met
