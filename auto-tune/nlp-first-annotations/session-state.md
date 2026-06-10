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
running

## Next Action
start run 1 (baseline — first-steps curriculum, no annotations yet)

## Patterns & Notes
- Block comment `(W1 W2 W3)` before multi-char sigs: positional zip binding
- Inline comment `S(ubject)` for single-char sigs
- Reference pattern: `data/example.ks`, `data/scripts/mw.ks`, `data/chats/example.ks`
- Binding resolver: `src/kscript/binding_resolver.py`
- AST emitter: `src/kscript/ast_emitter.py`
- NLPTokenizer: `src/kalvin/nlp_tokenizer.py`
- mhall-svo-single has runaway expansion — use first-steps for baseline

## Files Modified
(none yet)

## Run Log

(none yet)
