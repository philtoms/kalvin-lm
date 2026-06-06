# Reactive Scaffolding Submission — Implementation Plan

## Evidence

Auto-tune session: `auto-tune/reactive-scaffolding/` (3 runs, runs/001 through runs/003)

## Changes Made

### 1. `src/trainer/cogitation.py` — System prompt and sanitisation

- Rewrote `_SYSTEM_PROMPT` to document correct KScript syntax (uppercase names, valid operators only, parenthesised comments)
- Added `_strip_hash_comments()` function for defensive comment removal
- Modified `Cogitator.cogitate()` to strip `#` comments before compilation and handle all-comment scaffolding

### 2. `src/trainer/trainer.py` — Decompilation in cogitate adapter

- Added `Decompiler` import and instance in `_cogitate_adapter`
- Decompile `event.query` and `event.proposal` to KScript before passing to `MisfitInfo`
- Log decompiled source alongside raw hex repr

### 3. `src/trainer/reactor.py` — Observable log line

- Added `logger.info("submitted reactive scaffolding")` after reactive scaffolding bus send

## Test Mapping

| ID  | Test | File | Status |
|-----|------|------|--------|
| RS-1 | test_system_prompt_no_hex | test_reactive_scaffolding.py | ✅ |
| RS-2 | test_system_prompt_no_invalid_operators | test_reactive_scaffolding.py | ✅ |
| RS-3 | test_strip_hash_comments_removes_comments | test_reactive_scaffolding.py | ✅ |
| RS-4 | test_strip_hash_comments_all_comments | test_reactive_scaffolding.py | ✅ |
| RS-5 | test_strip_hash_comments_preserves_kscript | test_reactive_scaffolding.py | ✅ |
| RS-6 | test_cogitate_adapter_decompiles_query | test_reactive_scaffolding.py | ✅ |
| RS-7 | test_cogitate_adapter_decompiles_proposal | test_reactive_scaffolding.py | ✅ |
| RS-8 | test_cogitate_adapter_fallback_on_decompile_error | test_reactive_scaffolding.py | ✅ |
| RS-9 | test_cogitator_strips_and_logs | test_reactive_scaffolding.py | ✅ |
| RS-10 | test_cogitator_all_comments_returns_none | test_reactive_scaffolding.py | ✅ |
| RS-11 | test_reactor_submitted_log_line | test_reactive_scaffolding.py | ✅ |
| RS-12 | All existing cogitation tests | test_cogitation.py | ✅ |
| RS-13 | All existing reactor tests | test_reactor.py | ✅ |
| RS-14 | All existing trainer tests | test_trainer.py | ✅ |

## Design Decisions

1. **Strip `#` comments defensively**: Even with the corrected prompt, some LLMs will produce `#` comments. Rather than failing, we strip them and log the fact. This makes the pipeline robust against LLM variation.

2. **Decompile at adapter level, not in cogitation module**: The `_cogitate_adapter` in `trainer.py` is the bridge between the event world (hex klines) and the cogitation world (text prompts). Decompilation belongs here, not in the cogitation module itself, which should remain agnostic about where its text comes from.

3. **Log line wording**: "submitted reactive scaffolding" is deliberately present tense and active — it confirms an action taken, not a state observed.
