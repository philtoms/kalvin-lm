# Reactive Scaffolding Submission — Specification

## Overview

When the Trainer's LLM agent generates reactive scaffolding in response to S2/S3 events, the scaffolding must be compiled, validated, and submitted to Kalvin via the harness bus. Prior to this fix, the LLM agent produced scaffolding that was discarded due to three cascading failures in the prompt and sanitisation pipeline.

## Evidence

Auto-tune session: `auto-tune/reactive-scaffolding/`
- Run 1: LLM generated `#` comments → compilation failed → scaffolding discarded
- Run 2: LLM generated hex literals (`0x100`) → compilation failed → scaffolding discarded
- Run 3: LLM generated invalid operators (`~>`) → compilation failed → scaffolding discarded
- Run 4: Pipeline fixed → 2 scaffolding submissions confirmed in harness.log

## Root Causes

1. **Incorrect comment syntax in system prompt** (`cogitation.py`): Prompt claimed `Lines starting with # are comments` but KScript uses parenthesised comments `(...)`.
2. **Incorrect value syntax in system prompt** (`cogitation.py`): Prompt claimed `Signatures and nodes are hexadecimal integers prefixed with 0x` but KScript uses uppercase identifiers (A–Z only).
3. **Invalid operators in system prompt** (`cogitation.py`): Prompt listed `->`, `~>`, `<-` which don't exist in the KScript lexer. Only `==`, `=>`, `=`, `>` are valid.
4. **Hex repr in misfit summaries** (`trainer.py`): The `_cogitate_adapter` passed `repr(event.query)` as the expectation summary — raw hex like `KLine(sig=0x2000, nodes=[256])`. The LLM had no way to know the name mapping.
5. **No defensive comment stripping** (`cogitation.py`): Even with a corrected prompt, some LLMs may still produce `#` comments. The Cogitator did not sanitise before compilation.

## Fixes Applied

### RS-1: System prompt uses correct KScript syntax

The `_SYSTEM_PROMPT` in `cogitation.py` now documents:
- Identifiers: uppercase letters only (A–Z), no hex literals
- Operators: `==` (countersign), `=>` (canonize), `=` (undersign), `>` (relationship)
- Comments: parenthesised `(...)` only
- Explicitly lists forbidden patterns: no hex, no `~>`, no `<-`, no `->`

### RS-2: Misfit summaries are decompiled to KScript

The `_cogitate_adapter` in `trainer.py` decompiles event query and proposal klines to human-readable KScript (e.g., `M > H`) before passing them as `expectation_summary` and `proposal_summary` to the `MisfitInfo`. This gives the LLM the name context it needs to generate valid scaffolding.

### RS-3: Hash-comment stripping before compilation

`_strip_hash_comments()` in `cogitation.py` removes lines starting with `#` from scaffolding before compilation. This is a defensive measure — the system prompt no longer mentions `#`, but LLMs trained on the old prompt or defaulting to Python-style comments will still produce them.

### RS-4: "submitted reactive scaffolding" log line

The Reactor logs `"submitted reactive scaffolding"` at INFO level after successfully sending reactive scaffolding to the trainee role. This provides an observable confirmation in harness.log that the pipeline is working end-to-end.

## Behavioural Rules

1. The Cogitator must sanitise scaffolding source by removing `#`-prefixed comment lines before attempting compilation.
2. If sanitisation removes all content (scaffolding was all comments), the Cogitator returns `scaffolding=None` without attempting compilation.
3. The system prompt must only document operators that the KScript lexer actually supports.
4. Misfit summaries passed to the LLM must use decompiled KScript, not hex repr.
5. The Reactor must log "submitted reactive scaffolding" after sending scaffolding to the trainee role.

## Test Matrix

| ID  | Criterion | Status |
|-----|-----------|--------|
| RS-1 | System prompt contains no hex literal syntax | ✅ |
| RS-2 | System prompt contains no invalid operators (~>, <-, ->) | ✅ |
| RS-3 | `_strip_hash_comments` removes # lines, keeps valid KScript | ✅ |
| RS-4 | `_strip_hash_comments` returns empty string for all-comment input | ✅ |
| RS-5 | `_cogitate_adapter` decompiles query kline to KScript | ✅ |
| RS-6 | `_cogitate_adapter` decompiles proposal kline to KScript | ✅ |
| RS-7 | `_cogitate_adapter` falls back to repr on decompilation failure | ✅ |
| RS-8 | Cogitator logs stripped comment count when # comments removed | ✅ |
| RS-9 | Cogitator returns None when scaffolding is all comments | ✅ |
| RS-10 | Reactor logs "submitted reactive scaffolding" after bus send | ✅ |
| RS-11 | All existing cogitation tests still pass | ✅ |
| RS-12 | All existing reactor tests still pass | ✅ |
| RS-13 | All existing trainer tests still pass | ✅ |
