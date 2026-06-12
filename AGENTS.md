# Kalvin — Agent Instructions

## Git

- **Task work (kb tasks):** commit autonomously. No confirmation needed.
- **Ad-hoc work:** ask for explicit confirmation before running any `git commit`.

## Task Board

- Only create kb tasks when the user asks you to, or when work genuinely needs multi-step orchestration across sessions.
- Do not use the task board as a logging or tracking tool for work you are already doing.

## Tokenizers

**Never use `Mod32Tokenizer` as the main tokenizer for any script, application, or harness that accepts a tokenizer argument.**

The `Mod32Tokenizer` is a research tool designed for KScript and hand-crafting small knowledge graphs. Its 31-bit vocabulary and character-level encoding make it unsuitable for real workloads. When running a script or app that takes a tokenizer, always use `Mod64Tokenizer` (or the appropriate production tokenizer) unless the task is explicitly KScript research.

## Development

- Follow the cascade development model closely (docs/cascade-development.md): origin → spec → plan → triage. No content duplication across layers.
