# Kalvin — Agent Instructions

## Git

- **Task work (kb tasks):** commit autonomously. No confirmation needed.
- **Ad-hoc work:** ask for explicit confirmation before running any `git commit`.
- **Classification rule:** work is "task work" only when the user explicitly put it through the task board. If you created the task yourself on a direct instruction, it is ad-hoc work — do not use the task as justification for autonomous commit.

## Task Board

- Only create kb tasks when the user asks you to, or when work genuinely needs multi-step orchestration across sessions.
- Direct instructions ("do X", "add feature Y") are ad-hoc work. Do not create a task for them.
- Do not use the task board as a logging or tracking tool for work you are already doing.

## Development

- Follow the cascade development model closely (docs/cascade-development.md): origin → spec → plan → triage. No content duplication across layers.
