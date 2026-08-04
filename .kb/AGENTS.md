# kb Agent Instructions

<!-- ─────────────────────────────────────────────────────────────────────────
   This file is read by EVERY kb agent (triage, executor, reviewer, merge) and
   appended to its system prompt. It is project-local kb guidance.

   How prompts reach kb agents (additive, in order):
     1. ~/.pi/agent/APPEND_SYSTEM.md        — global, all projects
     2. ./AGENTS.md | ./CLAUDE.md           — repo root (pi-native, walked up)
     3. ./.pi/APPEND_SYSTEM.md              — project (pi-native)
     4. ./.kb/AGENTS.md                     — THIS FILE (kb-patch addition)
     5. $KB_APPEND_SYSTEM (file or raw)     — explicit override

   Format notes:
   - Plain Markdown. It is concatenated after kb's own agent prompt.
   - Keep it short and behavioral — agents follow concrete rules best.
   - Use H2 sections so an agent can navigate to the relevant rule.
   - Prefer the imperative voice ("Do X", "Never Y").
   - This file REPLACES nothing; it only adds. kb's built-in prompts still run.
   ───────────────────────────────────────────────────────────────────────── -->

## Working style

- Prefer small, reviewable commits — one logical change per commit.
- When unsure about scope, write a follow-up task with `task_create` rather than expanding the current one.
- Read the relevant file before editing it; never guess at an API from memory.

## This project's conventions

- All commits are scoped by task ID: `feat(KB-012): ...`, `fix(KB-012): ...`.
- Keep changes inside the task's File Scope. If a change is needed outside it, stop and create a follow-up task.
- Documentation is part of done — if you changed behavior, note which doc layer it affects.

## Example: what an appended rule looks like

```markdown
## Testing

- Every implementation step writes real automated tests (assertions, not prints).
- Typechecks and builds are NOT tests.
- The final step runs the full suite — zero failures allowed.
```

That block, placed above, would be injected verbatim into each agent's system prompt.
