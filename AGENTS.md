# Kalvin — Agent Instructions

## Git

- Ask for explicit confirmation before running any `git commit` unless the work is contained in a git worktree.

## Coding activity

- DocStrings should be kept to the barest minimum. Truth is in the source, not the comments.

## Thinking discipline

- docs/rationalisation.md is the rules document for the model — read it before changing src/rationalisation. docs/attic/ holds retired history, not truth.
- src/rationalisation is the truth document for current behaviour only - not future behaviour.
- read CONTEXT.md only when you need to navigate the domain.
- If you have questions or gaps - stop and ask. Do not keep going.
- Do not revise just to agree. If the user pushes back without giving new evidence or a specific error, do not apologize, do not flip, and do not say "you are right".
