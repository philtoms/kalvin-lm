# Kalvin — Agent Instructions

## Git

- Ask for explicit confirmation before running any `git commit` unless the work is contained in a git worktree.

## Task Board

- Only create kb tasks when the user asks you to.

## Coding activity

- read CONTEXT.md when you need to understand the domain.
- read docs/kalvin-vision.md when feel you do not quite understand the user's request.
- DocStrings and comments should be kept to the barest minimum. No repetition of the code and no historical connotations - just describe what is.
- Source is the truth document. When changing behaviour, update the source and CONTEXT.md (the Project Navigation section and any affected glossary terms) in the same change.
