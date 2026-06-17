# Kalvin — Agent Instructions

## Git

- Ask for explicit confirmation before running any `git commit` unless the work is contained in a git worktree.

## Task Board

- Only create kb tasks when the user asks you to.

## Coding activity

- **Every ad-hoc coding request leads to a documentation maintenance phase.** An ad-hoc request is any code change that does not go through the full cascade (e.g. bug fixes, tweaks, small features, refactors, debugging). After the code change is complete ask user before:
  1. **Locate** — find the cascade layer the change relates to: check `docs/kalvin-vision.md`, `specs/`, and `plans/` for affected concepts, contracts, or file locations.
  2. **Assess** — determine whether the change contradicts, extends, or is already covered by the existing docs.
  3. **Update only if needed** — edit the single owning layer (per the cascade's content-ownership table). Do not duplicate content across layers; cross-reference instead. If the docs already accurately describe the new state, make no change.
  4. **Report** — state which doc(s) you checked and what (if anything) you changed.
