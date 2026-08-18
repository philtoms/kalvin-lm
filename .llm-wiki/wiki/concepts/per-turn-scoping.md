---
type: concept
title: Per-turn scoping
description: observations and _incoming reset to fresh lists at the top of every rationalise() call — each turn's emissions are independent of prior turns'.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Per-turn scoping

Each `rationalise()` call starts clean.

## Definition

`observations` and `_incoming` reset to fresh lists at the top of every
`rationalise()` call. The engine is stateless about its own emissions within a
turn — dedup lives in the actor, not the engine. State that persists across
turns lives in [[concepts/frame]] and [[entities/ltm-long-term-memory|LTM]], not
in the per-turn registers.

This is what lets the [[concepts/non-judging-harness]] present each turn's trace
as an independent unit.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — the scoping rule
- [[entities/rationalise]] — the engine function
- [[concepts/non-judging-harness]] — relies on per-turn independence
- [[concepts/frame]] — what _does_ persist across turns
