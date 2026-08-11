---
type: entity
title: rationalise()
description: The engine function driving one full rationalisation turn — resets observations and _incoming, runs route → cogitate → _promote, and emits the turn's batch.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# rationalise()

The engine function driving one full rationalisation turn.

## Overview

`rationalise()` is the top-level turn entry. At the top of every call it resets
`observations` and `_incoming` to fresh lists (see
[[concepts/per-turn-scoping]]), then drives the pipeline:
[[entities/route|route]] → [[entities/cogitate|cogitate]] →
[[entities/promote|_promote]]. The turn's emissions form the `batch` returned to
the caller.

State that persists across turns lives in [[concepts/frame|Frame]] and
[[entities/ltm-long-term-memory|LTM]], not in `rationalise()`'s registers.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md)
- [[concepts/per-turn-scoping]] — the reset behaviour
- [[entities/route]], [[entities/cogitate]], [[entities/promote]] — the pipeline it drives
- [[entities/k-engine]] — the engine rationalise() belongs to
