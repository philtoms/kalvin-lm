---
type: concept
title: Frame
description: Recognised working context persisted across sessions; monotonic. Not a log.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# Frame

Recognised working context persisted across sessions.

## Definition

Frame holds working context that has been recognised and is worth keeping. It is
**monotonic** — it only grows, never shrinks (see
[[concepts/monotonic-growth]]). Structurally identical to
[[entities/ltm-long-term-memory]]; the distinction is semantic.

_Avoid_: session log (Frame is not a log), session.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[entities/ltm-long-term-memory]] — structurally identical, semantically distinct
- [[concepts/monotonic-growth]] — Frame only grows
