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

Frame holds working context that has been recognised and is worth keeping. In
the production [[entities/k-engine|engine]] memory model (`kalvin.Model`), Frame
is **monotonic** — it only grows, never shrinks (see
[[concepts/monotonic-growth]]). Structurally identical to
[[entities/ltm-long-term-memory]]; the distinction is semantic.

In the lean [[entities/harness|dialogue harness]]'s `EngineState`, `frame` has a
narrower, operational meaning: it is the **emission memory** — the outgoing
kline proposals and identity requests K has emitted — and it is **not**
monotonic. The fast route matches incoming S1/S4 queries against it, and
`unframe` removes a framed ask when a terminal reply consumes it. See
[[entities/k-engine]]'s four-store model.

_Avoid_: session log (Frame is not a log), session.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[entities/ltm-long-term-memory]] — structurally identical, semantically distinct
- [[concepts/monotonic-growth]] — Frame only grows
