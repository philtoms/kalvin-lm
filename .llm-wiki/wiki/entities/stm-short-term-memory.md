---
type: entity
title: STM (Short-Term Memory)
description: The lowest tier in the write cascade (production Model) and Kalvin's event register. In the lean EngineState, reserved for expansion and currently unwired. Empty at session start.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# STM (Short-Term Memory)

The lowest tier in the write cascade and Kalvin's event register.

## Overview

In the production [[entities/k-engine|engine]] memory model
(`kalvin.Model`), every write reaches STM; it is the foundation on top of which
[[entities/ltm-long-term-memory|LTM]] and [[concepts/frame|Frame]] are built. STM
is empty at session start — it accumulates the session's events as
rationalisation proceeds.

In the lean [[entities/harness|dialogue harness]]'s `EngineState`, STM has a
different, narrower role: it is **reserved for the expansion strategies'
exclusive use** and is currently **not wired into any logic**. `EngineState`
holds four independent stores — `work_list` (the cogitator queue), `ltm`
(ratified klines), `frame` (outgoing proposals), and `stm` (empty until
expansion uses it) — and writes to one store do not cascade to the others.
See [[entities/k-engine]].

_Avoid_: STM caching (too vague), working memory (too vague), context window
(implies a passive buffer).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[entities/ltm-long-term-memory]] — the tier above STM (production model)
- [[concepts/frame]] — persistent working context built on the memory tiers
- [[entities/k-engine]] — the lean EngineState's four-store model
- [obs-2026-08-11-enginestate-four-store-model-grounded-ltm-stm-disconnected](/sources/obs-2026-08-11-enginestate-four-store-model-grounded-ltm-stm-disconnected.md) — lean-engine STM is reserved, not cascaded
