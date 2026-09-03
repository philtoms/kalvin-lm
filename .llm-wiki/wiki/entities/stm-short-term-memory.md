---
type: entity
title: STM (Short-Term Memory)
description: Recent attention — what Kalvin was just thinking about. Written by attention itself; the temporally-situated relation to held klines. Empty at session start.
created: 2026-08-11
updated: 2026-08-14
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# STM (Short-Term Memory)

Recent attention — what Kalvin was just thinking about.

## Overview

STM is a mode of relation to held klines, not a storage location (see
[[concepts/memory]]): a kline in STM is *recently attended to*. STM is written
by attention itself — whatever cogitation touches hits it. This is how
traversal is temporally situated, and how Kalvin can notice it is revisiting
something. Empty at session start.

**Writing STM is attending.** This is the concept, not a bookkeeping rule: any
implementation (bounded window, dual-keyed index) is an embodiment of the
attention relation.

In the production [[entities/k-engine|engine]] memory model (`kalvin.Model`),
STM is the first tier of the write cascade — every write reaches it, and
`grounded()` deliberately excludes it (transient entries have not been
rationalised).

In the dialogue harness's `EngineState`, the pending-attention store
(formerly `work_list`) **is** STM: incoming entries and the ungrounded
signatures/nodes their routing unpacks sit in `stm` until they ground or are
asked about — exactly the written-by-attention relation. The separate
`kalvin.stm.STM` index field was removed once the two were recognised as the
same concept.

_Avoid_: STM caching (too vague), working memory (too vague), context window
(implies a passive buffer), an index (implementation detail, not the concept).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical seed (CONTEXT.md)
- [[concepts/memory]] — the tiered structure STM belongs to
- [[concepts/model]] — the model whose attention relation STM carries
- [[entities/ltm-long-term-memory]] — held knowledge (a different relation to the same klines)
- [[entities/k-engine]] — both implementation senses