---
type: entity
title: STM (Short-Term Memory)
description: The lowest tier in the write cascade and Kalvin's event register — every write reaches it. Empty at session start.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# STM (Short-Term Memory)

The lowest tier in the write cascade and Kalvin's event register.

## Overview

Every write reaches STM; it is the foundation on top of which
[[entities/ltm-long-term-memory|LTM]] and [[concepts/frame|Frame]] are built. STM
is empty at session start — it accumulates the session's events as
rationalisation proceeds.

_Avoid_: STM caching (too vague), working memory (too vague), context window
(implies a passive buffer).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[entities/ltm-long-term-memory]] — the tier above STM
- [[concepts/frame]] — persistent working context built on the memory tiers
