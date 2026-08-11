---
type: entity
title: LTM (Long-Term Memory)
description: Persistent knowledge that survives across sessions. Structurally identical to Frame; the distinction is semantic. A kline residing in LTM is grounded.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# LTM (Long-Term Memory)

Persistent knowledge that survives across sessions.

## Overview

LTM is structurally identical to [[concepts/frame|Frame]]; the distinction is
semantic — Frame holds recognised working context, LTM holds the durable
knowledge base. A kline residing in LTM is [[concepts/grounding|grounded]]
(counted as S1), whether by its own structure or via
[[concepts/ratify|ratification]].

LTM is [[concepts/monotonic-growth|monotonic]]: it only grows. Correction
happens by outcompeting, not deletion.

In the lean [[entities/k-engine|engine]], LTM is the `EngineState.ltm` field
(ratified klines, keyed by signature). It was renamed from `grounded` to
separate the store from the S1-realising action (`_ground`/`_promote`); see
[obs-2026-08-11-enginestate-four-store-model-grounded-ltm-stm-disconnected](/sources/obs-2026-08-11-enginestate-four-store-model-grounded-ltm-stm-disconnected.md).

_Avoid_: persistent store (too vague), knowledge base, LTM frame.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[concepts/frame]] — structurally identical, semantically distinct
- [[entities/stm-short-term-memory]] — the tier below
- [[concepts/grounding]] — LTM residency is grounding
- [[concepts/monotonic-growth]] — LTM only grows
