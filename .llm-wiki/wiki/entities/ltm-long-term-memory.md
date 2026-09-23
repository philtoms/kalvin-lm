---
type: entity
title: LTM (Long-Term Memory)
description: What Kalvin holds as grounded knowledge — the commitment relation to held klines. Structurally identical to Frame; the distinction is what is counted on vs what is in focus.
created: 2026-08-11
updated: 2026-08-14
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# LTM (Long-Term Memory)

What Kalvin **holds as grounded knowledge**.

## Overview

LTM is a mode of relation to held klines, not a storage location (see
[[concepts/memory]]): a kline in LTM is *counted on*. Structurally identical to
[[concepts/frame|Frame]]; the distinction is the relation — Frame is what is
in focus, LTM is what is held as knowledge. A kline residing in LTM is
[[concepts/grounding|grounded]].

LTM is [[concepts/monotonic-growth|monotonic]]: it only grows. Correction
happens by outcompeting, not deletion.

**Promotion (open).** How a kline moves from Frame to LTM is not established —
candidate ideas (edge-count thresholds like `MIN_EDGES`, counted across frames)
exist but are unimplemented. Currently, both implementations populate their
LTM-equivalent directly on grounding/ratification.

In the dialogue harness's `Memory`, LTM is the `ltm` field (ratified
klines, keyed by signature; renamed from `grounded`).

_Avoid_: persistent store (too vague), knowledge base, LTM frame.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical seed (CONTEXT.md)
- [[concepts/memory]] — the tiered structure LTM belongs to
- [[concepts/model]] — the model whose commitment relation LTM carries
- [[concepts/frame]] — focus of attention; structurally identical, relationally distinct
- [[concepts/grounding]] — what LTM residency means
- [[concepts/monotonic-growth]] — LTM only grows