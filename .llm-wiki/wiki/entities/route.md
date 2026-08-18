---
type: entity
title: route()
description: The engine function that dispatches an incoming kline on its structural significance (sig_level), not on the producer's compiled stamp. The entry to fast/slow routing.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# route()

The engine's routing function.

## Overview

`route()` dispatches on [[concepts/structural-significance]] (the `sig_level`),
not on the producer's compiled stamp. It decides whether an incoming kline takes
the fast route (identities and seen-signature canons admitted directly) or the
slow route (unseen-signature canons unpacked as asks; S2/S3 routed to
[[concepts/cogitation]]; S4 recognised as a reply to K's own framed ask).

The routing decision is the first branch in a turn; everything downstream —
[[concepts/signature-discovery]], [[concepts/grounding]],
[[entities/cogitate|cogitate]] — follows from it.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md)
- [[concepts/fast-route-vs-slow-route]] — what route() chooses between
- [[concepts/structural-significance]] — what it dispatches on
- [[entities/cogitate]], [[entities/rationalise]] — downstream functions
- [[entities/k-engine]] — the engine route() belongs to
