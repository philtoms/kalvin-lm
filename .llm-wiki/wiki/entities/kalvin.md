---
type: entity
title: Kalvin
description: The rationalising system — an agent whose every response carries significance, a measurement of how well-grounded the response is in what it already knows.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-003
    resource: /sources/SRC-2026-08-11-003.md
---

# Kalvin

The rationalising system whose entire world is built from [[concepts/kline|klines]].

## Overview

Kalvin receives new information, rationalises it against existing knowledge, and
produces a response carrying a [[concepts/significance]] measurement indicating
how well it understood. It is neither an oracle (which gives answers with no
basis for trust) nor a lookup table (which returns immediately without
thinking). Understanding emerges from three pillars: [[concepts/fit]],
[[concepts/learned-preferences]], and [[concepts/significance]].

There is no training mode: Kalvin rationalises all klines identically. The
[[concepts/harness|harness]] code distinguishes training from operational use,
not Kalvin itself. Kalvin can autonomously reach S2 at most; S1 requires
[[concepts/ratify|ratification]] by another agent.

The codebase implements Kalvin as the [[entities/k-engine|K engine]] running
inside the [[entities/harness|training harness]] or the dialogue harness.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — domain glossary
- [SRC-2026-08-11-003](/sources/SRC-2026-08-11-003.md) — vision: Kalvin as a rational agent
- [[entities/k-engine]] — the implementation
- [[concepts/kline]] — the unit Kalvin rationalises
- [[concepts/significance]] — what every response carries
- [[concepts/cogitation]] — how Kalvin rationalises
