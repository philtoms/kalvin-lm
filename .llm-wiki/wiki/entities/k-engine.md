---
type: entity
title: K (engine)
description: The cognitive/reasoning engine — the stateless core that derives one dialogue turn from (state, incoming) and returns (batch, observations). The implementation of Kalvin's rationalisation.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# K (engine)

The cognitive/reasoning engine that is the target system being studied and
improved.

## Overview

The engine is pure mechanism: it holds an `EngineState` and a `MisfitStrategy`,
both fully constructed by the caller. A turn derives from `(state, incoming)`
and returns `(batch, observations)` — dialogue emissions and K's internal S1
groundings this turn. The engine is stateless about its own emissions; dedup
lives in the actor (see [[concepts/per-turn-scoping]]).

The factories that assemble the engine (signifier, state, strategy, engine) live
in `dialogue.harness`. Routing is done by [[entities/route|route()]];
[[concepts/cogitation]] by [[entities/cogitate|cogitate()]];
[[concepts/grounding]] resolution by [[entities/isgroundable|_is_groundable]]
and [[entities/promote|_promote]].

The engine is the target of [[concepts/engine-first|engine-first]] discipline
and of [[entities/auto-tune|auto-tune]].

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md)
- [[entities/route]], [[entities/cogitate]], [[entities/isgroundable]], [[entities/promote]], [[entities/rationalise]] — the engine's functions
- [[entities/harness]] — assembles and drives the engine
- [[concepts/non-judging-harness]] — how the engine is exercised
- [[concepts/engine-first]] — the discipline for editing it
