---
type: concept
title: Cogitation
description: The slow path of rationalisation — model traversal that tests a kline's structural claim against what Kalvin holds, draining a backlog of S2/S3 klines and emitting proposals.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
  - id: SRC-2026-08-11-003
    resource: /sources/SRC-2026-08-11-003.md
---

# Cogitation

The slow path of rationalisation — model traversal that tests a kline's
structural claim against what Kalvin holds.

## Definition

Where [[concepts/structural-significance]] is derived from the signature–nodes
relationship alone, cogitation expands the kline through the model: retracing
paths, discovering connections, classifying each against the
[[concepts/rational-significance]] levels. It drains a backlog of unresolved
(S2/S3) klines, emitting [[concepts/proposal|proposals]] for ratification. Its
result is a Rationally Significant KLine — a kline that Kalvin understands.

The engine's `cogitate` runs one full LIFO pass over the work-list (no
short-circuit): per entry it asks (S4), countersigns (S3), proposes (S2), or
grounds (S1). It speaks in semantic predicates (`is_identity`, `is_unknown`,
`is_canon`, `is_relationship`), never raw `kline.nodes`.

Two cogitation strategies exist — [[entities/similarfit]] (graft heuristic) and
[[entities/expand]] (grades grounded candidates) — selectable via the 
harness `-s` flag. They diverge on S2 emissions, not on the grounded model: on
`mhall` both reach the same grounded model, but `similar_fit` emits 12 S2
proposals and `expand` emits 1 (see [[concepts/canonical-synthesis]],
[[concepts/silent-synthesis]]).

S2 and S3 are active reasoning states during cogitation — not failed S1s.
Cogitation alone can reach S2 at most; S1 requires [[concepts/ratify|ratification]]
by another agent.

_Avoid_: thinking (informal), background thread (implementation), the cogitator
(the implementation class).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — engine mechanics (LIFO pass, semantic predicates, strategy divergence)
- [SRC-2026-08-11-003](/sources/SRC-2026-08-11-003.md) — S2/S3 as active reasoning, not failures
- [[concepts/structural-significance]] — the claim cogitation tests
- [[concepts/rational-significance]] — the measurement cogitation derives
- [[concepts/proposal]] — what cogitation emits
- [[concepts/s2-expansion]] — how S2 misfits are expanded
- [[entities/cogitate]] — the engine function
- [[entities/similarfit]], [[entities/expand]] — the two strategies
