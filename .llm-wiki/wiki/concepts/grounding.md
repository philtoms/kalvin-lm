---
type: concept
title: Grounding
description: The model's mechanism for realising S1 — either by structure (a canon self-grounds) or by LTM residency through ratification. Grounding is how S1 is produced, not what S1 means.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Grounding

The model's mechanism for realising **S1** (recognised).

## Definition

A kline is **grounded** when the model counts it as S1, via one of two paths:

- **by its own structure** — a [[concepts/canon]] self-grounds (signature =
  signature_of(nodes)), as does an [[concepts/identity]]'s self-reference;
- **by residing in LTM via [[concepts/ratify|ratification]]** — a structural fact
  the model owns (see [[entities/ltm-long-term-memory]]).

Grounding is how S1 is _produced_, not what S1 _means_; "recognised" is the
significance-level concept ([[concepts/rational-significance]]).

The engine's `_is_groundable` branches in order: identity → signature grounded;
canon → all nodes grounded; single-node [[concepts/relationship]] → reciprocal
grounded; [[concepts/misfit]] → both signature AND all nodes grounded. The
`_promote` cascade grounds groundable entries to fixed point at S1; it does not
emit S2 proposals.

_Avoid_: self-grounded (legacy; conflates the mechanism with the level), grounded
identity (grounding applies to any kline that attains S1, not just identities).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — `_is_groundable` branching, `_promote` cascade
- [[concepts/canon]], [[concepts/identity]] — self-grounding structures
- [[concepts/ratify]] — the path to LTM residency
- [[entities/ltm-long-term-memory]] — where grounded klines reside
- [[entities/isgroundable]], [[entities/promote]] — the engine functions
