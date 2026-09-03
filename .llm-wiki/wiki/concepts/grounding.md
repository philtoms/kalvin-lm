---
type: concept
title: Grounding
description: The model's mechanism for realising significance — at any level. A grounded signature guarantees its nodes are grounded; Frame-grounded klines are available for cogitation; LTM-grounded klines are frame promotions Kalvin deems important enough to remember.
created: 2026-08-11
updated: 2026-08-17
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# Grounding

The [[concepts/model|model]]'s mechanism for realising **significance** — at
any level, not only S1.

## Definition

The glossary is normative here: it states what the code must follow, not what
it currently says.

- If a signature is grounded, then Kalvin knows that all of its nodes are
  grounded also.
- KLines grounded in a [[concepts/frame|Frame]] are available for
  [[concepts/cogitation|cogitation]].
- KLines grounded in [[entities/ltm-long-term-memory|LTM]] are frame
  promotions that Kalvin deems important enough to remember.

Kalvin is free to ground any information **it understands at any level of
significance** — if that information leads through cogitation to a previously
grounded or ratified proposal. Grounding is thus broader than commitment: S1
is what full commitment looks like, but a kline understood at S2 or S3 can be
grounded in Frame, where cogitation can re-traverse it.

This points at what "deems important enough to remember" means in practice:
LTM promotion is earned by **participation in further rationalisation** — a
kline matters because cogitation has used it to ground or ratify something
else — not by static criteria such as edge counts.

Grounding is how significance is _produced_ in memory, not what it _means_;
"recognised" is the significance-level concept
([[concepts/rational-significance]]).

### Current engine state (S1-shaped)

The engine implements the S1 special case: `_is_groundable` admits an
identity (self-referential, its node is itself) or any kline whose every node
is already grounded, and the fast route enforces this for canons. Grounding
at S2/S3 levels — Frame-grounded partial knowledge — is not yet implemented.

_Avoid_: self-grounded (a canon does NOT self-ground — its nodes must be
grounded first), grounded identity (grounding applies to any kline that
attains significance, not just identities), reducing grounding to S1
commitment (S1 is the special case, not the mechanism).

## Links

- [CONTEXT.md](../../../../CONTEXT.md) — canonical definition (glossary, Grounding entry)
- [[concepts/model]] — the model grounding admits into
- [[concepts/memory]] — grounding as a memory relation
- [[concepts/frame]] — Frame-grounded klines are available for cogitation
- [[entities/ltm-long-term-memory]] — promotion = importance earned through participation
- [[concepts/cogitation]] — the traversal grounding feeds and is fed by
- [[entities/isgroundable]] — the engine predicate (currently S1-shaped)