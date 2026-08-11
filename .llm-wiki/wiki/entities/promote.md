---
type: entity
title: _promote()
description: The engine cascade that grounds groundable entries to fixed point at S1. Does not emit S2 proposals — promotion is the fast S1 path.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# _promote()

The engine cascade that grounds groundable entries to S1.

## Overview

`_promote` runs after [[entities/cogitate|cogitate]]. It iterates the
groundability predicate ([[entities/isgroundable|_is_groundable]]) to fixed
point: every kline that can be grounded becomes S1. It does _not_ emit S2
proposals — promotion is the fast path to S1, distinct from the S2
expansion path in cogitate.

This separation is why feeding an S1 identity alone can ground it (via
`_promote`) only _after_ the signature has been framed — the cascade runs on
what the slow route has already discovered
([[concepts/signature-discovery]]).

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md)
- [[concepts/grounding]] — what _promote realises
- [[entities/isgroundable]] — the predicate it iterates
- [[entities/cogitate]] — the S2 path _promote is distinct from
- [[concepts/signature-discovery]] — the precondition for promotion
