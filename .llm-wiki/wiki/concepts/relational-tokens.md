---
type: concept
title: Relational Tokens
description: The closed set of KScript tokens (==, =>, >, =, none) declaring how a kline is produced and its provenance — COUNTERSIGNS, CANONICALZES, CONNOTES, DENOTES, UNKNOWN.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# Relational Tokens

The closed set of written tokens that declare how a kline is produced in KScript.

## Definition

A compiler/provenance concept: the token declares an _intent_, which the
resulting kline's actual [[concepts/structural-significance]] may or may not
satisfy. The five tokens:

| Token | Name         | Form                                     | Effect                                                               |
| ----- | ------------ | ---------------------------------------- | -------------------------------------------------------------------- |
| `==`  | COUNTERSIGNS | 1:1 reciprocal pair `{A:[B]}`, `{B:[A]}` | signature countersigns each other's nodes                            |
| `=>`  | CANONICALZES | 1:many `{A:[B,C,D]}`                     | declares intent to aggregate (need not produce a [[concepts/canon]]) |
| `>`   | CONNOTES     | 1:1 `{A:[B]}`                            | A connotes B (subjectively, _A is a B_)                              |
| `=`   | DENOTES      | 1:1 `{B:[A]}`                            | A denotes B (objectively, _B is an A_)                               |
| none  | UNKNOWN      | bare unbound signature                   | compiles to the empty [[concepts/unknown]] `{A:[]}` — the ask        |

A bare signature with no [[concepts/word-binding]] compiles to the empty
[[concepts/unknown]]; a bare signature that is word-bound compiles instead to an
[[concepts/identity]] `{A:[A]}`. Binding chooses the structure; the structure
then determines the [[concepts/target-significance]].

CONNOTES and DENOTES both produce the [[concepts/relationship]] structure
(single-node misfit); they differ in direction.

_Avoid_: structural relationship (collides with Structural Significance),
relational operator (the token declares provenance, not an operation).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[concepts/structural-significance]] — what the token's intent may or may not satisfy
- [[concepts/target-significance]] — the label the generated structure carries
- [[concepts/relationship]] — CONNOTES/DENOTES produce this structure
- [[concepts/canon]] — what CANONICALZES intends but need not produce
- [[concepts/word-binding]] — chooses identity vs unknown for bare signatures
- [[entities/kscript]] — the language these tokens belong to
