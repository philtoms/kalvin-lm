---
type: concept
title: Structural Significance
description: The significance a kline's structure claims on its own — an S-level derived from the signature–nodes relationship alone, without model traversal. The ground truth every participant measures against.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Structural Significance

The significance a kline's structure **claims** — independent of who is looking.

## Definition

An S-level (the same S1–S4 as [[concepts/rational-significance]]) derived from
the signature–nodes relationship alone, without model traversal. The claim is
**coverage** — how much of the kline the signature covers — with rules
independent of node count and of the relational token that compiled the shape:

| Structure | Kline | Claim | Why |
| --------- | ----- | ----- | --- |
| [[concepts/unknown]]   | `A:[]`        | S4 | no nodes |
| [[concepts/identity]], [[concepts/canon]] | `A:[A]`, `ABC:[A,B,C]` | S1 | signature covers its nodes exactly |
| Underfit / Overfit / Under+over | `ABC:[A,C]`, `AB:[A,B,C]`, `ABC:[B,C,D]` | S2 | at least one node covered |
| Denotation | `AB:[B]` | S2 | the compound signature covers its node |
| [[concepts/misfit]] (no-fit), Connotation | `AB:[C,D]`, `A:[B]` | S3 | no node covered |

Structure is the ground truth every participant measures against; it is
independent of the observer. [[concepts/cogitation]] then measures this claim
against what Kalvin actually holds to arrive at rational significance.

In the engine, `route()` dispatches on structural significance (the
`sig_level`), not the producer's compiled stamp — see
[[concepts/fast-route-vs-slow-route]].

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — `sig_level` dispatch
- [[concepts/significance-spectrum-s1s4]] — the shared spectrum
- [[concepts/rational-significance]] — the rationalised measurement against this claim
- [[concepts/cogitation]] — the slow path that tests this claim
- [[concepts/fast-route-vs-slow-route]] — routing dispatches on sig_level
