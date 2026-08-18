---
type: concept
title: Signature
description: The value occupying a kline's head position — the head value a kline's nodes compose against, and a value other klines hold as nodes.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Signature

The value occupying a kline's head position.

## Definition

The signature is the head value a kline's nodes compose against — the locus of
the [[concepts/structural-significance]] claim. It is also a value other klines
hold as nodes when they reference this kline, which is how
[[concepts/rational-significance]] gets evaluated across klines.

A signature is not directly visible until the engine discovers it. Per
[[concepts/signature-discovery]], signatures are only discovered when the slow
route unpacks them from S2/S3 incomings; feeding an S1 identity alone does not
ground it until K has framed the signature first.

Multi-character KScript identifiers become compound signatures via
[[concepts/mts-multi-token-signature]] expansion.

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — discovery rule
- [[concepts/kline]] — the structure a signature heads
- [[concepts/node]] — slots that may themselves be signatures
- [[concepts/structural-significance]] — what the signature–nodes relationship claims
- [[concepts/signature-discovery]] — when the engine discovers signatures
- [[concepts/mts-multi-token-signature]] — compound signatures
