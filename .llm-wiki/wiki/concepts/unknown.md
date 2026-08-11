---
type: concept
title: Unknown
description: "A terminal kline structure with empty nodes (`{S: []}`) that claims S4 — the structural form of an ask, requesting an Identity ratification."
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Unknown

A kline **structure**: a [[concepts/terminal]] with empty nodes.

## Definition

Structural shape `{S: []}`. Claims **S4** — _"I don't know this"_ (nothing held
for this signature). The structural form of an **ask**: an S4 proposal that
requests an [[concepts/identity]] ratification.

In the engine, an incoming S4 (`{X:[]}`) is a reply to K's own framed ask —
feeding one never discovers a signature ([[concepts/signature-discovery]]).

_Avoid_: empty kline (describes syntax, not the meaning), bare signature
(describes syntax, not the structure), identity (the empty form is _not_ an
identity — it is the opposite: unknown, not known).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — S4 incoming semantics
- [[concepts/identity]] — the ratification an Unknown requests
- [[concepts/terminal]] — Unknown is one of the terminal shapes
- [[concepts/structural-significance]] — the S4 claim
- [[concepts/signature-discovery]] — why S4 incomings don't discover signatures
