---
type: concept
title: Identity
description: "A terminal kline structure that is directly decodable — a known value (`{S: [S]}`) that claims S1, 'I know this.'"
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Identity

A kline **structure**: a [[concepts/terminal]] that is directly decodable.

## Definition

A known value that translates to something in the outside world. Claims **S1** —
_"I know this."_ One structural shape: self-referential (`{S: [S]}`).

A bare signature that is [[concepts/word-binding|word-bound]] compiles to an
identity; a bare unbound signature compiles to an [[concepts/unknown]] instead.
Binding chooses the structure; the structure determines the
[[concepts/target-significance]].

Engine note: feeding an S1 identity alone does not ground it; the engine grounds
it only once K has framed the signature first ([[concepts/signature-discovery]]).

_Avoid_: unsigned (implementation term), bare signature (describes syntax, not
the structure), treating the empty kline as an Identity (it is an
[[concepts/unknown]]).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — identity feeding rule
- [[concepts/unknown]] — the opposite terminal (what an identity is _not_)
- [[concepts/terminal]] — identity is one of the terminal shapes
- [[concepts/word-binding]] — how a bare signature becomes an identity
- [[concepts/grounding]] — identities are S1 by their own structure
