---
type: concept
title: MTS (Multi-Token Signature)
description: A KScript compiler device that expands a multi-character signature identifier into its constituent character identities plus one MTS relationship.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# MTS (Multi-Token Signature)

A KScript device for representing a multi-token signature on the LHS in simpler
syntax than would otherwise be required.

## Definition

A compound signature built from more than one Token ID by composition. The
compiler expands a multi-character KScript identifier into its constituent
character identities plus one MTS relationship. This expansion is a property of
the *signature string*, distinct from any CANONIZES decomposition a script
declares for that signature via a block — a CANONIZES scope's nodes are the
declared block operands, never the signature's own MTS character expansion.

In the compiled output, MTS entries are emitted *after* all source entries, with
`KDbg.scope = 1` (source entries are scope 0). Each kline owns its own
annotation; an MTS spawned by a signature inherits the owning scope's
annotation. Consequently symbolic-entry indices do not align with
compiled-KValue indices — compiler output order ≠ authored order.

_Avoid_: decomposition (overloaded — a Canon decomposes into its nodes; an MTS
expands a signature into characters); "Module Type Signatures" (a misnomer — the
M is Multi-token, not Module).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — compilation behaviour (scope, ordering, annotation inheritance)
- [[concepts/canon]] — distinct from MTS: a Canon decomposes a signature into declared block operands
- [[concepts/relational-tokens]] — CANONIZES declares an intent that may use an MTS signature
