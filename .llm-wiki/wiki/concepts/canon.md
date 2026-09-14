---
type: concept
title: Canon
description: A kline structure where signature = signature_of(nodes), claiming S1 — the signature stands for its nodes and carries no information beyond them.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# Canon

A kline **structure**: the signature equals `signature_of(nodes)`.

## Definition

Claims **S1** — the signature stands for its nodes, so it is safe to use the
signature in place of them. The signature carries no information beyond what its
nodes already express. Structural shape `{AB: [A, B]}` where `AB` represents a
combination of two or more nodes.

A canon is one of the two ways a kline self-grounds (see [[concepts/grounding]]);
the other is an [[concepts/identity]]'s self-reference.

Distinct from the CANONICALISES relational token (see [[concepts/relational-tokens]]):
`=>` declares an _intent_ to compose, and a CANONICALISES statement need not
construct a Canon. Distinct from [[concepts/mts-multi-token-signature]], which is
an example of a compound signature, not the Canon concept itself.

_Avoid_: canonical (ambiguous with Relational Tokens); treating `=>` as
synonymous with being a Canon; MTS (an example, not the concept).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[concepts/identity]] — the other S1-claiming structure
- [[concepts/misfit]] — the S2-claiming structure (signature ≠ signature_of(nodes))
- [[concepts/grounding]] — canons self-ground
- [[concepts/relational-tokens]] — CANONICALISES declares intent, may not produce a Canon
- [[concepts/mts-multi-token-signature]] — compound signature, distinct from Canon
