---
type: entity
title: TokenEncoder
description: The KScript compiler stage that resolves word bindings and produces encoded KLines from symbolic entries. Where identities receive their node labels.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# TokenEncoder

The KScript compiler stage that produces encoded KLines.

## Overview

TokenEncoder is the final stage of the `ks` pipeline: it takes symbolic entries
(from ASTEmitter, after binding-scope resolution) and produces encoded
[[concepts/kline|KLines]]. This is where [[concepts/word-binding|word bindings]]
are applied to their final tokens and where identities receive their
`node_labels`.

Two ordering facts matter for reading compiled output (see
[[concepts/mts-multi-token-signature|MTS]] and the behaviour-notes compilation
section): MTS entries are emitted after all source entries, and symbolic-entry
indices do not align with compiled-KValue indices — compiler output order ≠
authored order.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md)
- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — KScript section
- [[entities/kscript]] — the language whose compiler this belongs to
- [[concepts/word-binding]] — what TokenEncoder applies
- [[concepts/mts-multi-token-signature]] — emission ordering
