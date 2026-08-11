---
type: concept
title: Fast route vs slow route
description: Engine routing paths — fast admits identities and seen-signature canons; slow unpacks unseen-signature canons as asks and discovers signatures. Dispatch is on sig_level, not the compiled stamp.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Fast route vs slow route

The two routing paths through the engine.

## Definition

`route()` dispatches on [[concepts/structural-significance]] (the `sig_level`),
_not_ on the producer's compiled stamp. The two paths:

- **fast route** — admits identities and seen-signature canons directly.
- **slow route** — unpacks unseen-signature canons as asks (S4); this is where
  [[concepts/signature-discovery]] happens and where
  [[concepts/cogitation]] runs over S2/S3 incomings.

Two consequences: unseen-signature identities are dropped, and unseen-signature
canons take the slow route. The distinction is what makes
[[concepts/grounding]] depend on signature framing — feeding an S1 identity
alone does not ground it until K has framed the signature first.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — routing rules
- [[concepts/structural-significance]] — what routing dispatches on
- [[concepts/signature-discovery]] — what the slow route enables
- [[concepts/cogitation]] — runs on the slow route
- [[concepts/grounding]] — depends on routing outcomes
- [[entities/route]] — the engine function
