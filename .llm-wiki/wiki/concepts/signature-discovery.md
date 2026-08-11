---
type: concept
title: Signature discovery
description: The rule that signatures are only discovered when the slow route unpacks them from S2/S3 incomings — feeding an S1 identity alone doesn't ground it until K frames the signature first.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Signature discovery

When and how the engine discovers signatures.

## Definition

A signature is only discovered when the slow route unpacks it from an S2/S3
incoming; unreferenced signatures stay invisible. This is why feeding an S1
identity alone does not ground it — the engine grounds it only once K has framed
the signature first.

Two boundary cases:

- An incoming S4 (`{X:[]}`) is a reply to K's own framed ask — feeding one never
  discovers a signature.
- S4 identity asks whose signature the curriculum defines as `X:[X]` are
  answered inline, mechanically, with per-step dedup.

Discovery is the precondition for [[concepts/grounding]] — without a framed
signature, there is nothing to ground against.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — the discovery rule
- [[concepts/signature]] — what gets discovered
- [[concepts/fast-route-vs-slow-route]] — discovery happens on the slow route
- [[concepts/grounding]] — depends on discovery
- [[concepts/unknown]] — the S4 reply case
