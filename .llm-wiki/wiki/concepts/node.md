---
type: concept
title: Node
description: A structural slot in a kline's nodes list — either a Token Id or the signature of another kline.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# Node

A structural slot: a value occupying a position in a kline's nodes list.

## Definition

A node is either a **Token Id** (a value produced by the tokenizer) or the
**signature** of another kline. This duality is how klines reference each other
and form the graph the model traverses during [[concepts/cogitation]].

_Avoid_: child, element — the structural slot is specifically a node. (A
[[concepts/terminal]] is a kline, not a node.)

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[concepts/kline]] — the structure nodes belong to
- [[concepts/signature]] — a node may hold another kline's signature
- [[concepts/cogitation]] — model traversal walks node references
