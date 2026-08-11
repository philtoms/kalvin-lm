---
type: concept
title: Terminal
description: "A kline whose structure carries no further decomposition — a leaf that tells Kalvin to stop traversing. Three shapes: empty nodes, self-referential nodes, and the compound-word form."
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-001
    resource: /sources/SRC-2026-08-11-001.md
---

# Terminal

A kline whose structure carries no further decomposition.

## Definition

A terminal is a leaf that tells Kalvin to stop traversing the model. Three
structural shapes are terminal:

- **empty nodes** — the [[concepts/unknown]] (`{S: []}`)
- **self-referential nodes** — the [[concepts/identity]] (`{S: [S]}`)
- **the compound-word form**

[[concepts/unknown]] and [[concepts/identity]] are the two terminal structures;
[[concepts/canon]] and [[concepts/misfit]] are non-terminal.

_Avoid_: leaf node (a terminal is a kline, not a node), base case (implementation
term), atomic (overloaded).

## Links

- [SRC-2026-08-11-001](/sources/SRC-2026-08-11-001.md) — canonical definition (CONTEXT.md)
- [[concepts/unknown]] — terminal, claims S4
- [[concepts/identity]] — terminal, claims S1
- [[concepts/canon]], [[concepts/misfit]] — non-terminal structures
- [[concepts/cogitation]] — the traversal that terminals end
