---
type: entity
title: _is_groundable()
description: "The engine predicate that decides whether a kline can be grounded, branching in order: identity → signature grounded; canon → all nodes grounded; relationship → reciprocal grounded; misfit → both."
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# _is_groundable()

The engine predicate implementing the grounding decision.

## Overview

`_is_groundable` branches in order of structure:

- [[concepts/identity]] → signature grounded
- [[concepts/canon]] → all nodes grounded
- single-node [[concepts/relationship]] → reciprocal grounded
- [[concepts/misfit]] (no-fit/underfit/overfit) → both signature AND all nodes grounded

Its output drives [[entities/promote|_promote]], which grounds groundable
entries to fixed point at S1. `_is_groundable` is the predicate;
`_promote` is the cascade.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md)
- [[concepts/grounding]] — the concept _is_groundable implements
- [[entities/promote]] — the cascade that consumes its output
- [[concepts/identity]], [[concepts/canon]], [[concepts/relationship]], [[concepts/misfit]] — the branch cases
