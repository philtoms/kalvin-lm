---
type: entity
title: similar_fit
description: A cogitation strategy for the S2 arm — the graft heuristic. Emits many proposals (12 on mhall), including the canonical synthesis at step 16.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# similar_fit

A cogitation strategy for the S2 (misfit) arm — the graft heuristic.

## Overview

`similar_fit` is implemented in `dialogue/similar_fit.py`. Selectable via the
dialogue harness `-s similar_fit` flag (the default is `expand`). On the
[[entities/mhall]] test case it emits 12 [[concepts/proposal|proposals]],
including the [[concepts/canonical-synthesis]] at step 16 (the correct
recombination for the WDMH↔MHALL pair) plus spurious recombinations under wrong
signatures.

Both `similar_fit` and [[entities/expand]] reach the same grounded model; they
diverge only on which S2 proposals they emit. `similar_fit` emits more, which is
noisier but surfaces the synthesis `expand` silences.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md)
- [[entities/expand]] — the other strategy
- [[entities/cogitate]] — the function whose S2 arm similar_fit implements
- [[concepts/s2-expansion]] — the general mechanism
- [[concepts/canonical-synthesis]] — the step-16 proposal similar_fit surfaces
- [[concepts/silent-synthesis]] — what expand does instead
