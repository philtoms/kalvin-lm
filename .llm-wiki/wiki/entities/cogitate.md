---
type: entity
title: cogitate()
description: The engine function implementing cogitation — one full LIFO pass over the work-list, emitting semantic predicates (ask S4 / countersign S3 / propose S2 / ground S1).
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# cogitate()

The engine function implementing [[concepts/cogitation]].

## Overview

`cogitate` runs one full LIFO pass over the work-list (no short-circuit): per
entry it asks (S4), countersigns (S3), proposes (S2), or grounds (S1). It speaks
in semantic predicates (`is_identity`, `is_unknown`, `is_canon`,
`is_relationship`), never raw `kline.nodes`. The result is a list of proposals
for ratification and a set of groundings.

Two strategies implement the S2 arm — [[entities/similarfit]] (graft heuristic)
and [[entities/expand]] (grades grounded candidates) — selectable via the lean
harness `-s` flag (default `expand`). They diverge on S2 emissions, not on the
grounded model.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md)
- [[concepts/cogitation]] — the concept cogitate implements
- [[concepts/significance-spectrum-s1s4]] — the four emission kinds
- [[entities/similarfit]], [[entities/expand]] — the two S2 strategies
- [[concepts/canonical-synthesis]], [[concepts/silent-synthesis]] — live diagnostic on strategy divergence
