---
type: entity
title: expand
description: A cogitation strategy for the S2 arm — grades grounded candidates via kalvin.expand.expand. The dialogue harness default. Emits few proposals; exhibits silent synthesis on mhall.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# expand

A cogitation strategy for the S2 (misfit) arm.

## Overview

`expand` grades grounded candidates via `kalvin.expand.expand`. It is the 
dialogue harness's default strategy (`-s expand`). On the [[entities/mhall]]
test case it emits only 1 proposal (`WDMH:[Mary]` at step 10) and goes silent on
the WDMH↔MHALL recombination — the [[concepts/silent-synthesis]] open question.

Both `expand` and [[entities/similarfit|similar_fit]] reach the same grounded
model; they diverge only on which S2 proposals they emit. `expand` emits fewer.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md)
- [[entities/similarfit]] — the other strategy
- [[entities/cogitate]] — the function whose S2 arm expand implements
- [[concepts/s2-expansion]] — the general mechanism
- [[concepts/silent-synthesis]] — expand's characteristic behaviour on mhall
- [[concepts/canonical-synthesis]] — what similar_fit emits and expand doesn't
