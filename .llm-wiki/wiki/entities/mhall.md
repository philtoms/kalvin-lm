---
type: entity
title: mhall
description: Test case ('Mary had a little lamb') on which both cogitation strategies reach the same grounded model but diverge on S2 emissions.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# mhall

Test case signature `MHALL` — "Mary had a little lamb".

## Overview

`mhall` is the canonical test case for [[concepts/s2-expansion|S2 expansion]]
strategy comparison. On it, [[entities/similarfit|similar_fit]] and
[[entities/expand|expand]] reach the same grounded model but diverge on S2
emissions: `similar_fit` emits 12 [[concepts/proposal|proposals]] (including the
[[concepts/canonical-synthesis]] at step 16, plus spurious recombinations);
`expand` emits 1 (`WDMH:[Mary]` at step 10) and leaves
`WDMH:[Mary,had,a,little,lamb]` ungrounded.

Both strategies leave the WDMH↔MHALL pair ungrounded, which is the crux of the
[[concepts/silent-synthesis]] open question.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — the strategy-divergence finding
- [[entities/wdmh]] — the pair member mhall is compared against
- [[entities/marys-world]] — the example mhall comes from
- [[concepts/canonical-synthesis]], [[concepts/silent-synthesis]] — the live diagnostic concepts on this case
- [[concepts/s2-expansion]] — the general mechanism under test
