---
type: concept
title: Silent synthesis
description: expand's behaviour where the WDMH↔MHALL pair grades S1/S3 and nothing is proposed — an open question whether this is correct or a regression.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Silent synthesis

`expand`'s non-emission of the [[concepts/canonical-synthesis]].

## Definition

On the `mhall` test case, `expand` grades the WDMH↔MHALL pair S1/S3 and emits no
[[concepts/proposal]] for it — the synthesis that `similar_fit` surfaces at step
16 stays silent. The open question (2026-08-11):

> Is `expand`'s silent synthesis correct (the pair grades S1/S3, so nothing
> _should_ be proposed) — or a regression (the S2 path should propose the
> recombination regardless of band)?

To resolve: read what `expand` grades for
`(WDMH:[Mary,DH], MHALL:[Mary,had,a,little,lamb])` and why it is not S2.

This is a live diagnostic concept, not a domain term.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — the open question
- [[concepts/canonical-synthesis]] — what `similar_fit` emits and `expand` doesn't
- [[concepts/s2-expansion]] — the general mechanism
- [[entities/expand]] — the strategy exhibiting silence
- [[entities/mhall]], [[entities/wdmh]] — the test case and the silent pair
