---
type: concept
title: Canonical synthesis
description: The S2 proposal at step 16 in similar_fit that represents the correct recombination for mhall — the benchmark proposal expand fails to emit.
created: 2026-08-11
updated: 2026-08-11
sources:
  - id: SRC-2026-08-11-002
    resource: /sources/SRC-2026-08-11-002.md
---

# Canonical synthesis

The S2 [[concepts/proposal|proposal]] at step 16 in
[[entities/similarfit|similar_fit]].

## Definition

On the `mhall` test case, `similar_fit` emits 12 S2 proposals, of which the
canonical synthesis at step 16 represents the _correct_ recombination for the
WDMH↔MHALL pair. [[entities/expand|expand]] emits only 1 proposal
(`WDMH:[Mary]` at step 10) and never surfaces the synthesis — see
[[concepts/silent-synthesis]].

Both strategies reach the same grounded model on `mhall`; they diverge only on
which S2 proposals they emit. The canonical synthesis is the proposal that
distinguishes the two strategies' output.

This is a live diagnostic concept tied to a specific test case ([[entities/mhall]]),
not a domain term — useful while the strategy-divergence question is open.

## Links

- [SRC-2026-08-11-002](/sources/SRC-2026-08-11-002.md) — step 16 on mhall
- [[concepts/silent-synthesis]] — expand's non-emission of this proposal
- [[concepts/s2-expansion]] — the general mechanism
- [[entities/similarfit]], [[entities/expand]] — the diverging strategies
- [[entities/mhall]] — the test case
