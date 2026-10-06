---
type: source
title: "Observation: TRS block committed: harness, KNode realisation, LIFO runner, I-1 removal"
tags:
  - trs
  - commit
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-trs-block-committed-harness-knode-realisation-lifo-runner-i-
relevance: medium
observed_at: 2026-10-06T11:40:34.651Z
---

# 🔍 Observation: TRS block committed: harness, KNode realisation, LIFO runner, I-1 removal

COMMITTED 02e85ef on trs branch: the whole uncommitted block landed as one commit — 'trs: harness hookup (content-space KNode realisation), LIFO runner, seeding withdrawal, I-1 removed'. Covers: TrsRunner hookup with --model trs flag, content-space masking at the boundary (WORD_MASK, KNode preserved), KNode provenance labels throughout model ops, LIFO ordinary-stack runner (duplicates, per-work-entry openers), amendment 19/D16 (seeding withdrawn, unrecognised nodes route S4), amendment 20 (I-1 withdrawn from spec and code — Domain.i1_closed deleted; Definition 3 now states no referential invariant is claimed, the no-neighbours route carries consistency). Tests: 156 passing after removing i1 assertions (one leftover in test_lifo_cycles_starve_queued_products found via failing run, not grep). Wiki observation files included in the commit.

*Relevance: medium*
*Tags: trs commit*

---
*Observed: 2026-10-06T11:40:34.651Z*
