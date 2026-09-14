---
type: source
title: Identity klines are never proposals
status: insight
category: bugfix
created: 2026-08-26
updated: 2026-08-26
slug: identity-klines-never-proposals
---

# Identity klines are never proposals

Running mhall.ks on the lean harness, step 5 emitted `proposes Object:[Object] S1 255` and `S3 62` — an identity restated as a proposal. Instrumentation pinned the proposer: `ALL:[Query]` (the `Q = ALL` denotes entry, a structural misfit, so the cogitate misfit arm was correct to propose). The bug was in `src/dialogue/expand_fit.py` `propose()`: it accepted any candidate whose `state.find(candidate)` hit, including identity klines, and `expand` then yielded the identity itself (terminal + case-D side-candidate). Fix: skip identity candidates in `propose()` — matching the exclusions already in `_answers_from_ltm` and `_connotations`'s bridge scan ("identities are asks or facts, never answers/proposals"). After the fix: proposals gone, grounded set unchanged, STM drains empty, clean under `-e` too. Note: the trainer's "no misfits in the script" premise is subtly off — denotes/connotes entries compile to relationships, which ARE misfits per CONTEXT.md. Committed 691c2f9 on branch `expand` (not `dialogue`).

*Category: bugfix*

---
*Captured: 2026-08-26*

## Related

_Add links to related pages._
