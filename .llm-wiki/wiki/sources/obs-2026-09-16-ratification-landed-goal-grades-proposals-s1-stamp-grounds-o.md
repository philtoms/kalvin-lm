---
type: source
title: "Observation: Ratification landed: goal grades proposals; S1 stamp grounds on receipt"
tags:
  - harness
  - engine
  - ratification
  - grading
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-ratification-landed-goal-grades-proposals-s1-stamp-grounds-o
relevance: critical
observed_at: 2026-09-16T17:16:52.546Z
source_context: "Ratification: goal-reached proposals answered at S1, not S4"
---

# 🔴 Observation: Ratification landed: goal grades proposals; S1 stamp grounds on receipt

Landed the ratification fix (the user's last pre-commit fix): the harness no longer answers goal-paired proposals at S4. Two coordinated changes: (1) harness._drive, after _answer returns None and before escalation, calls new _grade_proposal(ask, goals) — a proposal whose head pairs with a `==` goal grades at γ of its content against the goal's TARGET (the goal entry's signature value, NOT its witness nodes — the witness MHALL:[had,ALL] is the underfit key missing Mary; grading against it gave J=4/5→S2): content==target → S1 ratification; off-goal grades low and refuses on re-feed. _drive gained the goals param (3 call sites). The answered-set still swallows re-proposals (terminates). (2) engine._fast_route S1 branch: the is_groundable gate removed — the stamp, not structure, is the licence ('A stamped-S1 query is a ratification: ground on receipt' — the comment's own promise, previously betrayed for the reached-goal answer, a misfit in the question's head that is never structurally groundable); guards: an ask never grounds however stamped, nor an empty unknown. Verified end-to-end: T03 feeds the ask S3, proposes WDMH:[a,little,lamb,had,Mary]; T04 feeds it at S1 and GROUNDS it (persisted state: is_grounded True). Residual noise: one re-proposal (S3 64) in T04's batch from the still-attended ask — answered-set swallows it, no loop. tests/test_ratification.py (5 tests: reached-goal ratifies at S1, off-goal grades below, unpaired None, stamped-S1 grounds a structurally-ungroundable misfit, stamped-S1 ask never grounds); 69/69 green; CONTEXT.md `==` entry states the proposal grading contract.

*Relevance: critical*
*Context: Ratification: goal-reached proposals answered at S1, not S4*
*Tags: harness engine ratification grading*

---
*Observed: 2026-09-16T17:16:52.546Z*
