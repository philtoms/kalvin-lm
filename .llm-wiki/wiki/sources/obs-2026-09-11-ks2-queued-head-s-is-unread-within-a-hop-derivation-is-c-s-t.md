---
type: source
title: "Observation: ks2: queued head s is unread within a hop — derivation is C's two sides"
tags:
  - ks2
  - formalisation
  - derivation
  - signature-inertness
status: observation
created: 2026-09-11
updated: 2026-09-11
slug: obs-2026-09-11-ks2-queued-head-s-is-unread-within-a-hop-derivation-is-c-s-t
relevance: high
observed_at: 2026-09-11T10:26:02.388Z
source_context: Reviewing Def 16 (selection) example in docs/ks2.md; user proposed invariant "A is always a full canon"
---

# ⭐ Observation: ks2: queued head s is unread within a hop — derivation is C's two sides

Exhaustive check of ks2.md §§6–11 confirms: within a derivation hop, the queued kline's signature s is never read — no rule (Def 13), license (Def 14), ending (Def 15), bound (T1/T2), grade (γ) or selection clause (Def 16) consults it; all read only C's two sides: head σ(ν_A) (definitional, exact against ν_A at every state — Identity at one node, Canon beyond; empty when ν_A = []) and target t:ν_B. s survives the hop only via absorb/reentry (end state s:ν_final queues as hop k+1 input) and the claim's eventual grounding. This corrects Phil's initial intuition ("A is always a full canon") — false for queued A's (§9's worked example queues mall:[m,a], a misfit; §13 `=>` results need not be Canon) but true-by-construction for the derivation's A-side. Root cause of the misreading: Def 15's "case 1's s = ∅" overloads s for C's head (σ(ν_A)) vs the queued head. Proposed edits to Def 12 (declare in-hop unreadness) and Def 15 (disambiguate) pending Phil's confirmation. Context: Def 16 example debate — abc in abc:[a] is decoration; band and selection clauses read only ν_A, ν_B.

*Relevance: high*
*Context: Reviewing Def 16 (selection) example in docs/ks2.md; user proposed invariant "A is always a full canon"*
*Tags: ks2 formalisation derivation signature-inertness*

---
*Observed: 2026-09-11T10:26:02.388Z*
