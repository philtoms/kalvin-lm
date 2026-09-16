---
type: source
title: "Observation: WDMH ask: zero candidates (empty nodes); canon form orders verb→answer"
tags:
  - engine
  - hop
  - selection
  - ask
  - candidates
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-wdmh-ask-zero-candidates-empty-nodes-canon-form-orders-verb-
relevance: high
observed_at: 2026-09-16T14:48:20.671Z
source_context: WDMH candidate generation investigation
---

# ⭐ Observation: WDMH ask: zero candidates (empty nodes); canon form orders verb→answer

Empirically answered 'what candidates does WDMH generate' (dev/dialogue/probe_wdmh_candidates.py, post-run state of mhall→wdmh): (1) The empty ask WDMH:[] generates ZERO candidates — structural, not a selection bug: Def 22's coverage pool is read on the queued kline's NODES ('held klines whose content covers a node of the queued witness'); empty nodes → nothing to cover → empty goal list → hop stuck before any derivation. This is the algebra's honest 'no goal held — the ask'. (2) Counterfactual: the withheld canon WDMH:[what,did,Mary,have] as A₀ generates a correctly-ordered list: γ=0.500 had:[did,have] (the verb bridge — algebra Steps 1–2), γ=0.143 MHALL:[Mary,had,A,L,L] (the answer's full canon — Step 3's goal B), everything else 0.000. Notably the answer-key entry MHALL:[had,ALL] is NOT a candidate (its content {had,A,L} shares no atom with the question's word nodes — node-level coverage); selection picks the answer's CANON form, matching the worked example's B=mhall:[m,h,a,l,l]. (3) Machinery gap confirmed as the user suspected: all four question words grounded (what/did/Mary/have), the withheld canon is is_groundable=True (any canon is), yet nothing in the engine queues or synthesizes it — the old bridge (harness fed the MTS canon at S1 + cogitator's find_canon(entry.signature) query resolution) is gone on both ends. Fork for restoring: (A) engine emits stuck-on-empty as a signature-discovery ask → harness answers from pools with the canon (dialogue protocol); (B) engine synthesizes an ungrounded proposal under the ask — held identities whose atoms partition the ask signature compose the canon proposal (self-checking by signature_of(nodes)==signature), ratified by the supervisor like any proposal. (B) matches 'the ask is the halt signal under which strategy generates ungrounded proposals'.

*Relevance: high*
*Context: WDMH candidate generation investigation*
*Tags: engine hop selection ask candidates*

---
*Observed: 2026-09-16T14:48:20.671Z*
