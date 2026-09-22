---
type: source
title: "Observation: mhall stall root: ask dies at S4-refusal; attending ask still derives nothing"
tags:
  - mhall
  - ask
  - refusal
  - def22
  - stall
  - rationaliser
status: observation
created: 2026-09-22
updated: 2026-09-22
slug: obs-2026-09-22-mhall-stall-root-ask-dies-at-s4-refusal-attending-ask-still-
relevance: high
observed_at: 2026-09-22T13:47:17.087Z
source_context: Diagnosing the silent canonical mhall run
---

# ⭐ Observation: mhall stall root: ask dies at S4-refusal; attending ask still derives nothing

Anatomy of the mhall stall at HEAD (genuine rationaliser work, not regression): (1) The MHALL ask is graded by the harness at J(MHALL, SVO)=0 — ask content (Mary,had,a,little,lamb word bits) and goal content (Subject,Verb,Object) are disjoint — so gamma_to_byte(0)=0x00 → S4 → the rationaliser fast path refuses it and removes it from the work list. The question never cogitates, and it is the only kline whose trawl roots (the sentence words) reach the S2 connotation bridges (MarySubject:[Subject] head carries Mary's bit). (2) Counterfactual probe (dev/dialogue/probe_mhall_ask_attends.py): feeding the ask at S3 so it attends still yields zero emissions — derivations stuck. Why: under Def 22 the SVO canon never becomes a goal for the ask (its content covers none of the ask's nodes); the only legitimate goal is the ALL canon, toward which connotation moves are not progress. The S2/S3 scaffolds queue with goals (SVO canon + sibling connotations) but no licensed bridge exists in their [Subject]-rooted scopes — Mary/had bits unreachable. Per training intent (goal/evidence split), the expected composition is the evidence kline SVO:[Mary, had, a, little, lamb] via canonicalisation ([a,little,lamb]→ALL), connotation application (Mary⇉[Subject] etc.), 2-hop Query descent (ALL→Query→Object). Open semantics fork: should an ask-marked kline be refusable on feed at all, or must asks always attend and let cogitation emit them for the harness to answer?

*Relevance: high*
*Context: Diagnosing the silent canonical mhall run*
*Tags: mhall ask refusal def22 stall rationaliser*

---
*Observed: 2026-09-22T13:47:17.087Z*
