---
type: source
title: "Observation: Hop rebuilt as Def 21: derivation-whole, fixed (M,B), write/consume boundary for re-entry"
tags:
  - docs
  - kalvin
  - hop
  - re-entry
  - strategy
  - def21
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-hop-rebuilt-as-def-21-derivation-whole-fixed-m-b-write-consu
relevance: high
observed_at: 2026-09-16T10:43:14.138Z
source_context: Making Hop a substantial definition to ground re-entry work
---

# ⭐ Observation: Hop rebuilt as Def 21: derivation-whole, fixed (M,B), write/consume boundary for re-entry

Reworked §11 around a substantial Hop definition, per Phil's push (he needs solid re-entry foundations). Chose "substantial claim" over removing hops. The circularity he found: "select a hop" in the loop vs hop-as-container-of-selection; and defining hop as "one pass of the loop" is rootless once the loop says "select a candidate". Resolution — direction reversed, hop is primary: **Def 21 — Hop**: "a derivation taken as a whole: its selection at entry, its steps, its ending, and the writing of its result to memory. The derivation's parameters — the memory state and the goal — are fixed for the hop's duration (Definition 12) and change only between hops. Whatever a hop writes is therefore available to later hops alone: memory may grow between hops, never within one." THE KEY RELATIONSHIP for re-entry: a derivation cannot consume its own output (M fixed, Def 12), so the hop is the write/consume boundary — write at hop k, consume at hop k+1 (e.g. §9's walk terminal w:[a,l,l] is strictly consumed by a LATER hop). Loop now: select a candidate → derive to an ending → add the result to memory → re-enter; "The first three phases are a hop (Definition 21); the last queues its result as the next hop's input." Def 12 gained the grounding sentence: "The subscript names the derivation's parameters. M and B are fixed for the derivation's duration..." Re-entry reworked: endings-based ("Every ending — done, stuck, or abandoned (Definition 16) — leaves a result..."), killing the odd "otherwise selected hop" phrase; hop k parameterised (M_k, B_k), produces next queued state = "the kline its derivation ends on, or a correspondence written along the way". Selection renumbered 21→22. CONTEXT: Hop entry rewritten substantially, Cogitation loop phase "select a hop"→"select a candidate", Candidates (Def 22). Maude: "hop selection"→"loop control" in outside-the-algebra note. Probes 7/7 PASS. Doc now 22 definitions. Not committed.

*Relevance: high*
*Context: Making Hop a substantial definition to ground re-entry work*
*Tags: docs kalvin hop re-entry strategy def21*

---
*Observed: 2026-09-16T10:43:14.138Z*
