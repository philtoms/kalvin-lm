---
type: source
title: "Observation: Scoped-memory design landed: Def 22 selects goals, Def 23 scope trawl, re-entry changes A only"
tags:
  - docs
  - kalvin
  - selection
  - scope
  - hop
  - re-entry
  - architecture
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-scoped-memory-design-landed-def-22-selects-goals-def-23-scop
relevance: critical
observed_at: 2026-09-16T11:26:03.551Z
source_context: Implementing the scoped-memory fork with goal selection in kalvin-algebra.md
---

# 🔴 Observation: Scoped-memory design landed: Def 22 selects goals, Def 23 scope trawl, re-entry changes A only

Fork decided and doc reworked per Phil's three-point architecture correction. THE DESIGN: (1) SELECTION SELECTS GOALS, NOT EVIDENCE — Def 22's pool (held klines whose content covers a node of ν_A, Def 8) ordered by descending γ(A, K) is the CANDIDATE GOAL list; the engine's queue strategy picks B from it, re-selected each hop from the current A. MHALL is a candidate goal over WDMH; once B lands, the derivation's memory need (S2|S3 misfit edges joining the parties) is already known. (2) SCOPE = new Def 23: a depth-limited, dual-rooted (A and B) graph trawl over the correspondence graph — "fast but stupid, unranked" — frozen for the hop's duration; writes go to MEMORY (shared reservoir, grows within derivation as Phil said earlier) but the derivation never consumes what it writes — later hops' trawls reach the writes. This makes the hop boundary real: within a hop A's head, B, and scope are fixed; the §9 walk-terminal-consumption pattern strictly spans two hops (appendix line reworded to "in reach of the next hop ... so the trawl finds it"). (3) RE-ENTRY CHANGES A, NEVER B — B is re-selected from the new A (may be the same kline). Strategy loop now five phases: select a goal → scope the memory → derive to an ending → add the result to memory → re-enter (first four = hop, last = chaining). Def 12 gloss rewritten: subscript = scope (fixed as trawled, Def 23) + goal (fixed for duration); writes go to memory not scope. γ orientation in Def 22 changed to γ(A, K) (work direction A→K) — depth-charging question left implicit, flagged to Phil. CONTEXT: Candidates entry retargeted to goals, new Scope entry, Hop entry, Reentry entry (A-only), Cogitation 5 phases, Progressive Path. Probes 7/7. Doc now 23 definitions. NOT yet in code: Derivation(state) still reads full memory — scoping is caller-side future work. Not committed.

*Relevance: critical*
*Context: Implementing the scoped-memory fork with goal selection in kalvin-algebra.md*
*Tags: docs kalvin selection scope hop re-entry architecture*

---
*Observed: 2026-09-16T11:26:03.551Z*
