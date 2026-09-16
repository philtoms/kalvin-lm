---
type: source
title: "Observation: Hop layer implemented: selection, trawl, frozen scope, two-hop §9 completion with Ĥ=9/5"
tags:
  - kalvin
  - implementation
  - hop
  - selection
  - scope
  - re-entry
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-hop-layer-implemented-selection-trawl-frozen-scope-two-hop-9
relevance: critical
observed_at: 2026-09-16T12:36:33.563Z
source_context: Implementing Defs 21-23 strategy model in src/kalvin
---

# 🔴 Observation: Hop layer implemented: selection, trawl, frozen scope, two-hop §9 completion with Ĥ=9/5

Committed docs (4c82085), then implemented the strategy model in code. NEW src/kalvin/hop.py: candidate_goals (Def 22 — Def-8 coverage pool over held klines, J-descending order since γ(A,K)'s depth factor is constant in K; goal taken from top), trawl (Def 23 — dual-rooted BFS over the correspondence graph from A's+B's nodes, depth-bounded, terminals excluded, "fast but stupid"), Hop (Def 21 — derivations down the goal list, per-goal scope trawled fresh — so later derivations within a hop see earlier writes via their trawls; hop ends done/exhausted(stuck)/goal-bound(abandoned)), run_hops (re-entry chain, hop ceiling MAX_HOPS). KEY DESIGN DECISIONS MADE DURING TRACING: (1) re-entry default = ending state of the derivation that WROTE (evidence-building route), else last derivation — argmax-j1 is WRONG because j1 is goal-relative (a canon-goal's .5 isn't progress toward mhall); traced WDMH through both defaults to prove it. (2) Stall guard counts writes as progress: the overfit appendix case completes same-A-with-grown-memory (hop 2 reselects same top goal mhall, now finds m:[m,all] in scope). Derivation.py CHANGES (frozen scope, Def 12/23): removed _ground_composed's memory.append + targetings' '+ self.composed' (a derivation never consumes what it writes); _record_arrival cost now k.acq_depth-or-band; _walk guarded on empty excess (Def 15: walk bridges to the overfit). GOLDEN RESULTS: run_hops(_memory, wdmh) completes §9 in 2 hops with the done derivation reproducing §10's example EXACTLY (j1=1.0, Ĥ=9/5, γ=2^-9/5); run_hops(overfit) completes the appendix same-A. Tests: test_derivation.py updated to frozen-scope/hop model, new tests/test_hop.py — 51 pass; probes 7/7 (self-contained, untouched). Dialogue layer UNAFFECTED: engine.py imports dialogue.derivation (separate port) — future re-entry wiring point. Uncommitted: hop.py, derivation.py, both test files.

*Relevance: critical*
*Context: Implementing Defs 21-23 strategy model in src/kalvin*
*Tags: kalvin implementation hop selection scope re-entry*

---
*Observed: 2026-09-16T12:36:33.563Z*
