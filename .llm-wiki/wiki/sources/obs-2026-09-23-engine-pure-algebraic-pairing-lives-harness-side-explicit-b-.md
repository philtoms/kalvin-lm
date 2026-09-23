---
type: source
title: "Observation: Engine pure-algebraic: pairing lives harness-side; explicit-B derivations marked (given)"
tags:
  - engine
  - purity
  - pairing
  - harness
  - protocol
  - def12
  - given-selected
status: observation
created: 2026-09-23
updated: 2026-09-23
slug: obs-2026-09-23-engine-pure-algebraic-pairing-lives-harness-side-explicit-b-
relevance: critical
observed_at: 2026-09-23T13:20:20.840Z
source_context: Making the engine pure-algebraic; pairing to the harness
---

# 🔴 Observation: Engine pure-algebraic: pairing lives harness-side; explicit-B derivations marked (given)

Engine made pure-algebraic per Phil's ruling: the == pairing decision moved entirely to the harness (protocol side). Changes: (1) hop.candidate_goals reverted to pure Def 22 (coverage pool, γ-order — scored_candidates deleted, no dbg.goal reading anywhere in src/kalvin — remaining dbg uses are label/presentation only); (2) Hop/run_hops/cogitate gained an explicit goal parameter — a B-parameterised derivation is pure Def 12 (A ⊢_{M,B} is indifferent to how B arrived), labelled goal_source="given"; engine-selected derivations are "selected"; (3) harness._drive now runs two phases per feed: the pure work-list cogitation (engine alone, Def 22 selection), then the paired phase — for each attending ask with a declared == goal, cogitate(state, ask, goal=goal.kline) hands B explicitly; the goals map (canon_key→goal) was already harness-side. Verified mhall T02 trace now shows the separation cleanly: pure passes (MHALL→ALL selected stuck j1=0.600, canonicalised re-entry [Mary, had, ALL]→ALL selected stuck — the engine alone honestly stalls), then (given) derivations: entry-state stuck j1=0.000, prepared-state done j1=1.000 → proposes MHALL:[Subject, Verb, Object] S1 255; wdmh unchanged (WDMH:[ALL, had, Mary] S1 255); 90/90 tests. The trainer's hand is now visible by construction: (selected) = the algebra alone; (given) = B handed over the protocol boundary.

*Relevance: critical*
*Context: Making the engine pure-algebraic; pairing to the harness*
*Tags: engine purity pairing harness protocol def12 given-selected*

---
*Observed: 2026-09-23T13:20:20.840Z*
