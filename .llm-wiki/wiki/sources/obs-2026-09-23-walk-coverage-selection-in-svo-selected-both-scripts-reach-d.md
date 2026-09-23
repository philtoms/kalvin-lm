---
type: source
title: "Observation: Walk-coverage selection in: SVO selected, both scripts reach done — with cascade + head-adoption side-effects"
tags:
  - def22
  - walk-coverage
  - selection
  - witness-paths
  - mhall
  - wdmh
  - side-effects
status: observation
created: 2026-09-23
updated: 2026-09-23
slug: obs-2026-09-23-walk-coverage-selection-in-svo-selected-both-scripts-reach-d
relevance: critical
observed_at: 2026-09-23T14:22:40.530Z
source_context: Implementing the walk-coverage selection rule
---

# 🔴 Observation: Walk-coverage selection in: SVO selected, both scripts reach done — with cascade + head-adoption side-effects

Implemented Phil's walk-coverage selection rule (Def 22 amended): a held kline K joins A's candidate list iff EVERY node of ν_K is covered from ν_A — a node of ν_A, or the end of a witness path over held klines walked in EITHER direction (bounded 8). First attempt used Def 15's A-side step restrictions (single-node descents) for the closure — too strict: wdmh went silent (A/L unreachable from what/did/Mary/have) and the §10 toy example broke. Correct shape: selection coverage is UNDIRECTED bipartite reachability over held correspondences (node ~ kline iff node in witness OR σ(head) contains node); Def 15's restrictions bind derivation STEPS, not selection reachability. hop.walk_closure rewritten as bounded reachability; doc Def 22 reworded accordingly. VERIFIED: mhall selects SVO purely (chains MS→S, HV→V, ALL→ALLQ→Query→QO→Object all covered) and reaches done: `MHALL:[Mary, had, lamb] → SVO done j1=1.000` → proposes MHALL:[SVO, Verb, Object] S1 255, ratified, grounds at T05. wdmh restored: WDMH:[what, had, Mary] → MHALL done j1=1.000 → proposes WDMH:[ALL, had, Mary] S1 255. TWO SIDE-EFFECTS awaiting Phil's ruling: (1) scaffold self-derivation cascade — undirected closure lets scaffolds select each other (Subject ∈ SVO-canon witness covers Object → MarySubject selects lamb), T01 proposes ALLQuery:[Object]/MarySubject:[Object]/hadVerb:[Query], escalations decline them at S4, but the cascade composes STM bridges MarySubject:[SVO], hadVerb:[SVO] at T02-T03; (2) head-atom adoption — the ask's first slot adopts B's head atom ([SVO, Verb, Object] not [Subject, Verb, Object]) via the pre-composed MarySubject:[SVO] bridge (one-node adoption of the whole overfit is Def 14 reverse-legal); content-equal by σ (claim-blind γ/done), structurally the shortcut not the walk. One test still failing: test_selection_orders_goals_by_overlap encodes the old Def-8 rule — to update once the above rulings land.

*Relevance: critical*
*Context: Implementing the walk-coverage selection rule*
*Tags: def22 walk-coverage selection witness-paths mhall wdmh side-effects*

---
*Observed: 2026-09-23T14:22:40.530Z*
