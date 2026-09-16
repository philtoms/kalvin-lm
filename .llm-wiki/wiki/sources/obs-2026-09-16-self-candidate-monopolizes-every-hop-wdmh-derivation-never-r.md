---
type: source
title: "Observation: Self-candidate monopolizes every hop; WDMH derivation never runs"
tags:
  - engine
  - hop
  - derivation
  - bug
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-self-candidate-monopolizes-every-hop-wdmh-derivation-never-r
relevance: high
observed_at: 2026-09-16T13:46:17.975Z
source_context: "Harness fitness review; outstanding task: verify WDMH→MHALL through harness"
---

# ⭐ Observation: Self-candidate monopolizes every hop; WDMH derivation never runs

Engine emits zero asks in every harness run (mhall.ks, wdmh-underfit.ks — "asks by band: S1=0 S2=0 S3=0 S4=0"). Root cause chain: Engine._propose builds the hop reservoir via EngineState.where(), which includes the work list — so the queued kline is always its own candidate; candidate_goals (kalvin/hop.py, Def 22) scores γ(A, K=A)=1.0, the maximum, putting the self-goal at the top of every list; Hop.run breaks on the first done; done-at-entry has trace length 1 and is filtered by the engine's len(trace)<2 guard — so the hop ends vacuously and the real goals down the list never get derivations. Fix: Def 22's candidate pool must exclude the queued kline itself (isomorphic by signature+nodes). Secondary engine seam: an arriving S4 Unknown should queue a derivation of its held content (resolve the empty ask to the held canon for its signature as A0), not refuse it. Separately, with the seam bypassed and self-goals excluded (dev/dialogue/probe_hop_wdmh.py), run_hops on A0=WDMH:[what,did,Mary,have] against full mhall+wdmh memory reaches only stuck/abandoned across 24 derivations — the verb contracts to `had` but the object crossover never completes; a zero-valued node KNode(0,'had') appears in stuck traces, violating Def 5's nonzero signature.

*Relevance: high*
*Context: Harness fitness review; outstanding task: verify WDMH→MHALL through harness*
*Tags: engine hop derivation bug*

---
*Observed: 2026-09-16T13:46:17.975Z*
