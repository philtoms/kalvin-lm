---
type: source
title: "Observation: WDMH→MHALL done end-to-end: zero-sig bug, denotations, re-entry"
tags:
  - derivation
  - engine
  - compiler
  - bugfix
  - milestone
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-wdmh-mhall-done-end-to-end-zero-sig-bug-denotations-re-entry
relevance: critical
observed_at: 2026-09-16T16:48:22.121Z
source_context: WDMH→MHALL derivation brought to done end-to-end
---

# 🔴 Observation: WDMH→MHALL done end-to-end: zero-sig bug, denotations, re-entry

ACHIEVED the outstanding goal: the harness running mhall.ks then wdmh-underfit.ks now gets Kalvin to propose MHALL's content for WDMH. Three root causes found and fixed for 'why not done': (1) **Compiler zero-sig bug** (src/ks/token_encoder.py): `is_compound_def = op==CANONICALISES and len(sig)>1` diverted lowercase multi-char sigs to the deferred-compute path, but step 3 only computes isupper() compounds — `h(ad) => did have` (sig resolves to the word 'had') minted signature 0. The atom vanished from every contracted state ([what,had(0x0),Mary] — content missing the had bit → J capped at 0.75 → never done). This was the 'zero-valued node KNode(0,had) in stuck traces' from the first session — never a derivation-layer bug. Fix: is_compound_def requires entry.sig.isupper(); 'had' now mints 0x1000000178 = the same atom as the MHALL canon's node (word continuity). (2) **Script crossover granularity**: `ALL > O < W` connotes compile to compound sigs (ALLO:[O], whatO:[O]) — opaque to the walk's value-exact occurrence ('occurrence reads nodes, never the atoms within them' — deliberate algebra). The §9 worked example memory needs DENOTATIONS w:[o], all:[o]. Script now: `W = O(bject)` + `ALL = O(bject)`. The walk then composes what:[A,L,L] exactly as the algebra's w:[a,l,l] (3 edges: what→Object→ALL→[A,L,L]). (3) **No re-entry**: _propose ran one Hop — the composed write is only consumable by a LATER derivation of the same queued kline. Switched _propose to run_hops (Def 21's chain, previously unused); the done-guard now keys on 'final state differs from the original nodes' instead of per-derivation trace length (done-at-entry of a re-entry state IS the answer). Verified: hop 1 on the ask (verb contract + compose, stuck) → hop 2 on WDMH:[what,had,Mary] (goal=MHALL canon) → done j0=0.4→j1=1.0 → proposal WDMH:[A,L,L,had,Mary] at γ byte 74 (S3) → supervisor declined (S4 stub) → T04 re-feeds it as an ask. Monitor: dev/dialogue/probe_derivation_monitor.py (memory prior + per-goal endings/traces). Known wart: the `==` goal entry MHALL:[had,what,ALL] still carries `what` (block canon = body heads; the W denotation pollutes the answer key) — harmless here (the held canon goal outranks it) but script- or compiler-side fixable. 59/59 tests green.

*Relevance: critical*
*Context: WDMH→MHALL derivation brought to done end-to-end*
*Tags: derivation engine compiler bugfix milestone*

---
*Observed: 2026-09-16T16:48:22.121Z*
