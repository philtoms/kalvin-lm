---
type: source
title: "Observation: STM dedup insufficient: WDMH loop re-routes via bridge family"
tags:
  - dialogue
  - engine
  - stm
  - dedup
  - wdmh
  - refuted
status: observation
created: 2026-09-17
updated: 2026-09-17
slug: obs-2026-09-17-stm-dedup-insufficient-wdmh-loop-re-routes-via-bridge-family
relevance: high
observed_at: 2026-09-17T14:21:22.812Z
source_context: "dialogue-dev: verifying the STM dedup fix against the WDMH loop"
---

# ⭐ Observation: STM dedup insufficient: WDMH loop re-routes via bridge family

After commit a23d194 (hops tiered into EngineState.stm, add_stm dedup by signature+nodes), re-ran wdmh.ks -p data/dialogue/mhall.json to test the user's hypothesis that dedup would prevent little:[a, Mod, lamb, a]. REFUTED: the proposal still appears (7 done arrivals) plus a NEW sibling proposal little:[Object]. Dedup verified working (goal list holds Det:[a,Mod,lamb] once). Two re-routes: (A) within-hop next-goal consumption — derivations against goals QueryObject:[Object], a:[Det], ALLQuery:[Query] each run ν_B walks writing MORE bridges (STM ends hop with a family of seven Mod:[...] bridges: [a,Mod,lamb,a], [a,Mod,lamb,lamb], [a,Mod,Object,a], [Object], [Det,Mod,lamb,a], [Det], [a,Mod,QueryObject,a]); goal 7 what:[Object] forward-replaces on fresh Mod:[Object] → done → little:[Object]. (B) cross-turn first-goal consumption — from pass 3, top goal Det:[a,Mod,lamb] finds Mod:[a,Mod,lamb,a] already in STM from pass 2, forward replace at entry, done on step 1, once per turn. Conclusion: the duplicate was only load-bearing for the original route; consuming a bridge needs ANY consistent goal, and the walk manufactures bridges against every goal with an excess slot. The bridge family is the engine; the trailing 'a' in [a,Mod,lamb,a] is Def 15's witness+departed-slot construction (decoded: "a little lamb a" — gibberish tail manufactured by the bridge format). Refusal still retires the proposal, never the queued entry.

*Relevance: high*
*Context: dialogue-dev: verifying the STM dedup fix against the WDMH loop*
*Tags: dialogue engine stm dedup wdmh refuted*

---
*Observed: 2026-09-17T14:21:22.812Z*
