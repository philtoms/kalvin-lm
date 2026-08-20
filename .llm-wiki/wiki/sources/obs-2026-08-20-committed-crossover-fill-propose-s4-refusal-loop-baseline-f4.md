---
type: source
title: "Observation: Committed crossover-fill propose + S4 refusal loop baseline (f48112d)"
tags:
  - dialogue
  - engine
  - commit
  - baseline
status: observation
created: 2026-08-20
updated: 2026-08-20
slug: obs-2026-08-20-committed-crossover-fill-propose-s4-refusal-loop-baseline-f4
relevance: critical
observed_at: 2026-08-20T09:50:55.158Z
source_context: Committing the dialogue engine session work
---

# 🔴 Observation: Committed crossover-fill propose + S4 refusal loop baseline (f48112d)

Committed f48112d on branch dialogue: merged propose/propose_gap into single propose(entry) with crossover-connotation fills (edge-hop chains from entry nodes + underfit-gap covering bridges, canon traversal); deleted ExpandFit._expand entirely. Harness: terminal-only identities, batched turn replies, dedup, run-to-completion with S4 rejection returns, proposes vs asks in trace, spectrum-based band classification. Engine: S4 route records refusal (EngineState.refused set) and removes the exact kline from STM; cogitate promotes groundable+denoted entries via shared Engine._ground. Baseline: mhall.ks completes S2=2 S4=25, wdmh-underfit.ks completes S2=7 S4=26, both with canonical stores grounded (incl. WDMH via countersign cascade). Deferred: pairing alignment in countersignature (currently commented out of cogitate), multi-proposal emission (K fires every graded fill), proposal ratification fork (T answering proposal asks with containing script kline), refused-set lifetime scoping, S1 ask (WDMH==MHALL resolution).

*Relevance: critical*
*Context: Committing the dialogue engine session work*
*Tags: dialogue engine commit baseline*

---
*Observed: 2026-08-20T09:50:55.158Z*
