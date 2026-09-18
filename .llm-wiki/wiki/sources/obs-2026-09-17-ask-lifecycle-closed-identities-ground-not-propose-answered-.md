---
type: source
title: "Observation: Ask lifecycle closed: identities ground not propose; answered asks leave attention"
tags:
  - dialogue-dev
  - engine
  - ask-lifecycle
  - is_answered
  - identity-proposals
  - ratification
status: observation
created: 2026-09-17
updated: 2026-09-17
slug: obs-2026-09-17-ask-lifecycle-closed-identities-ground-not-propose-answered-
relevance: critical
observed_at: 2026-09-17T23:32:19.186Z
source_context: "dialogue-dev: identity proposals + re-proposal after ratification"
---

# 🔴 Observation: Ask lifecycle closed: identities ground not propose; answered asks leave attention

dialogue-dev: fixed identity proposals + post-ratification re-proposals in wdmh.ks — both symptoms of one gap: the ask lifecycle had no ask→answer path except through _propose, and no answered-retirement. Changes: (1) engine._propose — a done result that IS the identity or canon of the queued signature grounds (the fact delivers itself, matching is_groundable's identity/canon arms and reentry's "grounded identity delivers itself") instead of being proposed; the groundable-misfit cascade then grounds and retires the ask entry next pass. Kills MHALL:[MHALL] and hadVerb:[hadVerb] identity proposals AND the supervisor-decline/refusal re-feed loop. (2) engine.cogitate retires an ask when state.is_answered; (3) EngineState.is_answered — sharp rule learned the hard way: first cut (any grounded kline at sans-ask sig) wrongly retired the WDMH ask because its signature value IS the 'what' word form (0x80000002f45 = what-bit + tiktoken 'what') and what:[what] grounded at T01 — a tautology. Correct rule: ask-marked only; a CONTENT answer is a grounded witness at the sig whose content ≠ the sig (excludes identities — the 2026-08-21 tautological-answer rule); the identity answers only a BARE word-resolution ask (empty nodes); the ask's own riding canon never counts. Result trace: T01 grounds scaffold + MHALL:[MHALL] identity silently; T02 proposes WDMH:[ALL,had,Mary] S1 255 exactly ONCE; T03 grounds it on receipt + delivers hadVerb identity + cascades hadVerb:[Verb] grounded; work list shrinks to genuine residue (MarySubject, QueryObject, a:[Det], little:[Mod]); asks by band S1=1, zero supervisor declines. Tests 69/69.

*Relevance: critical*
*Context: dialogue-dev: identity proposals + re-proposal after ratification*
*Tags: dialogue-dev engine ask-lifecycle is_answered identity-proposals ratification*

---
*Observed: 2026-09-17T23:32:19.186Z*
