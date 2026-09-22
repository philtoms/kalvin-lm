---
type: source
title: "Observation: mhall.ks training intent decoded: Query stress test, goal/evidence split"
tags:
  - kscript
  - mhall
  - curriculum
  - training-intent
status: observation
created: 2026-09-22
updated: 2026-09-22
slug: obs-2026-09-22-mhall-ks-training-intent-decoded-query-stress-test-goal-evid
relevance: high
observed_at: 2026-09-22T12:58:47.143Z
source_context: Analysing data/scripts/mhall.ks structure-to-intent with Phil
---

# ⭐ Observation: mhall.ks training intent decoded: Query stress test, goal/evidence split

Phil's authorial answers decoding data/scripts/mhall.ks training intent: (1) `Query` in `Object < Query < ALL` is a deliberate stress test — it blocks immediate resolution of the object slot and stresses the hop logic (grading already measures this: 2026-08-18 obs, W>O = 0xfc 1 hop vs W>Query>O = 0xfb 2 hops). (2) The two L bindings (little=Mod, lamb=O) are a compiler binding challenge, not teaching content. (3) The chained `Object < Query < ALL` syntax is a compiler parsing challenge, not training semantics. (4) The S4 ask MHALL:[] carrying its own words [Mary,had,a,little,lamb] is intended: a simple, canonical task where difficulty is arrangement, not word recovery. (5) Goal/evidence split: SVO:[Subject,Verb,Object] is the objective goal; SVO:[Mary had a little lamb] is the subjective evidence — where the session agent looks to see how K is doing.

*Relevance: high*
*Context: Analysing data/scripts/mhall.ks structure-to-intent with Phil*
*Tags: kscript mhall curriculum training-intent*

---
*Observed: 2026-09-22T12:58:47.143Z*
