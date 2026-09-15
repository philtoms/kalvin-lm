---
type: source
title: "Observation: WDMH proposal: 3 fixes landed; entry corruption + proposal flood remain"
tags:
  - derivation
  - engine
  - goal-enumeration
  - walk-bound
  - flood
  - wdmh
status: observation
created: 2026-09-15
updated: 2026-09-15
slug: obs-2026-09-15-wdmh-proposal-3-fixes-landed-entry-corruption-proposal-flood
relevance: high
observed_at: 2026-09-15T05:38:21.949Z
source_context: Diagnosing missing WDMH proposal on wdmh-underfit-o.ks
---

# ⭐ Observation: WDMH proposal: 3 fixes landed; entry corruption + proposal flood remain

Investigated "wdmh-overfit should yield a full WDMH:[m,h,a,l,l] proposal but doesn't". Three changes landed (uncommitted): (1) Engine._select replaced Def 16-occurrence goal enumeration with S2-relationship candidates ordered by ascending misfit mass (Def 14 — the misfit region connects the parties; Def 16 stays evidence-selection inside derive; verified MHALL:[Mary,had,a,little,lamb] enumerates); (2) MAX_WALK_STATES=256 T2-class bound added to slot_walk in BOTH cores — depth-8 BFS over dense engine memory (~60 klines) exploded combinatorially (faulthandler pinned it in occurs_rev Counter construction inside slot_walk); (3) cogitate's is_misfit internal-fit gate replaced by the relational gate — a question kline is canon-shaped (internally S1, well-formed); its misfit lives in C(entry,B) against a held goal. Also verified by injection: with the bare-headed binding what:[Object] grounded, derive(WDMH, MHALL) reaches done with full answer witness [a,little,lamb,had,Mary], J→1.0 (two walks: what→[Object]→[lambObject] composed what:[lambObject]@2, then lambObject→[Object]→[Query]→[ALL]→[a,little,lamb] composed @4). KEY FINDINGS still blocking the proposal: (A) wdmh-underfit-o.ks authors what:[Object] natively (CONNOTES A>B ⇒ A:[B] bare head) — the algebra-shaped binding; the = variant (DENOTES A=B ⇒ AB:[B] compound head) is opaque to node-occurrence and can never feed a slot walk — authoring guidance should prefer > for slot bindings. (B) The WDMH work entry reaches cogitation as WDMH:[DH,what] — content already reduced to what|did|have (Mary lost) via progressed variants re-entering work through the reply loop — against which MHALL is UNCOVERED → S3 → rejected as goal. (C) Proposal flood: 306 derive calls/219 done/21 junk asks (e.g. had:[Mary,had,a,little,lamb,MHALL,Mary]) — every misfit scaffold entry now derives toward every S2 candidate and every done-trace emits; scoping needed (per-entry best proposal, ask-stamped entries only, or supervision filter). (B) and (C) are coupled — the flood's grounded junk is what corrupts the entry. All 44 tests pass; mhall.ks 1.0s; both wdmh scripts ~2s. Emission policy is a design fork for Phil.

*Relevance: high*
*Context: Diagnosing missing WDMH proposal on wdmh-underfit-o.ks*
*Tags: derivation engine goal-enumeration walk-bound flood wdmh*

---
*Observed: 2026-09-15T05:38:21.949Z*
