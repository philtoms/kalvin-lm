---
type: source
title: "Observation: Word-bit trawl, scaffold-first groups, depth 5 landed; scope now full reservoir"
tags:
  - trawl
  - scope
  - harness
  - feed-order
  - engine
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-word-bit-trawl-scaffold-first-groups-depth-5-landed-scope-no
relevance: high
observed_at: 2026-09-16T16:26:08.669Z
source_context: Landing word-bit trawl + scaffold-first feed order + depth 5
---

# ⭐ Observation: Word-bit trawl, scaffold-first groups, depth 5 landed; scope now full reservoir

Landed three engine/harness changes: (1) Def 23 trawl touch test is now word-bit intersection — a kline scopes when its signature or any node shares a word bit with the reached set (mask WORD_BITS=0x7FFFFFFF00000000; reach is an OR-mask accumulated from roots and scoped klines; token-id bits carry no correspondence) — aligning the trawl with Def 22's atom-level coverage; CONTEXT.md Scope entry states the edge semantics. (2) TRAWL_DEPTH 4→5 — a:[Det]/little:[Mod] land at round 5 through the role web (MarySubject→Subject→SVO→Object→lamb→...). (3) Harness: scaffold groups open before ask groups (stable sort on opener dbg.op=='ASK' over the steps list, after the no-look-ahead pools rule) — the trainer primes K before asking; CONTEXT.md Scaffolding entry notes it. New test test_trawl_touches_at_word_bits (connoted compound scopes via shared word bit; low-bit-only overlap never touches); 59/59 pass. Verified live: wdmh feed order is now scaffold group first (ALLO/whatO/ALL:[A,L,L] then had:[did,have]) then question group (goal+identities+canon, then the ask at T03 S3); the ask's FIRST hop sees had:[did,have] in the reservoir as top goal (γ=0.5) with MHALL:[Mary,had,A,L,L] second (0.14) — the worked example's step order — and the scope is the full 18-kline reservoir with structured rounds. Still open: derivations on the rich scope end without emitting the ask (the stuck→ask seam in _propose remains the known next step); no escalations.

*Relevance: high*
*Context: Landing word-bit trawl + scaffold-first feed order + depth 5*
*Tags: trawl scope harness feed-order engine*

---
*Observed: 2026-09-16T16:26:08.669Z*
