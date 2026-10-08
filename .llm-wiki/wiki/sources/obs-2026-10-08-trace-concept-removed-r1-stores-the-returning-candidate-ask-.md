---
type: source
title: "Observation: Trace concept removed; R1 stores the returning candidate; ask ping-pongs"
tags:
  - trs
  - spec
  - trace-removed
  - r1
  - returning-candidate
  - rationalise3
  - refactor
status: observation
created: 2026-10-08
updated: 2026-10-08
slug: obs-2026-10-08-trace-concept-removed-r1-stores-the-returning-candidate-ask-
relevance: high
observed_at: 2026-10-08T09:42:53.909Z
source_context: User's opener-free refactor; trace removed from spec
---

# ⭐ Observation: Trace concept removed; R1 stores the returning candidate; ask ping-pongs

TRACE CONCEPT REMOVED FROM SPEC (user ruling: 'Trace concept can be taken out of spec' after their rationalise3 refactor removed the opener dependency). Spec changes: Def 8 trace paragraph DELETED (entry/re-entry distinction stays; 'R1 stores earned along its trace' → 'earned by its candidates'); §4 effort 'how far the trace did not get' → 'how far the rationalisation did not get'; R1 REWRITTEN — header 'store the returning candidate', law block 'a candidate : S1 — directly or through its products ⟹ D := D ∪ { C }', prose: the kline entering D is the returning candidate itself (the opener/constructable-from-the-former contrast deleted with the trace). Only remaining 'trace' is the harness idiom ('scaffolding needed to trace from ask to goal'). Code: rationalise3 docstring refreshed to returning-candidate semantics; S3-emptied end's kline fixed k→c (cosmetic). REFACTOR'S TRACE FACTS (user's opener-free rationalise3): GOAL identical (Subject:[Subject], Verb:[Verb], Object:[Object] + doubled ends: each failing candidate logs product-R4 + candidate-R4); ASK = PING-PONG FLOOD: 128 stores/136 rounds — Mary:[Subject]↔Subject:[Mary] ×60+, had:[Verb]↔Verb:[had] ×30+, then Det:[Det], Mary|had|lamb|a|little:[Det]; 20 ends. The mutual-witness loop: each pairing's split mints its mirror via witness_of, both store, D-liveness amplifies. FLAGS for user: (a) ping-pong duplication flood on ask, (b) double-end pattern (S3-emptied/S2-failed candidates append 'candidate with no adjacency' too), (c) no early-exit divergence noted yet. 159 tests passing.

*Relevance: high*
*Context: User's opener-free refactor; trace removed from spec*
*Tags: trs spec trace-removed r1 returning-candidate rationalise3 refactor*

---
*Observed: 2026-10-08T09:42:53.909Z*
