---
type: source
title: "Observation: Directed S3 walk: honest but answerless without authored reciprocals"
tags:
  - dialogue
  - engine
  - s3
  - walk
  - licence
  - direction
  - fork
status: observation
created: 2026-09-17
updated: 2026-09-17
slug: obs-2026-09-17-directed-s3-walk-honest-but-answerless-without-authored-reci
relevance: high
observed_at: 2026-09-17T15:40:52.157Z
source_context: "dialogue-dev: testing the directed-S3 walk licence"
---

# ⭐ Observation: Directed S3 walk: honest but answerless without authored reciprocals

Tested the user's directed-S3 walk licence: "a slot walk is an S3 licensed walk — node -> sig only; if little:[Mod] exists but Mod:[little] doesn't, the S3 rewrite doesn't happen." Implemented as MODE C (probe_wdmh_s3_walk.py): slot_walk fires reverse replaces only (occurs_rev — whole witness must occur), forward expansion removed; both departures kept; walk_b writes slot-headed. BARE RESULT: zero composed writes, WDMH answer gone — engine honestly stuck. Structural cause: from a single-value departure [what], only klines whose ENTIRE witness is present can contract, i.e. X:[slot] klines — nothing holds `what` as a witness node, so the question's walk dies at departure; what:[Object] points the wrong way (crossing what->Object needs Object:[what], the user's own Mod:[little] example inverted). COMPLETION TEST (probe_wdmh_s3_recip.py): fed ONE authored reciprocal Object:[what] at S1 — then exactly ONE composed write in the whole run (what:[a,little,lamb], the honest bridge: [what] ->rev Object:[what]-> [Object] ->rev ALL:[Object]-> [ALL] ->refine canon-> [a,little,lamb]), targeting contracts [did,have]->had, re-entry derivation consumes the bridge forward: done j1=1.0, WDMH:[a,little,lamb,had,Mary] grounds in frame. No fabrication family, no self-containing writes. Consequence: under the directed rule, denotation direction is real teaching material — W = O licenses only O->W crossings; a question holding `what` cannot walk to its answer unless the script (or compiler reciprocals) authors the way back. Open fork for the user: (1) scripts author both directions, (2) compiler emits reciprocal denotation pairs (un-bans little->Mod via Mod:[little]), (3) depart-hub-only strategy. Also noted: the user's ALL > O < W diagram has arrows INTO O, but compiled ALL:[O], W:[O] under node->sig license crossings OUT of O — direction of `=` teaching needs their ruling.

*Relevance: high*
*Context: dialogue-dev: testing the directed-S3 walk licence*
*Tags: dialogue engine s3 walk licence direction fork*

---
*Observed: 2026-09-17T15:40:52.157Z*
