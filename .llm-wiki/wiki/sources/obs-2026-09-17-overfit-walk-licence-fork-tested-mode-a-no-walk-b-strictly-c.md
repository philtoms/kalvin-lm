---
type: source
title: "Observation: Overfit-walk licence fork tested: MODE A (no walk_b) strictly cleaner"
tags:
  - dialogue
  - engine
  - walk
  - licence
  - overfit
  - fork
status: observation
created: 2026-09-17
updated: 2026-09-17
slug: obs-2026-09-17-overfit-walk-licence-fork-tested-mode-a-no-walk-b-strictly-c
relevance: high
observed_at: 2026-09-17T14:58:47.786Z
source_context: "dialogue-dev: testing the overfit-walk licence fork"
---

# ⭐ Observation: Overfit-walk licence fork tested: MODE A (no walk_b) strictly cleaner

Tested the user's theory "the overfit walk requires the same licence as the underfit walk" on wdmh.ks -p mhall.json, via dev/dialogue/probe_wdmh_walk_licence.py (MODE env var, no engine source changed). Verified first: no historical Det⇄Object bridge ever existed (goal list shows Det only via a:[Det], Object only via ALL/lamb/QueryObject/what:[Object]); and no S2 operation can mint a new (signature,witness) pairing — canonicalisation/targeting only rewrite nodes; only the Def 15 composed write mints edges. MODE A (b_walks=False — no overfit walk; each party walks only its own slots): all junk gone (a:[Object], little:[Object], little:[a,Mod,lamb,a]); step 2 proposes nothing (baseline's 3 were all declined); only 3 composed writes in the whole run, all honest slot:[arrival] shapes headed at the departed slot (what:[a,little,lamb], what:[Det,little,lamb], what:[a,Mod,lamb]); WDMH answer WDMH:[a,little,lamb,had,Mary] unchanged; grounded/frame/work_list identical to baseline. MODE B (walk_b runs but writes KLine(slot, end) instead of anchor:[witness+slot]): trace surface identical to A, but 604 composed writes computed (230x the same kline — re-derivation waste) and a new self-containing family appears (lamb:[Det,little,lamb], lamb:[a,little,lamb] — head survives into its own arrival), still minting klines headed at the GOAL's nodes. Reading that fits "same licence": the underfit licence is that the departed slot is a node of the QUEUED kline's own witness; excess atoms live in the goal's witness, so under the same licence A can never depart an overfit slot — the overfit walk belongs to the queued kline holding those nodes (everything in the work list gets its own hop). This matches the user's S3 model: the bridge forms from both ends, meeting at shared values ("only these klines can meet at O"). MODE A recommended; requires a Def 15/docs change plus flipping b_walks.

*Relevance: high*
*Context: dialogue-dev: testing the overfit-walk licence fork*
*Tags: dialogue engine walk licence overfit fork*

---
*Observed: 2026-09-17T14:58:47.786Z*
