---
type: source
title: "Observation: rationalise2: KNode, candidate fallback, R1 stores first candidate"
tags:
  - trs
  - redesign
  - rationalise2
  - knode
  - fallback
  - opener
  - ask-step
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-rationalise2-knode-candidate-fallback-r1-stores-first-candid
relevance: high
observed_at: 2026-10-06T16:45:04.360Z
source_context: "Points 1-3 on rationalise2: KNode, candidate fallback, opener storage"
---

# ⭐ Observation: rationalise2: KNode, candidate fallback, R1 stores first candidate

THREE MORE POINTS implemented in rationalise2 (user rulings): (1) nodes always KNode — annotations and flows fixed (consume(u: KNode), candidates(head: KNode), bare); (2) CANDIDATE FALLBACK — a work item's candidates iterate in order until one RESOLVES at S1; consume() returns bool; failed candidates leave their stores (knowledge stays) and the parent tries the next; 'no candidate resolved' end when all fail; (3) R1 STORES THE FIRST CANDIDATE — resolve(cand, opener): the S1 arm stores the work item's OPENER even when a later candidate's chain resolves; the ENRICHED push is a NEW work item whose opener is itself (this combination preserves the 1.2 trace: Object/VO/SVO each store their own opener). ASK STEP (MHALL) result — divergence-zone findings: terminates at 35 rounds, 9 stores, 0 ends; recursion descends sentence residuals storing per-level openers incl. VALUE-DUPLICATE Det:[Det] stores (truly-always log; derived heads carry composed provenance labels); final top store = MHALL:[Mary] (the opener, Mary-identity pairing) — the full-sentence MHALL canon was never re-stored (primed; top enrichment only partially intersects MHALL: Det ∉ MHALL bits). TWO SAFETY MECHANISMS the ask forced: global rounds bound in resolve (enriched chain bypassed consume's depth guard — RecursionError at ~1000 frames) and a NO-GROWTH STALL CHECK (a pass whose held-set grew nothing returns False → next candidate) — monotone held growth makes cycles terminate. 159 tests passing. Goal step trace unchanged.

*Relevance: high*
*Context: Points 1-3 on rationalise2: KNode, candidate fallback, opener storage*
*Tags: trs redesign rationalise2 knode fallback opener ask-step*

---
*Observed: 2026-10-06T16:45:04.360Z*
