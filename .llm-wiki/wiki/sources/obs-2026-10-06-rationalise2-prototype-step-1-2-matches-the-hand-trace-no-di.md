---
type: source
title: "Observation: rationalise2 prototype: step 1.2 matches the hand-trace, no divergence"
tags:
  - trs
  - redesign
  - step-1-2
  - rationalise2
  - witness-accumulation
  - recursion
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-rationalise2-prototype-step-1-2-matches-the-hand-trace-no-di
relevance: high
observed_at: 2026-10-06T16:17:18.793Z
source_context: Implementing step 1.2 redesign prototype; verifying against the hand-trace
---

# ⭐ Observation: rationalise2 prototype: step 1.2 matches the hand-trace, no divergence

STEP 1.2 IMPLEMENTED (prototype `rationalise2` in src/trs/model.py, exported): the redesigned R2 consumption — r2_products consumed as ONE response; F accumulates the witness (starts as the fit-producing candidate's adjacency; after U's recursion returns, every kline the recursion STORED whose sig ⊆ σ contributes its ADJACENCY to F — 'add them to F' read as adding stored witnesses); U is RECURSIVELY APPLIED (its value is the head, one candidate from neighbour overlap, dispatch by band); S1 stores ALWAYS (idempotent adds); enriched F is pushed and iterated. Single-threaded per the user's trace: first candidate per work kline (sig-ascending for now — ordering is a noted non-algebraic placeholder); remaining candidates are a DEFERRED CHOICE POINT. RESULT: NO DIVERGENCE — mhall goal step reproduces the hand-trace exactly: 3 rounds, 0 ends, stores in order [Object:[Object], VO:[Verb,Object], SVO:[Subject,Verb,Object]] (heads render with composed provenance labels, values are correct). Pinned by abstract test test_rationalise2_accumulates_witness_bottom_up (bits S=1,V=2,O=4, three connotation-shaped primed klines). SIDE FIX (latent flake found during the run, not caused by the redesign): test_witness_returns_first_stored iterated the Domain SET (not insertion order) to define 'first-stored' — hypothesis found a case where set order ≠ insertion order. Fixed by making Domain insertion-ordered: _order list alongside the set, __iter__ yields insertion order (first-stored is spec-meaningful per Definition 6 and now observable). 159 tests passing. rationalise (old) untouched — harness still runs it; switch-over deferred.

*Relevance: high*
*Context: Implementing step 1.2 redesign prototype; verifying against the hand-trace*
*Tags: trs redesign step-1-2 rationalise2 witness-accumulation recursion*

---
*Observed: 2026-10-06T16:17:18.793Z*
