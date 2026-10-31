---
type: source
title: "Observation: I-2 withdrawn (amendment 17): witnesses coexist, first-stored wins"
tags:
  - trs
  - i2-withdrawn
  - d15
  - witness
  - first-stored
  - amendment-17
  - coexistence
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-i-2-withdrawn-amendment-17-witnesses-coexist-first-stored-wi
relevance: high
observed_at: 2026-10-06T06:00:52.127Z
source_context: Withdrawing I-2; multiple canonical witnesses coexist
---

# ⭐ Observation: I-2 withdrawn (amendment 17): witnesses coexist, first-stored wins

I-2 WITHDRAWN (amendment 17, D15, docs/trs-significance.md): the 'Invariant I-2 — Canonical keys' section deleted; Definition 6 (witness) restated — witness_D(s) returns the FIRST-STORED N with ⟨s,N⟩ ∈ D and can(N) = s, ∅ otherwise; 'A signature may hold several canonical witnesses — C:[AB] alongside C:[A,B] — witness structure the system keeps rather than collapses... selection among held witnesses at use is a deferred choice point.' The fundamental law can(witness_D(s)) = s unchanged. Code: NO behavioural change needed — Domain's index already keeps first-stored canonical adjacency per head (setdefault); comments updated to cite D15. Tests: closed_domains generator no longer enforces I-2-validity (domains may hold several canonical witnesses per head — the property suite now exercises coexistence); test_i2_static_domains REPLACED by test_witness_returns_first_stored (coexistence detected in generated domains → first-stored wins, law holds); model doctrine test test_multiple_canonical_witnesses_coexist (the old seam fixture k(3,{3}) + k(3,{1,2}) now LEGAL and asserted). 155 tests passing. Rationale (user): the R1 rewrite/reentry removed the reason I-2 existed — two canons under a head are two original understandings, not a minting collision; the nuance (C:[AB] coarse vs C:[A,B] decomposed — different substitution surfaces) is witness structure to keep. NEXT FOCUS (user): hook up the dialogue harness and scripted runs — nail that BEFORE continuous operation. Uncommitted.

*Relevance: high*
*Context: Withdrawing I-2; multiple canonical witnesses coexist*
*Tags: trs i2-withdrawn d15 witness first-stored amendment-17 coexistence*

---
*Observed: 2026-10-06T06:00:52.127Z*
