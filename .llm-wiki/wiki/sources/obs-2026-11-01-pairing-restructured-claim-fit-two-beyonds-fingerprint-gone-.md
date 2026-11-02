---
type: source
title: "Observation: Pairing restructured: claim-fit, two beyonds — fingerprint gone from _pair"
tags:
  - ruling
  - pairing
  - claim-fit
  - split-beyonds
  - fingerprint-removed
  - semantics
status: observation
created: 2026-11-01
updated: 2026-11-01
slug: obs-2026-11-01-pairing-restructured-claim-fit-two-beyonds-fingerprint-gone-
relevance: critical
observed_at: 2026-11-01T15:06:20.744Z
source_context: Implementing the claim-fit + split-beyonds pairing per the user's ruling
---

# 🔴 Observation: Pairing restructured: claim-fit, two beyonds — fingerprint gone from _pair

THE FINGERPRINT LEFT _pair. Final pairing structure (user-ruled after measuring that across the whole corpus: F comes from the claim half in 509/509 rows; the decomposition half NEVER trims U; O needs both beyonds — 265 claim-extension rows, 28 decomposition-offering rows, exactly 1 mixed): F = E ∩ c.sig and U = E ∖ c.sig (the entry meets the CLAIM — signatures fit, not just select); the o-side is TWO separate reentries — the claim's extension (σ∖E) and the decomposition's offering (can(N)∖E) — deduped when both name the same content; row.O remains only as the displayed/settled region (the sum). _side_ok now requires ALL of a side's children resolved (was: any). Frame mints when E == F (the claim covers the entry), value E. No σ∪can(N) value is ever constructed, minted, or reentered: the one mixed corpus row (wdmh's HALL × the stored claim ⟨MHALL,{SVO}⟩) now carries two pure o-children — Subject|Verb|Object and Mary — where the old code minted the composite 0xe100000000 (the MHSVO shape); 7 of 181 rows have split beyonds; ZERO composite frames in the run. Corpus reproduces byte-identically: mhall grounds with the same table, wdmh grounds both scenarios with the same tables (incl. the DH exchange closing). First implementation had a duplicate-children bug (both beyonds computing the same region, e.g. Subject from claim AND decomposition of MS:Subject → two identical reentries → nothing grounded) — fixed by one-reentry-per-content. Suite: 162 passed, 15 skipped, 16 xfailed; runtime ~3x slower (10s vs 3s) — the split children explore more reentries. side() deleted from the model (user); like-for-like bar = c.fingerprint() != k.sig; proven-skip on claim-cover + pre-settled beyonds. DOCTRINE FINAL FORM: signatures prime and fit; adjacencies offer (their composition is reentry material); fingerprints control loops (seen-dedup, the like-for-like bar) — never constructed, never minted, never reentered.

*Relevance: critical*
*Context: Implementing the claim-fit + split-beyonds pairing per the user's ruling*
*Tags: ruling pairing claim-fit split-beyonds fingerprint-removed semantics*

---
*Observed: 2026-11-01T15:06:20.744Z*
