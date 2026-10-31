---
type: source
title: "Observation: TRS identities filtered from candidates — self-confirmation suppressed"
tags:
  - trs
  - rationalise
  - redesign
  - identity-filter
  - self-confirmation
status: observation
created: 2026-10-16
updated: 2026-10-16
slug: obs-2026-10-16-trs-identities-filtered-from-candidates-self-confirmation-su
relevance: high
observed_at: 2026-10-16T14:17:44.286Z
source_context: "TRS rationalise redesign, step 2: identity neighbours filtered from candidates"
---

# ⭐ Observation: TRS identities filtered from candidates — self-confirmation suppressed

Redesign step 2 implemented in src/trs/model.py (uncommitted, on top of consolidation). In rationalise: nbs = [nb for nb in d.overlap(entry.sig) if nb.adj != frozenset([nb.sig])] — identity neighbours (⟨s,{s}⟩, same predicate as R3's like-for-like filter) are excluded before candidate generation. Rationale: identities let an entry SELF-CONFIRM — the mhall ask ⟨MHALL,{Mary,had,a,little,lamb}⟩ grounded its own store purely through word identities Mary:[Mary] etc. With the filter, grounding must come from cross-structure (canons like ALL:[a,little,lamb], claims like MarySubject:[Subject]). Effects on mhall.ks: step 1 (goal SVO) unchanged — grounds via claims; step 2 no longer stores MHALL — admits the ALL fit and raises Mary|had:[Subject]/[Verb] acquire asks at E=5/6 plus the MHALL-level category asks. Atoms in test base_domain() are identities, so 4 step-1 tests rewritten to cross-canon witnesses; new test_identities_witness_nothing pins suppression (entry over pure atom substrate → connect ask, no stores). S3 licensing, canon memo, R1 store tests survived untouched. 167 tests pass.

*Relevance: high*
*Context: TRS rationalise redesign, step 2: identity neighbours filtered from candidates*
*Tags: trs rationalise redesign identity-filter self-confirmation*

---
*Observed: 2026-10-16T14:17:44.286Z*
