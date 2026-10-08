---
type: source
title: "Observation: Spec R2 now states reconstruction: F/O from candidate's witness, U from entry's canon"
tags:
  - trs
  - spec
  - r2
  - reconstruction
status: observation
created: 2026-10-08
updated: 2026-10-08
slug: obs-2026-10-08-spec-r2-now-states-reconstruction-f-o-from-candidate-s-witne
relevance: high
observed_at: 2026-10-08T09:58:20.646Z
source_context: Spec R2 updated to the reconstruction rule
---

# ⭐ Observation: Spec R2 now states reconstruction: F/O from candidate's witness, U from entry's canon

SPEC R2 UPDATED TO RECONSTRUCTION (ruling: update now — the bridge is orthogonal). R2 now reads: compute F,U,O (Def 12); 'The heads are special: F and O are pieces of can(N) — a held witness — and U is a piece of σ = can(N_E), the entry's canon. Decomposing a witness retains witness status across its components, so each head's adjacency is reconstructed from the witness whose canon covers it — no lookup' with the three reconstruction lines (N_F = {n∈N: n⪯F} candidate F-side; N_O = {n∈N: n⪯O} candidate O-side; N_U = {n∈N_E: n⪯U} entry U-side); products ⟨F,N_F⟩⟨U,N_O⟩⟨O,N_U⟩ for non-empty heads; cross-wise sentence now 'claim in one product and as adjacency in the other' (user had already dropped 'lookup key'); bands paragraph: 'fixed by the reconstruction. A node-clean split gives can(N_F) = F: the F product arrives at S1 (Definition 5). The cross products' adjacencies canonicalise to the other head — disjoint by construction (Definition 12): each arrives at S3...'; ≥2 S1 rule unchanged; witness_D lookups GONE from R2. Def 6 (witness) untouched — still the domain's lookup definition, just no longer R2's mechanism. Tree diagram's ⟨F,N_F⟩(S1)/⟨U,N_O⟩(S3)/⟨O,N_U⟩(S3) notation now reads correctly under reconstruction. 159 tests passing. NEXT FORK: word-bits/abstract-bits bridge (ask canon is word-bits; SVO is abstract-bits; connotation lives in kline structure not node values).

*Relevance: high*
*Context: Spec R2 updated to the reconstruction rule*
*Tags: trs spec r2 reconstruction*

---
*Observed: 2026-10-08T09:58:20.646Z*
