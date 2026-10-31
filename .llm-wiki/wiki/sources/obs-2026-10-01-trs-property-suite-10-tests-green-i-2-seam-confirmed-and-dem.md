---
type: source
title: "Observation: TRS property suite: 10 tests green; I-2 seam confirmed and demonstrated"
tags:
  - trs
  - property-testing
  - hypothesis
  - invariants
  - i2-seam
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-trs-property-suite-10-tests-green-i-2-seam-confirmed-and-dem
relevance: high
observed_at: 2026-10-01T12:38:02.203Z
source_context: Building property-based invariant tests for src/trs
---

# ⭐ Observation: TRS property suite: 10 tests green; I-2 seam confirmed and demonstrated

Property-based test suite built: tests/test_trs_properties.py (10 hypothesis tests, all passing; full suite 136). Properties: band totality vs independent oracle (S4 iff no adj; S1 iff U=O=⊥; S3 iff F=⊥; else S2); decomposition laws (σ=F∨U, â=F∨O, pairwise disjoint, side containment, S2 equivalence); witness law (can(witness(s))=s for held; ∅ otherwise; total); I-2 on static domains; R2 product well-formedness (heads ⊆ {F,U,O} non-empty, cross-paired adjacency = witness of the other side, witnessed adjacencies held); R3 product well-formedness (legal head, I-1 survives substitution); I-1 maintained by construction across 150 random rationalisations; NO-FABRICATION (domain grows only via S1 stores, every store canonical over a held adjacency, and — at the BIT level — atoms(D_final) ⊆ atoms(D_0 ∪ entry)); bounded and honest termination. Generators: 6-atom universe, I-1-closed domain growth, admissible entries, CONSTRUCTED band-specific klines (S2/S3 pools derived from the domain — random+assume filtering trips hypothesis health checks). FINDINGS: (1) I-2 SEAM CONFIRMED and demonstrated deterministically (test_i2_seam_demonstrated): k(3,{3}) and k(3,{1,2}) are both canonical under head 3; R1 mints the second while the first is held — I-2 is NOT maintained by the rules. Needs a ruling: R1 admission guard (reject/replace competing canonical under a held head) or I-2 restated (witness = first-stored; alternatives held as ordinary klines). Current Domain behaviour: index keeps first, law holds either way. (2) Two of my test bugs were instructive: 'no new atoms' must be measured at the BIT level (composite sigs like 3=1|2 are compositions, not fabrications — the model minted AB from held A,B exactly as 'composes what it holds' predicts); R3 property requires admissible input adjacency. hypothesis>=6.100.0 added to dev extras. Uncommitted.

*Relevance: high*
*Context: Building property-based invariant tests for src/trs*
*Tags: trs property-testing hypothesis invariants i2-seam*

---
*Observed: 2026-10-01T12:38:02.203Z*
