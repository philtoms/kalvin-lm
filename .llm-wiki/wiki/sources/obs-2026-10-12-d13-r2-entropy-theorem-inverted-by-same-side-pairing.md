---
type: source
title: "Observation: D13 R2 entropy theorem inverted by same-side pairing"
tags:
  - trs
  - entropy
  - theorem
status: observation
created: 2026-10-12
updated: 2026-10-12
slug: obs-2026-10-12-d13-r2-entropy-theorem-inverted-by-same-side-pairing
relevance: high
observed_at: 2026-10-12T16:05:47.676Z
source_context: Updating Definition 13 entropy-flow bullets in docs/trs-significance.md
---

# ⭐ Observation: D13 R2 entropy theorem inverted by same-side pairing

The Definition 13 entropy-flow theorem "R2 never reduces entropy — Σ μ(head)·H(product) ≥ μ(σ∨â)·H(C)" was withdrawn with the move to same-side pairing. Under same-side pairing can(N_h) ⪯ h for every product, so no product arrives at S3; a node-clean split (no node of N or N_E crosses head boundaries) lands every product at S1 — R2 now discharges surprise rather than partitioning it. Interior entropy survives only where a head's atoms cross node boundaries: H = 1 − μ(can(N_h))/μ(h). Pinned by test_r2_reconstruction_same_side_discharges_surprise and test_r2_reconstruction_well_formed in tests/test_trs_properties.py.

*Relevance: high*
*Context: Updating Definition 13 entropy-flow bullets in docs/trs-significance.md*
*Tags: trs entropy theorem*

---
*Observed: 2026-10-12T16:05:47.676Z*
