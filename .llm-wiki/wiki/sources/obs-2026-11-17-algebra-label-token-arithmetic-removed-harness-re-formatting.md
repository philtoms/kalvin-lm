---
type: source
title: "Observation: Algebra label-token arithmetic removed; harness re-formatting retired"
tags:
  - algebra
  - labels
  - mts
  - harness
  - rationalisation
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-algebra-label-token-arithmetic-removed-harness-re-formatting
relevance: high
observed_at: 2026-11-17T16:14:21.171Z
source_context: Removing the harness's label re-formatting responsibility
---

# ⭐ Observation: Algebra label-token arithmetic removed; harness re-formatting retired

src/rationalisation/algebra.py: removed _tokens and the word-token set arithmetic from underfit/overfit/fit. It parsed bar-composed labels ("A|B") to name residuals/meets from label parts; with opaque MTS labels it cannot split, and keeping it would mint WRONG residual names (e.g. underfit(MHALL, ALL) would label the MH-valued residual "MHALL"). Derived values now carry honest provenance labels ("MSO∖SV", "MSO∧SV", "⊥" when zero). The harness's _tlabel already routes ∖/∧ labels to the _words word-bit decomposition, which is bit-accurate by construction — so the trace table renders byte-identically (verified on mhall.ks and wdmh.ks). The bar branch and ⊥ guard were removed from _tlabel; it now imports mts from kalvin.kline for its remaining presentation needs (_humps of authored camelCase identifiers, _words decompositions).

*Relevance: high*
*Context: Removing the harness's label re-formatting responsibility*
*Tags: algebra labels mts harness rationalisation*

---
*Observed: 2026-11-17T16:14:21.171Z*
