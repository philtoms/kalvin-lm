---
type: source
title: "Observation: Layer 1 review: Def 5 misfit partition not disjoint; coverage must be primary split"
tags:
  - formalisation
  - algebra
  - review
  - partition
  - coverage
status: observation
created: 2026-09-10
updated: 2026-09-10
slug: obs-2026-09-10-layer-1-review-def-5-misfit-partition-not-disjoint-coverage-
relevance: medium
observed_at: 2026-09-10T12:24:41.474Z
source_context: Reviewing docs/kalvin-symbolic.md Layer 1 (the algebra) for sense, gaps, and vocabulary
---

# 🔍 Observation: Layer 1 review: Def 5 misfit partition not disjoint; coverage must be primary split

Review of docs/kalvin-symbolic.md §1 found the misfit sub-classification (Def 5) is not mutually exclusive as stated: for any uncovered node pair, gap g = v ∧ ¬σ(ν) and excess e = σ(ν) ∧ ¬v are both nonzero (uncovered nodes are disjoint from the head), so Connotation a:[b] and every No-fit satisfy Under+over's stated condition (g≠0, e≠0). Fix: make coverage (n ∧ v ≠ 0) the primary partition criterion — uncovered → S3 shapes (refine by node count: Connotation single / No-fit multi), covered → S2 shapes (refine by gap/excess; Denotation = single-node underfit). Bands then derive from shape instead of being restated. Also found: complement ¬ and 1 used in Def 5 but absent from the declared signature (V, ∨, ∧, 0); §1.1's closure-of-atoms-under-∨ excludes 0 from V yet 0 is the unit; Def 6's "single-kline classification is the special case of C(A,B)" is wrong (C(A,A) = σ(ν):ν is always terminal-or-Canon — fit(v,ν) is a special case of the classifier, not of the relationship construction); "the algebra's only operations are ∨ and kline formation" contradicted by ∧, ¬, σ; single-node under+over ab:[bc] exists unnamed alongside the noted single-node overfit.

*Relevance: medium*
*Context: Reviewing docs/kalvin-symbolic.md Layer 1 (the algebra) for sense, gaps, and vocabulary*
*Tags: formalisation algebra review partition coverage*

---
*Observed: 2026-09-10T12:24:41.474Z*
