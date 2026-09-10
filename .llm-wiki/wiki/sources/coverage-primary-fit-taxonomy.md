---
type: source
title: Coverage-primary split for fit classifiers over set-algebra values
status: insight
category: design
created: 2026-09-10
updated: 2026-09-10
slug: coverage-primary-fit-taxonomy
---

# Coverage-primary split for fit classifiers over set-algebra values

When classifying (head, nodes) fit against a set-algebra of values (union/intersection/complement), gap/excess conditions alone cannot yield a disjoint partition: an uncovered node is disjoint from the head, so gap and excess are *both* automatically nonzero — every no-coverage case satisfies the "under+over" condition. Kalvin's kalvin-symbolic.md Def 5 had exactly this bug (Connotation `a:[b]` and No-fit satisfied Under+over's g≠0, e≠0).

The fix generalises: make **coverage** (n ∧ head ≠ ∅ — overlap, not containment) the primary partition criterion, and use gap/excess only to subdivide *within* covered cases; refine terminal/no-coverage cases by node count. Bands (S1 exact / S2 partial / S3 unrelated / S4 no-claim) then derive from shape instead of being asserted separately. Corollary: excluding zero heads and zero nodes makes the case split clean, since uncovered + exact is impossible with nonzero nodes. See [[sources/obs-2026-09-10-layer-1-review-def-5-misfit-partition-not-disjoint-coverage-]] and the claim/witness vocabulary in [[sources/obs-2026-09-10-layer-1-vocabulary-proposal-values-as-sets-claim-witness-reg]].

*Category: design*

---
*Captured: 2026-09-10*

## Related

_Add links to related pages._
