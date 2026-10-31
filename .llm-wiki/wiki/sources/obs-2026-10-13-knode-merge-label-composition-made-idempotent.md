---
type: source
title: "Observation: KNode.merge label composition made idempotent"
tags:
  - knode
  - merge
  - labels
  - bugfix
status: observation
created: 2026-10-13
updated: 2026-10-13
slug: obs-2026-10-13-knode-merge-label-composition-made-idempotent
relevance: medium
observed_at: 2026-10-13T09:55:06.666Z
source_context: Fixing knode.merge label duplication
---

# 🔍 Observation: KNode.merge label composition made idempotent

KNode.merge in src/kalvin/kline.py composed labels by plain "|".join of the two sides, so A.merge(A) produced label "A|A" — non-idempotent. Since R2 accumulation (src/trs/model.py rationalise) repeatedly merges earned compositions, duplicates compounded. Fixed: label composition now splits both sides on "|", dedupes preserving order, and joins — a union of provenance parts. Regression tests in tests/test_knode_merge.py. Note: KNode equality (int subclass) ignores labels, so the bug was cosmetic/debug-output only; two pre-existing test_trs_model failures are unrelated (fail identically with the fix stashed).

*Relevance: medium*
*Context: Fixing knode.merge label duplication*
*Tags: knode merge labels bugfix*

---
*Observed: 2026-10-13T09:55:06.666Z*
