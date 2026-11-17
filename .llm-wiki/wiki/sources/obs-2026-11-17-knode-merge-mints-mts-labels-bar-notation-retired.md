---
type: source
title: "Observation: KNode.merge mints MTS labels; bar notation retired"
tags:
  - knode
  - labels
  - mts
  - notation
  - merge
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-knode-merge-mints-mts-labels-bar-notation-retired
relevance: high
observed_at: 2026-11-17T16:14:15.851Z
source_context: Replacing bar notation in KNode.merge with MTS notation
---

# ⭐ Observation: KNode.merge mints MTS labels; bar notation retired

KNode.merge (src/kalvin/kline.py) now composes labels through the new kalvin.kline.mts() helper instead of "|"-joining parts: a single part stands whole, a compound of 2+ takes each part whole when already an abbreviation (all-upper) else its uppercased initial (Mary+Subject→MS, had+Verb→HV, SVO+Mary→SVOM). Equal parts compose to themselves (A.merge(A)→A). The composition folds associatively (can/signature_of) because accumulated MTS labels are all-upper and thus stand whole as the next part. Follows the 85d8d1b5 convention: authored identifiers stay in KDbg and the _compound_sigs registry keys; only value labels switched notation.

*Relevance: high*
*Context: Replacing bar notation in KNode.merge with MTS notation*
*Tags: knode labels mts notation merge*

---
*Observed: 2026-11-17T16:14:15.851Z*
