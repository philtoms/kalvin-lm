---
type: source
title: "Observation: Amendment 16 repaired; I-2 dissolution direction: drop uniqueness, witnesses coexist"
tags:
  - trs
  - repair
  - amendment-16
  - i2
  - direction
  - witness-structure
  - continuous-operation
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-amendment-16-repaired-i-2-dissolution-direction-drop-uniquen
relevance: high
observed_at: 2026-10-01T15:42:08.328Z
source_context: Repairing non-atomic amendment-16 edit; I-2 direction steer
---

# ⭐ Observation: Amendment 16 repaired; I-2 dissolution direction: drop uniqueness, witnesses coexist

REPAIR: the amendment-16 edit round failed NON-ATOMICALLY — the tool reported failure of the batch, but the follow-up changelog-only edit landed while the §5 D-liveness paragraph, the D14 row, and the count change did not; commit ac54929 therefore claimed an amendment its body didn't carry (user caught it: 'only the changelog note landed'). Repaired in aa8c207: paragraph, D14, count all verified present; 154 tests green. LESSON: after any tool-reported edit failure, grep the file for the intended insertions — 'failed' batches can leave partial state, and follow-up edits make it look complete.

I-2 STEER (user, on hold but direction now identified): 'Not convinced this invariant is necessary now that we have correctly realised reentry. When a signature C has two canons: C:[AB] and C:[A,B] this offers the opportunity to explore nuanced witness structure.' Direction for the continuous-operation dissolution: DROP uniqueness rather than arbitrate — multiple canonical klines per head coexist as multiple witnesses (compound C:[AB] vs decomposed C:[A,B] — the nuance is decomposition structure); witness becomes a CHOICE POINT at use (R2's witness(F)/witness(O)/witness(U)) rather than a function D → N: either branching (R2 widens over witness choices — rationalisation becomes search over witness structure) or selection at use (grade/strategy). The parked granularity admission card becomes unnecessary — witnesses accumulate instead of competing for admission. NOTE: with the R1 rewrite, canonical minting happens only at first-round S1 — collisions are rare and presentation-shaped; D-liveness (D14) means later runs see earlier runs' canons, so continuous operation accumulates witness structure naturally.

*Relevance: high*
*Context: Repairing non-atomic amendment-16 edit; I-2 direction steer*
*Tags: trs repair amendment-16 i2 direction witness-structure continuous-operation*

---
*Observed: 2026-10-01T15:42:08.328Z*
