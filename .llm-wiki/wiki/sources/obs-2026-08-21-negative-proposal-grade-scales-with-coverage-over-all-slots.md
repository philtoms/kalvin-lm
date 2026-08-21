---
type: source
title: "Observation: Negative proposal grade scales with coverage over all slots"
tags:
  - dialogue
  - significance
  - coverage
  - negatives
status: observation
created: 2026-08-21
updated: 2026-08-21
slug: obs-2026-08-21-negative-proposal-grade-scales-with-coverage-over-all-slots
relevance: high
observed_at: 2026-08-21T14:28:38.256Z
source_context: Different proposals same significance 127 in DFPAS step
---

# ⭐ Observation: Negative proposal grade scales with coverage over all slots

Negative proposal grades now scale with coverage: ProposalAggregator's negative branch takes mean accountedness over ALL slots (quality x coverage) instead of the mean of understood slots only — the latter was always 1.0 whenever anything was understood, collapsing all negatives to boundary-1 (DFPAS proposals of different guess counts all graded 127). Spread now: DMHAS [did,Mary,have,a,lamb] 4/5 -> S3 102 (confident no); DFPAS [Mary,had,a,little,lamb] 1/5 -> S3 25 (near-noise). Each additional wild guess costs distance from the boundary; positives unchanged (weakest claim decides).

*Relevance: high*
*Context: Different proposals same significance 127 in DFPAS step*
*Tags: dialogue significance coverage negatives*

---
*Observed: 2026-08-21T14:28:38.256Z*
