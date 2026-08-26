---
type: source
title: "Observation: Boundary-relative proposal grading: weakest claim signs, context magnitudes"
tags:
  - dialogue
  - significance
  - proposals
  - aggregator
  - boundary
status: observation
created: 2026-08-21
updated: 2026-08-21
slug: obs-2026-08-21-boundary-relative-proposal-grading-weakest-claim-signs-conte
relevance: high
observed_at: 2026-08-21T13:41:32.874Z
source_context: Significance variation for crossover gap-filling proposals
---

# ⭐ Observation: Boundary-relative proposal grading: weakest claim signs, context magnitudes

Proposal grading now composes against the S2|S3 boundary as the zero point of judgement (ProposalAggregator in kalvin/significance.py, used by pivot_fill.\_grade). A proposal is a conjunction: the weakest claim decides the sign, the rest is context. All slots at least partially accounted -> positive, byte rises from the boundary with the weakest claim. Any unaccounted slot (wild guess, no connotational path) -> negative, byte falls from the boundary with context accountedness: DMHAS:[did,Mary,have,a,lamb] = S3 127 (fully understood ask, guess slot = confident long-worded "no"); DMHAS:[Mary,had,a,little,lamb] = S3 126 (little unresolvable dents context); a guess with no understood context floors at 0. DMHAL:[had,Mary,a,lamb] = S2 252 (all slots resolve through canon chains - genuinely positive). This lays the rationalisation axis: a high-confidence negative (grade near boundary-1 with uniform context) is structurally recognisable as a decorated "no" awaiting compression.

_Relevance: high_
_Context: Significance variation for crossover gap-filling proposals_
_Tags: dialogue significance proposals aggregator boundary_

---

_Observed: 2026-08-21T13:41:32.874Z_
