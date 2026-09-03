---
type: source
title: "Observation: kalvin-symbolic.md converted from denotational to operational framing"
tags:
  - formalisation
  - algebra
  - solver
  - docs
status: observation
created: 2026-09-02
updated: 2026-09-02
slug: obs-2026-09-02-kalvin-symbolic-md-converted-from-denotational-to-operationa
relevance: high
observed_at: 2026-09-02T13:42:27.980Z
source_context: Reconciling the denotational body of kalvin-symbolic.md with the operational solver reading
---

# ⭐ Observation: kalvin-symbolic.md converted from denotational to operational framing

Reviewed docs/kalvin-symbolic.md §§1–7 against the A.2 solver reading. Verdict: document was denotational (terms as static facts, significance as fixed valuation), now re-anchored as operational. §2 (carrier) and §3 (symbol algebra) survive untouched — temporality does not affect them. Changes made: §1 now states operations are denotational in what they specify, operational in how they evaluate. §4 table reinterpreted: "Produced shape" = the constraint denoted (not guaranteed result); Band column = Target Significance of the open constraint — this explains (not contradicts) why `=>` sits at S2 while a solved Canon claims S1. §4.1 significance re-typed as solver search state; the three significances are one process at three moments (constraint / syntactic status / search position); S2 ceiling and state-preservation become safety properties, eventual generation becomes liveness. §5 axiom 6 extended with safety/liveness/state-preservation. §6 inference row changed from graph traversal to constraint solving over the realised subalgebra. Gaps 3, 4, 6 rewritten: gap 6 downgraded from contradiction to missing open/solved distinction in `band_significance`; gap 3 now includes the missing halting condition and the realised-domain theorem.

*Relevance: high*
*Context: Reconciling the denotational body of kalvin-symbolic.md with the operational solver reading*
*Tags: formalisation algebra solver docs*

---
*Observed: 2026-09-02T13:42:27.980Z*
