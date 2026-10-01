---
type: source
title: "Observation: Amendment 15: proposals ordered by graded significance (ascending effort)"
tags:
  - trs
  - proposals
  - ordering
  - graded-significance
  - amendment-15
  - mid-run-rescue
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-amendment-15-proposals-ordered-by-graded-significance-ascend
relevance: high
observed_at: 2026-10-01T13:55:07.350Z
source_context: "Ruling: proposal ordering is graded significance, not strategy"
---

# ⭐ Observation: Amendment 15: proposals ordered by graded significance (ascending effort)

Amendment 15 (user ruling): PROPOSAL ORDERING removed from strategy — it IS graded significance. Proposals ordered ascending by effort (Definition 14), most significant first per the identity 'more significant == less entropy': the least-entropy proposal is the most significant next step — finish the nearly-grounded before acquiring new; ties keep generation order. Spec: Definition 15 tail, D13, §7 strategy bullet reverted (neighbours/substitutions/products only), amendment 15 logged. Code: proposals() returns sorted(out, key=effort) — Python sort is stable so ties keep end order. Tests: deterministic fixture (base + terminal k(AB,∅) + k(ABC,{AB}) + terminal k(CD,∅), entry ⟨C,{C}⟩) yields exactly two ends — emptied ⟨C,{AB}⟩ at E=2/3 and no-adjacency ⟨C,∅⟩ at E=1 — asserted order [2/3, 1]; property asserts ps == sorted(ps, key=effort) over random runs. 153 passing. FINDING en route (behavioural, not a bug): the first ordering fixture self-healed — in round 2, the store licensed moments earlier (⟨AB,{C}⟩, from the R1 rewrite) entered covered(3) and made the substitution {AB} → {C} instead of ∅, RESCUING the would-be emptied expansion mid-run: a rationalisation's own fresh knowledge repairs a halt before it happens (substitution is D-relative, and D grows during the run). Worth remembering as a d44 candidate for the dialogue harness.

*Relevance: high*
*Context: Ruling: proposal ordering is graded significance, not strategy*
*Tags: trs proposals ordering graded-significance amendment-15 mid-run-rescue*

---
*Observed: 2026-10-01T13:55:07.350Z*
