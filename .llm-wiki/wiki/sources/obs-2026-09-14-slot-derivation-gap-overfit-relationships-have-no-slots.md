---
type: source
title: "Observation: Slot derivation gap: overfit relationships have no slots"
tags:
  - kalvin-algebra
  - slot-derivation
  - overfit
  - design
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-slot-derivation-gap-overfit-relationships-have-no-slots
relevance: high
observed_at: 2026-09-14T14:40:29.452Z
source_context: Design discussion of slot derivation overfit gap in kalvin-algebra.md
---

# ⭐ Observation: Slot derivation gap: overfit relationships have no slots

Identified design gap in docs/kalvin-algebra.md Def 17 (Slot derivation): the slot criterion ("a node of ν_A is a slot when it contains an atom from the current underfit") is vacuous for Overfit relationships (u = ∅), so no slot walk is seeded and the derivation strands in Stuck case 2 (ask) — even when memory contains a latent route. Note: application of overfit evidence already works (selection + adopt-fwd, e.g. held c:[c,d] finishes C(A,B)=abc:[a,b,c,d] in one step); only construction is missing. Key asymmetry: underfit is positively located in ν_A (diagnostic slots), overfit is positively located in ν_B — "absence has no location on A", so option 1 (overfit slots on ν_A) collapses into option 2 (reverse derivation from ν_B). Option 2 is half-sanctioned by existing clauses: band-general mirror clause (Def 13), "klines are direction-free, arrival orients them" (§9), and C(B,A)=abcd:[a,b,c] is an Underfit relationship so Def 17 nearly applies with roles swapped — except the goal is never rewritten; the walk writes a bridge (e.g. c:[c,d]) oriented head-ward to A by arrival, consumed by existing selection/adopt-fwd. Design fork recorded: 2a = reverse walks only (done unchanged, "goal is not selected" weakens to a rewrite ban, T1 untouched) vs 2b = full mutual derivation (done becomes shared correspondence; endings, T1, goal-scoping all re-derived). Open wrinkle: Def 17's end condition doesn't mirror cleanly (mirror of "arrival in the goal's overfit" is vacuous in pure Overfit; reverse walk must end at A-held/shared content, and the overfit-sharpening does semantic work beyond Δ-reduction — excludes junk bridges like w:[h] which also shrink Δ). No edits made yet.

*Relevance: high*
*Context: Design discussion of slot derivation overfit gap in kalvin-algebra.md*
*Tags: kalvin-algebra slot-derivation overfit design*

---
*Observed: 2026-09-14T14:40:29.452Z*
