---
type: source
title: "Observation: A-1 deleted: unheld residual is S4; creation only at entry/store"
tags:
  - trs
  - significance
  - spec
  - s4
  - boundary
  - axioms
status: observation
created: 2026-09-29
updated: 2026-09-29
slug: obs-2026-09-29-a-1-deleted-unheld-residual-is-s4-creation-only-at-entry-sto
relevance: high
observed_at: 2026-09-29T17:30:02.499Z
source_context: Resolving A-1 vs S4 contradiction in docs/trs-significance.md
---

# ⭐ Observation: A-1 deleted: unheld residual is S4; creation only at entry/store

Third amendment round on docs/trs-significance.md, triggered by user's question "when a derived signature is not held, surely that is a transition to S4? What does A-1 serve?" Resolution: Axiom A-1 DELETED entirely. Reasoning: (a) its never-stuck guarantee is rejected by design — stuck IS S4, the honest knowledge boundary; (b) no-fabrication is structural in the rules (R2 lookups, R3 held witnesses, R1 proven-canonical stores) — now stated as provable property P-4, not an axiom; (c) decodability is I-1: decode outputs are held adjacencies, so "every node in a decoded adjacency is a domain signature" (the user's intended theorem) is a corollary of I-1, which admission and R1 stores maintain by construction. Key reconciliation of the user's two rulings: signature creation happens only where new information ENTERS (entry admission, R1 stores mint new signatures like c_E over held adjacencies) — never inside a rewrite step; R2 now says "a residual not held is a transition to S4 — the misfit asks, later-arriving information may complete the path". Residuals fail independently per product. Decode (Def 6) reverted to partial lookup over Can_D (reinstated). OQ on decode construction dissolved (pure lookup needs no construction rule). OQs now: OQ-1 R3 splice, OQ-2 grading, OQ-3 novel atoms. Model now has exactly two S4 routes: empty adjacency at valuation, and unheld residual at projection — both reading as "knows what it doesn't".

*Relevance: high*
*Context: Resolving A-1 vs S4 contradiction in docs/trs-significance.md*
*Tags: trs significance spec s4 boundary axioms*

---
*Observed: 2026-09-29T17:30:02.499Z*
