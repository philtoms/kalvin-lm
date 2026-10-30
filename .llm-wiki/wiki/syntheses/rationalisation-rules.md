---
type: synthesis
title: Rationalisation rules
created: 2026-10-30
updated: 2026-10-30
---

# Rationalisation rules

Rules established while rewriting `Cogitator.rationalise` (src/trs/model.py), tuned against mhall.ks and wdmh.ks. Simple language; the source is the truth.

## The opening

1. The first pairing is the claim: the ask equals the goal. The root entry pairs the goal, and the goal is the only candidate at depth 1.
2. The residuals do the exploring. Every deeper reentry tries candidates, least content first — scaffolding before canons.
3. A pairing splits into F (fit — what entry and candidate share), U (underfit — the entry beyond the candidate) and O (overfit — the candidate beyond the entry).

## Paths

4. A candidate cannot be paired if it is already in the reentry's path. The path follows content flow: an o-side child's region came from its parent's candidate, so that candidate enters its path; a u-side child only inherits the parent's path.
5. A reentry may re-meet its own origin — the candidate it was isolated from — but no other candidate in its path. So `(VO × SVO)` is legal and `(had × MHALL)` is not (MHALL flowed into HALL, and had inherited it).
6. The goal's signature may pair below the root (mhall's SVO is the scaffold itself), but see 4–5 for when.

## Framing

7. All framing is on-path. Exactly two rules frame:
   - **F = E** — the candidate covers the entry, so the fit frames.
   - **U = O, stalled** — two stalled reentries with the same content witness each other (the exchange) and the shared content frames jointly.
   Nothing else frames. The global frame pool is the learning record (stores), never an input to resolution.
8. A frame covers a side only when a **proper ancestor** produced it. A reentry's own pairings never frame its own sides (no self-satisfaction through sibling rows).

## Resolution

9. An entry is resolved when both residuals — U and O — are settled. The fit F is the agreement between entry and candidate; it is recorded, not proven. It never blocks.
10. A side is settled when it is empty, when its reentry resolved, or when a proper ancestor framed it on the way down. A stalled reentry settles only through the exchange.
11. Resolution propagates backwards: an entry resolving settles the parent row's side, and so on up to the root. When the root resolves, the claim grounds.

## The exchange

12. Stalls match on their C. When a reentry stalls, C is a U or an O by definition (C = F ∨ U ∨ O). Two stalls on one signature, from different parents, witness each other: the shared content frames jointly and back propagation resolves both chains. No side polarity is needed.

## The trace table

13. One row per reentry (orthogonality).
14. Every tabled row traces back to the root through tabled pushers. A row whose reentry no tabled row claims is dropped, and the drops cascade.
15. A reentry's slot goes to its earliest-created resolving row — scaffolding first, not the chain that happened to close first.
16. Steps claim every executed (popped) reentry, even ones with no rows (the V convention).
17. The frame column shows the row's own frame plus what it leans on. Each support cites the row that established it, preferring tabled rows.
18. Resolution markers follow the entry and combine with reenter claims — line 1 shows the claim's resolution.
19. Compound labels render in MTS notation (MarySubject → MS, ALLQuery → ALLQ). Presentation only.

## Learning

20. Resolutions admit stores — the run's grounded knowledge (e.g. `DH:[did, have]`, `WDMH:[MHALL]`).
21. A reentry whose whole region is already framed is already known — it resolves without exploring (load-bearing; removing this breaks grounding).

## Known edges

- wdmh's WDH row: the fit `what` frames nowhere — honest under rule 9 (the pairing is the agreement).
- wdmh's HALL row: the `reenter ALL` side closes invisibly — the ⟨ALL⟩ reentry is already known on an ancestor's frame (rule 21 + rule 8).
