---
type: source
title: "Observation: decode renamed witness: strict converse of canonicalisation"
tags:
  - trs
  - significance
  - terminology
  - witness
  - decode
  - rename
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-decode-renamed-witness-strict-converse-of-canonicalisation
relevance: high
observed_at: 2026-10-01T10:58:49.166Z
source_context: Renaming decode to witness across docs/trs-significance.md
---

# ⭐ Observation: decode renamed witness: strict converse of canonicalisation

Naming decision on docs/trs-significance.md: decode renamed WITNESS per user ruling. witness_D : Node → P_fin(Node), total — the held witness of a signature (∅ when not held); law can(witness_D(s)) = s for held s; Definition 6 retitled 'Witness' and states 'Witness is the converse of canonicalisation under strict conditions — the held witness of a signature, as the engine's held witness (kalvin-algebra.md Definition 6)'. Rationale recorded: decode collided with the engine's token decode; 'decompose' rejected because kalvin-algebra explicitly denies decomposition as a capability ('the only decomposition it ever sees is a held witness') — so the operation IS the held-witness lookup and takes its name; strictness is inherent to the term (no testimony → no witness → ∅ → S4/R4 'witnessless boundary'); 'resolve' rejected for loose-use collision, 'ground' for §14 collision, denotation/connotation are fit shapes. Terminology now: canonicalise/witness as the dual pair. Swept: Def 6, §0.1 notation table, D4, D8, I-1 corollary ('witnessed adjacency'), Def 7 lookup table ('serves witness'), R2 ('compute F,U,O and witness'), amendment note round 8. Side benefit: pre-existing uses — 'the other side's witness' (intro), 'replaces a reference with the witness it names' (R3), 'witnessed expand' (§6 mapping) — now all align with the formal term. Remaining 'decode' only inside the changelog (historical).

*Relevance: high*
*Context: Renaming decode to witness across docs/trs-significance.md*
*Tags: trs significance terminology witness decode rename*

---
*Observed: 2026-10-01T10:58:49.166Z*
