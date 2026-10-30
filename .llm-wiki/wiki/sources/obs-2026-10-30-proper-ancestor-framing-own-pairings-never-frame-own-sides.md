---
type: source
title: "Observation: Proper-ancestor framing: own pairings never frame own sides"
tags:
  - trs
  - framing
  - proper-ancestor
  - self-framing
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-proper-ancestor-framing-own-pairings-never-frame-own-sides
relevance: high
observed_at: 2026-10-30T14:11:33.625Z
source_context: Fixing the HALL ← MHALL unexpected framing
---

# ⭐ Observation: Proper-ancestor framing: own pairings never frame own sides

The \"HALL ← MHALL unexpected framing\" flag led to the proper-ancestor rule: a frame may cover a row's side only if produced by a PROPER ancestor reentry — a reentry's own alternative pairings never frame its sides. The leak: (HALL × MHALL-canon) — a sibling row of the tabled (HALL × had:DH) row, same ⟨HALL⟩ reentry — produced the HALL frame whose frame_re=⟨HALL⟩ sat in row 6's OWN chain, covering row 6's U=ALL (self-satisfaction through a side door). Fix: _onpath/_cover_frame take own=id(re) and exclude it; citations prefer tabled > on-chain > any producer. Probed alternatives: dropping the _run already-known skip entirely → both claims die (skip is load-bearing); goal-sig-only-at-opener (block the claim's sig below the root) → wdmh grounds beautifully (canon never pairs) but mhall dies on BOTH sides — mhall's goal SVO IS the grammar scaffold the whole script hangs off, so goal-sig pairings below root must stay legal ((S × SVO), (VO × SVO) in the user's own expected table). Remaining invisible support: wdmh row 6's U=ALL closes via the ⟨ALL⟩ child skipping as already-known on the HALL frame (legal: F==E framing from an ancestor reentry, proper for the child) — step claims \"reenter ALL\" with no tabled ALL row. State: both claims ground, no unexpected framings displayed, 134 tests, deterministic.

*Relevance: high*
*Context: Fixing the HALL ← MHALL unexpected framing*
*Tags: trs framing proper-ancestor self-framing*

---
*Observed: 2026-10-30T14:11:33.625Z*
