---
type: source
title: "Observation: FIFO cogitation + survive-on-no-proposal + unknown-not-groundable"
tags:
  - dialogue
  - engine
  - cogitation
  - stm
status: observation
created: 2026-08-17
updated: 2026-08-17
slug: obs-2026-08-17-fifo-cogitation-survive-on-no-proposal-unknown-not-groundabl
relevance: high
observed_at: 2026-08-17T16:05:48.057Z
source_context: Engine tuning after FIFO cogitation change
---

# ⭐ Observation: FIFO cogitation + survive-on-no-proposal + unknown-not-groundable

Three engine fixes on mhall: (1) cogitate scan reversed to oldest-first (FIFO) — temporality: the entry waiting longest cogitates first; LIFO was an accident that made MHALL:[SVO] countersign after SVO:[MHALL]. (2) The misfit (S2) arm now pops its STM entry only when the strategy produced a batch — no-proposal misfits survive, which is what lets relationships like MHALL:[SVO] persist until the operand canon arrives. (3) _is_groundable returns False for unknowns ({S:[]}) — the FIFO change unmasked all([])==True grounding empty asks as garbage. Result: mhall runs clean — 21 groundings including all script denotations (Mary:[Subject], had:[Verb], a:[Det], MHALL:[SVO], SVO:[MHALL], WDMH:[Mary,DH]), STM empty, S3=0 (all relationships ground via the universal rule; countersign arm now dormant). New active-state question: author a curriculum that forces compositional pairing.

*Relevance: high*
*Context: Engine tuning after FIFO cogitation change*
*Tags: dialogue engine cogitation stm*

---
*Observed: 2026-08-17T16:05:48.057Z*
