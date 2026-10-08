---
type: source
title: "Observation: R2 owns the cross-candidate accumulation; spec and code aligned"
tags:
  - trs
  - spec
  - r2
  - accumulation
  - attribution
status: observation
created: 2026-10-08
updated: 2026-10-08
slug: obs-2026-10-08-r2-owns-the-cross-candidate-accumulation-spec-and-code-align
relevance: medium
observed_at: 2026-10-08T12:33:23.657Z
source_context: Spec attribution fix for R2 accumulation
---

# 🔍 Observation: R2 owns the cross-candidate accumulation; spec and code aligned

ACCUMULATION ATTRIBUTED TO R2 (user: 'Not quite accurate spec-wise — only S2 accumulates across candidates'): spec processing paragraph now opens 'R2 accumulates across candidates: a product re-entry that returns at S1 earns its adjacency, and each later candidate of the entry is formed with the earned adjacencies joined to its neighbour's' (was smeared onto 'processing' generally). Implementation clarity refactor: held→earned, 'R2's accumulation — the running composition', earned |= prod.adj commented 'R2 — the sole accumulator', enrichment line commented as Definition 9 formation. Semantic-neutral — traces identical (goal: Subject:[Subject] ×3, Verb|Subject:[Subject,Verb], Subject|Verb|Object:[Object,Subject,Verb]; ask: Mary:[Mary] ×3 interleaved, had|Mary:[Mary,had], Mary|had:[Mary,had], Det:[Det] ×2, lamb|a|little:[Det], Mary|had|lamb|a|little:[Det,Mary,had]). 159 tests passing. OPEN: whether 'improve the implementation' meant deeper structure (only-S2 candidates see earnings; or intermediate as recursive re-entry inside R2 rather than joined formation) — flagged to user.

*Relevance: medium*
*Context: Spec attribution fix for R2 accumulation*
*Tags: trs spec r2 accumulation attribution*

---
*Observed: 2026-10-08T12:33:23.657Z*
