---
type: source
title: "Observation: Harness feeds asks at γ(A,B), not structural bands; Def 22 self-exclusion"
tags:
  - harness
  - significance
  - gamma
  - ask
  - countersigns
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-harness-feeds-asks-at-a-b-not-structural-bands-def-22-self-e
relevance: high
observed_at: 2026-09-16T14:04:05.827Z
source_context: Harness γ-significance correction after review
---

# ⭐ Observation: Harness feeds asks at γ(A,B), not structural bands; Def 22 self-exclusion

Implemented the user's correction to Break 1: the fix is NOT engine routing — the harness must not set a query's significance to the structural value. `WDMH:[]` is an S4 structure (the ask's shape, recomputable from structure by the engine), but the exchanged byte is the trainer's subjective significance γ(WDMH, MHALL) per Def 20. Changes: ks/ast_emitter SymbolicEntry.goal + _process_scope_body extracts the `==` RHS sig id (resolved through the same binding path) onto the ask; ks/token_encoder carries it to KDbg; kalvin/kline KDbg.goal (note: KLine.__init__ copies dbg field-by-field when a resolver is active — a new KDbg field MUST be added to that copy list or it is silently dropped); dialogue/harness run() pairs ask↔goal among compiled entries and graded() stamps the fed ask with gamma_to_byte(J(ask_sig, goal_sig)) — fresh content carries zero depths (Defs 18–19) so γ = J. Also landed the undisputed Def 22 fix: kalvin/hop.candidate_goals excludes the queued kline itself (γ(A,A)=1.0 was monopolizing the top of every goal list; Hop breaks on first done → every hop ended vacuously → zero asks ever). 51 tests pass. wdmh run: ask feeds S3 (γ=2/6), the MTS canon WDMH:[what,did,Mary,have] also grades S3 and slow-routes into cogitation as the algebra's A₀; first live engine ask since the switch (proposes MHALL:[MHALL] S2, escalated, declined). mhall consequence: γ(MHALL, SVO)=0 (disjoint contents) → ask AND sentence canon feed S4 → refused — fork to resolve.

*Relevance: high*
*Context: Harness γ-significance correction after review*
*Tags: harness significance gamma ask countersigns*

---
*Observed: 2026-09-16T14:04:05.827Z*
