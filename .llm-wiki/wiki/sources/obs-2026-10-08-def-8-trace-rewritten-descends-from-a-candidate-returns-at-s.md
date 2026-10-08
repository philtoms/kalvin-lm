---
type: source
title: "Observation: Def 8 trace rewritten: descends from a candidate, returns at S1 or S4"
tags:
  - trs
  - spec
  - definition8
  - trace
status: observation
created: 2026-10-08
updated: 2026-10-08
slug: obs-2026-10-08-def-8-trace-rewritten-descends-from-a-candidate-returns-at-s
relevance: medium
observed_at: 2026-10-08T08:56:50.538Z
source_context: Moving the opener detail into the trace definition
---

# 🔍 Observation: Def 8 trace rewritten: descends from a candidate, returns at S1 or S4

TRACE DEFINITION REWRITTEN (user ruling: the candidate-opens-its-own-trace detail belongs in the trace definition; 'a chain of entries E₀,E₁,E₂,…' was misleading — implied one linear trace per entry, hiding the per-candidate fork). Def 8 now: 'A trace descends from a candidate: C₀, formed on a presented entry E₀, opens the trace; the entries after E₀ are the re-entry products of the candidates' rationalisations (R2, R3), each forming the candidates that continue the trace, and the trace returns at S1 or S4. Candidates formed on one entry open distinct traces.' R1 prose slimmed to reference it: 'The kline that enters D is the trace's opening candidate C₀ (Definition 8) — not the candidate formed on the final entry…' — the R1 law block unchanged. Code already matched (rationalise(C, C, depth); S3 continues the trace, S2 products re-enter as entries whose candidates open their own traces). 159 tests passing.

*Relevance: medium*
*Context: Moving the opener detail into the trace definition*
*Tags: trs spec definition8 trace*

---
*Observed: 2026-10-08T08:56:50.538Z*
