---
type: source
title: "Observation: Derivation port reviewed: anchor refinement + wedge dedup fixed in both cores"
tags:
  - derivation
  - engine
  - def17
  - anchor
  - wedge
  - bugfix
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-derivation-port-reviewed-anchor-refinement-wedge-dedup-fixed
relevance: high
observed_at: 2026-09-14T17:43:57.001Z
source_context: Reviewing user's Derivation port into src/dialogue/engine.py
---

# ⭐ Observation: Derivation port reviewed: anchor refinement + wedge dedup fixed in both cores

Reviewed the user's uncommitted engine integration (src/dialogue/derivation.py new; engine.py wired to Derivation with _select goal enumeration; abstract.py/signifier.py KSig annotations). Verdict: structure faithful (both-ends Def 14 guard, ν_B walk, T2 identity keying present; core golden masters pass), but three defects found and fixed: (1) _walk_b composed anchors on content OVERLAP only (signifies) — on real engine memory the ν_B walk from slot `lamb` went [lamb] →fwd lamb:[Object] →rev whatObject:[Object] → [whatObject], arrival fired on the shared `what` word bit, but whatObject never occurs in ν_A → dead bridge; fixed by adding _refine_anchor (contract arrived nodes under held canons whose witness they exactly are, when the canon head occurs in ν_A) plus an occurrence requirement before composing — Def 17's anchor refinement. (2) Run-loop wedge: non-consumable composed bridges were re-derived every iteration (fresh seen-set) and grounded as duplicates — the WDMH→mhall derive on engine memory grounded 29 duplicate whatObject:[lamb] klines and ended "abandoned" at max_steps; fixed with _ground_composed dedup (key = (sig, nodes), once per run) in both dialogue and kalvin cores — now ends "stuck" with 0 composed, which is the honest Def 15 ending for that memory+goal (harness scaffolding supplies the answer via feeds/asks instead). (3) signifier.py `KSig(KNode(sig, label))` double-wrap silently dropped .label (KSig is TypeAlias = KNode; KNode(KNode(...)) loses the label param) — fixed to KSig(sig, label); also moved derive()'s misplaced docstring (was after statements). Ported fixes 1+2 to src/kalvin/derivation.py identically; added regression test test_b_walk_arrival_without_resolution_does_not_ground. All 44 tests pass; both pure-algebra probes green. Noted not-fixed: src/kalvin/derivation.py vs src/dialogue/derivation.py near-duplicate (411/405 lines) — consolidation decision open; engine._select reuses Def 16 occurrence to enumerate candidate GOALS (doc says goal "is declared separately") — vocabulary collision worth a CONTEXT.md note eventually.

*Relevance: high*
*Context: Reviewing user's Derivation port into src/dialogue/engine.py*
*Tags: derivation engine def17 anchor wedge bugfix*

---
*Observed: 2026-09-14T17:43:57.001Z*
