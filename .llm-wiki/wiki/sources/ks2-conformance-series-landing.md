---
type: source
title: "ks2 conformance series: 6 commits to the engine entry path"
status: insight
category: architecture
created: 2026-09-12
updated: 2026-09-12
slug: ks2-conformance-series-landing
---

# ks2 conformance series: 6 commits to the engine entry path

Follow-on to the [[sources/obs-2026-09-12-ks2-md-vs-harness-path-code-8-deviations-found-spec-faithful|ks2 conformance review]]. Five commits brought the engine's entry path (harness → engine → Cogitator → kline/significance/signifier) to ks2's algebraic standards on the mechanical deviations, each verified against the three scripts (wdmh-underfit, mhall, wdmh-underfit-o) and the 35 tests, committed per task:

1. `f273c73` — engine drops ask-atom branching; routing is `is_misfit` alone; Cogitator query = `find_canon(sig) or entry` (structural, no mark read); refusal guard added honouring `EngineState.refuse`'s "not to be re-proposed" contract.
2. `0f69f50` — Cogitator grades with γ ([[concepts/mts-multi-token-signature|MTS]]-independent): `expand` returns per-slot depth records (fit free, bridged at hops, crossover at both), `cogitate` composes `gamma_aggregate`/`gamma_to_byte` with J over the word atoms of the query-candidate pair. Raw hop counts as bytes is fixed — exact fills can grade S1, deep walks sink to the S3 floor.
3. `9babbeb` — coverage is overlap (`signifies`), not containment (`node_in`): single-node overfit `a:[ab]` and under+over `ab:[bc]` band S2 per Def 10 cases 7-8.
4. `bb0dcc5` — selection by occurrence (Def 16): `_selectable` yields held non-terminal klines whose signature occurs as a node of the entry; reduce's overlap scan stays as the reverse-occurrence search index (exact licence = witness membership, checked at use). All overlap-scanned S3 proposals vanished — they were the junk the supervisor declined; canonless misfits now emit structural asks instead. This is the honest state: per the two-overlap caveat, a B-side that neither occurs in A's nodes nor is a declared goal is not selectable.
5. `1898bb1` — canon evidence well-foundedness: `is_canon_evidence` (exact ∧ signature ∉ witness) gates `find_canon`/`canon_nodes`; `is_canon` stays the band predicate (self-containing canons are exact S1 shapes, inert as evidence).
6. `4c821c6` — `is_exact` tests atom-space (word-bit) equality via residual both ways, restoring Def 9's iff; BPE packing bits no longer participate in classification.

Remaining deviations are forks, not fixes: the declared-goal parameter and derivation states (item 5 — a restructure: KScript `=>` must reach cogitate as B, with T1/T2 budgets); compiler target-significance stamps vs §13's "once solved" column (item 8 — answer-key semantics, user decision); refusal-vs-endings and tier mechanics (protocol/CONTEXT.md territory, like the ask atom's packing clause).

*Category: architecture*

---
*Captured: 2026-09-12*

## Related

_Add links to related pages._
