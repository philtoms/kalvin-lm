---
type: source
title: "Observation: D2+D3 implemented: canonical γ in engine; expand.py repaired from dead state"
tags:
  - engine
  - gamma
  - reconciliation
  - significance
  - expand
  - d2
  - d3
status: observation
created: 2026-09-11
updated: 2026-09-11
slug: obs-2026-09-11-d2-d3-implemented-canonical-in-engine-expand-py-repaired-fro
relevance: critical
observed_at: 2026-09-11T18:17:35.180Z
source_context: Implementing D2+D3 engine reconciliation after the ks2 fourth pass
---

# 🔴 Observation: D2+D3 implemented: canonical γ in engine; expand.py repaired from dead state

Implemented D2+D3 engine reconciliation (γ conformance) in src/kalvin/significance.py + src/kalvin/expand.py, tests in tests/test_gamma.py (21 passing; only test module in suite). Discovery: kalvin.expand was ALREADY broken — it imported is_s1 from kalvin.significance which no longer exists (removed at some point, import never fixed) — the γ path was dead code; this change repaired it. significance.py additions: WORD_BITS=0xFFFF_FFFF_0000_0000 (atom space; mirrors signifier._TYPE_MASK, BPE half never weighs), word_atom_count (masked popcount), DEFAULT_DELTA=0.5 (the knob; each hop halves — replaces asymptotic k=50 which barely discounted), geometric_decay(h)=δ^h, SlotRecord=(atom weight, hop depth|None), gamma_aggregate(slots, a_sig, b_sig, delta)=J·δ^mean-depth where J = accounted atoms / UNION atoms (graph-mediated accountedness: slot with defined depth counts; union denominator fixes band-consistency — full A-inside-B coverage now <1) and depth = atom-weighted mean hops, unaccounted excluded (their cost is J's). gamma_to_byte maps γ through the same saturation guards as compose_terminal. expand.py rewritten: slot records instead of decayed floats (matched&grounded→(|n|,0), matched-ungrounded→(|n|,1), resolvable→(|n|,h), unresolvable→(|n|,None)); grounded check now model.grounded (ratified tiers: Frame/LTM/Base, STM excluded); aggregator param replaced by delta; side-candidate bytes via gamma_to_byte(geometric_decay(h)); vacuous node-less pair falls out of union==0→1.0. Scope note: dialogue layer's ProposalAggregator (weakest-claim sign) is a different grader, untouched — D7 territory. Symmetry note: engine slots span BOTH sides' mismatched nodes (symmetric γ), unlike ks2's directional γ — kept engine semantics, only the aggregation form canonicalised. D1 (ASK_BPE_TOKEN) skipped per user; D8 closed (experimental code, no derivation executor; pivot_fill stays). Remaining reconciliation: D4 (Ĥ carried across turns), D5 (is_connotation split), D6 (selection by occurrence), D7 (pivot_fill ↔ Def 17 vocabulary + consume scoping check).

*Relevance: critical*
*Context: Implementing D2+D3 engine reconciliation after the ks2 fourth pass*
*Tags: engine gamma reconciliation significance expand d2 d3*

---
*Observed: 2026-09-11T18:17:35.180Z*
