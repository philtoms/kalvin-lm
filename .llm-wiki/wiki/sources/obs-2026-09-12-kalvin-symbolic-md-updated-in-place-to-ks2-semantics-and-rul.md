---
type: source
title: "Observation: kalvin-symbolic.md updated in place to ks2 semantics and rules"
tags:
  - kalvin-symbolic
  - ks2
  - docs
status: observation
created: 2026-09-12
updated: 2026-09-12
slug: obs-2026-09-12-kalvin-symbolic-md-updated-in-place-to-ks2-semantics-and-rul
relevance: high
observed_at: 2026-09-12T07:55:40.548Z
source_context: Bringing kalvin-symbolic.md up to date with ks2 semantics
---

# ⭐ Observation: kalvin-symbolic.md updated in place to ks2 semantics and rules

Updated docs/kalvin-symbolic.md (first-pass formalisation) in place to ks2.md semantics and rules, keeping its narrative style (four-layer reading guide, sorts/term-algebra framing, numbered §1 definitions, metatheory "claims taken, claims renounced", solver reading, KScript appendix, micro-example). Key reconciliations: ask atom dropped from Def 1 (ask is structural S4); memory DAG → unrestricted graph with cycles (new Def 3); central claim restated from "section (right inverse)" to ks2's claims formulation; fit Def 6 rewritten as ks2's ordered 8-case total function with coverage-primary split, bands, ask, invariance; §2.2's expand/contract/retain + remove/add/replace with species-keyed permission table replaced by the single replace rule with modes, two licences, band-keyed licensing table, scoping clause, done/stuck/abandoned; §2.3 weak termination → T1 (Δ₀ bound) and T2 (witnessed cycles, strategy bounds); §3 selection rewritten to signature-in-node (ks2 Def 16) with slot walks/progressive path (Def 17), three bounds, reentry (M_k, B_k); §4 old γ sketch replaced with fixed γ = J·δ^(D̄+Ĥ) and four properties; §5 kept normative for KScript surface syntax with updated `=>`/ask notes; §6 mall example recast — the old "add l, add l" run is now unlicensable without a correspondence, so the example now holds `a:[a,l]` (declared `a => a l`, structurally Overfit) and finishes in one licensed replace, with stuck counterfactuals and γ pricing (Ĥ = 1/3, done grades δ^{1/3}).

*Relevance: high*
*Context: Bringing kalvin-symbolic.md up to date with ks2 semantics*
*Tags: kalvin-symbolic ks2 docs*

---
*Observed: 2026-09-12T07:55:40.548Z*
