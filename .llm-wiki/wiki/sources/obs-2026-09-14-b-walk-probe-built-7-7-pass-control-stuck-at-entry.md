---
type: source
title: "Observation: ν_B-walk probe built: 7/7 PASS, control stuck at entry"
tags:
  - kalvin-algebra
  - def17
  - probe
  - overfit
  - anchor
  - verification
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-b-walk-probe-built-7-7-pass-control-stuck-at-entry
relevance: high
observed_at: 2026-09-14T15:33:24.964Z
source_context: Building the pure-algebra probe for the Def 17 ν_B walk
---

# ⭐ Observation: ν_B-walk probe built: 7/7 PASS, control stuck at entry

Built and verified the ν_B-walk probe: dev/algebra/overfit-anchor-walk.py (uncommitted). Pure-algebra, self-contained like its sibling worked-example-wdmh.py (int values, K records, doc's 9-shape classifier). Scenario: A0 = mh:[m,h] (fragment "Mary had"), B = mhall:[m,h,all] (goal, held Canon) — C(A0,B) = Overfit S2, u=∅, o={a,l}; memory: mhall:[m,h,all], all:[o], all:[a,l,l], o:[m], m:[m]. Trace (7/7 PASS): no A-side slot exists; ν_B walk departs overfit slot `all` → forward(all:[o]) → forward(o:[m]), anchor m ∈ ν_A (no refinement edge; departed end already at goal's witness resolution); composed m:[m,all] (Overfit, adopt-fwd) written at depth 2; main line consumes m⇉[m,all] → [m,all,h], Δ 2→0, done by value equality, C(A,B)=Canon; Ĥ=2/3 (all@2), γ≈0.630, J 0.5→1.0 (docstring initially said 0.4 — that was the WDMH probe's number, fixed). Control run with ν_B walks disabled: stuck at entry — the false ask the one-party def would give (memory connects the parties via o:[m]). Implementation notes: find_targeting's S2 guard was generalized to Def 14's both-locations reading — forward licensed when signature touches u OR witness content touches o (the old A-centric gap-guard blocked all forward moves in pure overfit); reverse mirrored. walk(seed, end_mask) unified for both parties. Exit code 0; sibling probe still 7/7. Follow-up candidate: the generalized S2 guard arguably belongs back in the doc's Def 14 wording ("restricted to the misfit region" — arrival-side as well as departure-side).

*Relevance: high*
*Context: Building the pure-algebra probe for the Def 17 ν_B walk*
*Tags: kalvin-algebra def17 probe overfit anchor verification*

---
*Observed: 2026-09-14T15:33:24.964Z*
