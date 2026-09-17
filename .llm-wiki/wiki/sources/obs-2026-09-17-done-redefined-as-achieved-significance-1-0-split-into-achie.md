---
type: source
title: "Observation: Done redefined as achieved significance 1.0; γ split into achieved vs journey evaluations"
tags:
  - kalvin-algebra
  - definition-16
  - definition-20
  - significance
  - gamma
  - achieved-journey
  - proposals
status: observation
created: 2026-09-17
updated: 2026-09-17
slug: obs-2026-09-17-done-redefined-as-achieved-significance-1-0-split-into-achie
relevance: critical
observed_at: 2026-09-17T10:33:47.087Z
source_context: Redefining done via calculated significance; achieved vs journey γ split
---

# 🔴 Observation: Done redefined as achieved significance 1.0; γ split into achieved vs journey evaluations

Definition 16 rewritten and Definition 20 extended after the user rejected the SIG_S1 short-circuit ("short-circuits a true understanding of significance"). New structure: (1) Def 16 — a derivation's outcome is the significance achieved at the stopping state, a calculated level, never a boolean; the three stoppings (done/stuck/abandoned) are observations. Done is DEFINED as achieved significance = 1.0; the old band statement fit(C(A,B)) ∈ S1 is demoted to a proof that done means S1 (Canon by case condition, Identity because ν_B=[σ(ν_A)] collapses to equal content). (2) Def 20 gains "Two evaluations": achieved significance = γ at entry depths (Def 19: entry content depth 0 → δ⁰=1 → J(σ(ν_A),σ(ν_B))) — selects the band, travels with the proposal, 1.0 exactly at done; journey significance = γ at recorded depths (J·δ^(D̄+Ĥ)) — effort comparator for derivations achieving the same significance, rate-of-change steers strategy, never selects a band. Exchange paragraph updated: proposal travels at achieved significance, journey stays as acquisition depths. Code: engine.py _propose now stamps gamma_to_byte(result.j1) — j1 is the achieved significance computed in _finish (J of final content vs goal), so done ⇒ j1=1.0 ⇒ 0xFF S1 BY CALCULATION; journey result.gamma untouched. When stalled S2 proposals are later admitted, the same expression yields their band from j1<1.0. Verified: tests 69/69; probe on data/scripts/wdmh.ks (renamed from wdmh-underfit.ks) — T03 proposes WDMH:[a,little,lamb,had,Mary] S1 255, T04 grounds on receipt; monitor shows divergence (stuck hop j1=0.333 achieved vs γ=0.265 journey). Note: repo moved mid-session — user reverted earlier edits, renamed scripts, HEAD now d465283 + a stash exists.

*Relevance: critical*
*Context: Redefining done via calculated significance; achieved vs journey γ split*
*Tags: kalvin-algebra definition-16 definition-20 significance gamma achieved-journey proposals*

---
*Observed: 2026-09-17T10:33:47.087Z*
