---
type: source
title: "Observation: Layers 3–4 review + ks2 §§10–12: selection clauses split, band order axiom, Jaccard core"
tags:
  - formalisation
  - strategy
  - measurement
  - selection
  - graded-distance
  - ks2
status: observation
created: 2026-09-10
updated: 2026-09-10
slug: obs-2026-09-10-layers-3-4-review-ks2-10-12-selection-clauses-split-band-ord
relevance: medium
observed_at: 2026-09-10T14:04:29.799Z
source_context: Reviewing and rewriting kalvin-symbolic.md §§3–4 (strategy, measurement) into ks2.md
---

# 🔍 Observation: Layers 3–4 review + ks2 §§10–12: selection clauses split, band order axiom, Jaccard core

Layers 3–4 review applied to docs/ks2.md as §§10–12 (Def 16 selection; measurement; terminology), scope section renumbered to §13, status line now reads all four layers. Key review findings: (1) first-pass candidate selection conflated three distinct conditions — relationship band ≥ S2 (⇔ content overlap σ(ν_A) ∧ σ(ν_B) ≠ ∅), signature-in-node t ∈ ν_A (CONTEXT.md's selection clause, the grounding-propagation path), and mere signature-node overlap — and the claim "S2 implies grounded signature covered by a node of A" is false: A = abc:[a] vs B = x:[c,a] is Denotation/S2 with x neither in nor overlapping A's nodes; reverse direction B = x:[y] vs A = abc:[x] is in-node yet S3. ks2 Def 16 therefore defines selection as grounded(B) ∧ t ∈ ν_A, with the band routing (S2 → ordinary targeting, S3 → progressive path). (2) "At least S2" presupposed an unstated band order — now an explicit axiom in §11 (S1 > S2 > S3 > S4, within-band shapes unordered), plus the own-band (fit on itself) vs relationship-band distinction. (3) Reentry is composition, not self-application: hop k runs under (M_k, B_k), M grows between hops by STM writes, hop order is the only time axis. (4) Progressive connotation = the S3 replace composite executed incrementally, its licensed interior spaced out by STM writes. (5) Bounds gained natural units: D₀ targeting budget, witnessed-run bound (T2), hop ceiling. (6) Graded distance made parametric: per-slot accountedness α(n) = |n ∧ σ(ν_B)|/|n| with δ hop-discount; band-consistency requires weighing excess — the plain overlap fraction peaks at 1 for underfit (σ_A ⊂ σ_B, still S2), so Jaccard |σ_A ∧ σ_B|/|σ_A ∨ σ_B| is the depth-free core; hop-decay requirement ties to T2. (7) KValue carries the graded assessment; bands need not travel (recomputable from structure). Vocabulary: "working-memory (STM)" dropped per CONTEXT.md avoid-list; "self-application" → composition.

*Relevance: medium*
*Context: Reviewing and rewriting kalvin-symbolic.md §§3–4 (strategy, measurement) into ks2.md*
*Tags: formalisation strategy measurement selection graded-distance ks2*

---
*Observed: 2026-09-10T14:04:29.799Z*
