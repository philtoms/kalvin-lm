---
type: source
title: "Observation: Hop.run hang fixed: witness-driven survey + enforced walk state bound"
tags:
  - derivation
  - bugfix
  - performance
  - bounds
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-hop-run-hang-fixed-witness-driven-survey-enforced-walk-state
relevance: critical
observed_at: 2026-09-16T17:09:57.826Z
source_context: Fixing the infinite loop in Hop.run
---

# 🔴 Observation: Hop.run hang fixed: witness-driven survey + enforced walk state bound

Diagnosed the user-reported infinite loop in Hop.run (wdmh-underfit.ks): two unenforced/exponential paths in the derivation layer. (1) **canonicalisations enumerated all node subsets** — combinations(range(n), size) for every size 2..n-1, ~2^n iterations EVEN WITH NO MATCHING CANON. Node lists grow via chained forward expansions (each targeting replace inserts a canon's nodes; MAX_STEPS=32 × ~5 nodes → 150+); past n≈35 the survey alone is ~10^29 iterations — an effective hang inside one _canonicalise step inside Derivation.run inside Hop.run. Which derivation balloons is memory-dependent (my states happened to hit done/stuck first — the user's state didn't). Fix: witness-driven survey — for each held canon (2 ≤ len(nodes) < len(nodes-current)), enumerate multiset placements (per-value combinations + product) — the same relation (a group contracts iff a held canon counter-witnesses exactly it, Def 13) at polynomial cost. Order: (canon size, memory index) — smallest group first preserved. (2) **max_walk_states was stored but never enforced** — the slot walk's BFS was bounded only by edges (8) with branching ~2×|memory| per state. Fix: expanded-state counter returns None at the bound (the T2 abandonment the constant always claimed). tests/test_derivation_bounds.py: grown-45-node survey completes and contracts at the right position; disjoint repeated occurrences each place; walk returns None at max_walk_states=8 on a 40-denotation lattice. 64/64 green; all harness entry paths (fresh, mhall-prior, wdmh self-reload, on-demand) complete; the ask's proposal bytes unchanged (33/90/128/74). Note for the user: stale states saved under pre-fix code may still behave oddly — regenerate data/dialogue/*.json if weirdness persists.

*Relevance: critical*
*Context: Fixing the infinite loop in Hop.run*
*Tags: derivation bugfix performance bounds*

---
*Observed: 2026-09-16T17:09:57.826Z*
