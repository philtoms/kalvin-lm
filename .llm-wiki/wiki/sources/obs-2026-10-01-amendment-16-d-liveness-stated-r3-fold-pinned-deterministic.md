---
type: source
title: "Observation: Amendment 16: D-liveness stated; R3 fold pinned deterministic"
tags:
  - trs
  - d14
  - d-liveness
  - amendment-16
  - r3-fold-determinism
  - rescue
status: observation
created: 2026-10-01
updated: 2026-10-01
slug: obs-2026-10-01-amendment-16-d-liveness-stated-r3-fold-pinned-deterministic
relevance: high
observed_at: 2026-10-01T14:35:08.533Z
source_context: Accounting for D-growth during the run in the spec
---

# ⭐ Observation: Amendment 16: D-liveness stated; R3 fold pinned deterministic

Amendment 16 (D14): D-LIVENESS stated in the spec §5 — 'D is live throughout the tree: R1 stores enter D at the moment of arrival, and every later application — lookup, witness, substitution — reads D as it then stands. A run's own writes can close a route a later candidate would have taken, or open one... The tree is a sequence of applications, not an evaluation against a snapshot.' Triggered by the user: desired-but-unexpected behaviour means the truth document must account for it. CORRECTED MECHANISM from live verification: the mid-run rescue was NOT simply 'a fresh witness kept the expansion alive' — by round 2, covered(AB) held THREE AB-headed klines (terminal original + round-1 S1 store ⟨AB,{AB}⟩ + round-2 origin store ⟨AB,{C}⟩), and the R3 fold n_prime = (n_prime ∖ {sig}) ∪ adj(K) applied per occurring K is ORDER-DEPENDENT (a later discard strips an earlier contribution) — the observed N′={C} was one set-iteration order's outcome. §7 already scopes 'selection order among R3 substitutions' as strategy, but the reference was silently instantiating an arbitrary (set-order) strategy. FIX: r3_substitute now folds covered klines sorted by (sig, adjacency) — deterministic, documented as the reference's §7 strategy instantiation. REGRESSION: test_d_is_live_during_the_run pins the ruling — entry ⟨C,{AB}⟩ against D₀ = atoms + ⟨AB,∅⟩ + ⟨ABC,{AB}⟩ mints both stores mid-run and NO emptied-expansion halt fires; entry ⟨C,{C}⟩ against the same D₀ (no stores intervene) fires it. 154 tests passing. Uncommitted.

*Relevance: high*
*Context: Accounting for D-growth during the run in the spec*
*Tags: trs d14 d-liveness amendment-16 r3-fold-determinism rescue*

---
*Observed: 2026-10-01T14:35:08.533Z*
