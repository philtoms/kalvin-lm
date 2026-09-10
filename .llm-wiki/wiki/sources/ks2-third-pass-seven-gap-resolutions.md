---
type: source
title: "ks2 third pass: seven gap resolutions fixing the formal algebra"
status: insight
category: formalisation
created: 2026-09-10
updated: 2026-09-10
slug: ks2-third-pass-seven-gap-resolutions
---

# ks2 third pass: seven gap resolutions fixing the formal algebra

Third pass on [[docs/ks2.md]] resolved all seven review gaps via [[concepts/grilling]] rounds (one gap per round, numbered questions with recommendations). Seven decisions:

1. **G1 — well-founded witnesses, unrestricted memory.** Any kline may be held (cycles included); expand/contract license only canons with `n ∉ νₙ`. The cycle collapse theorem justifies the clause: a canon's nodes are atom-subsets of its head, so expansion cycles force atom-equality — a self-containing canon. Contract is gated too (congruence symmetry). §8 decidability re-derived from terminating licensed expansion alone. See [[sources/obs-2026-09-10-ks2-md-review-verified-sound-core-7-gaps-incl-witness-dag-in]].

2. **G2 — replace interior.** Never-empty interleaving constraint; interiors provably ∈ {S3, S2, done}; early done legitimated (value-equality outruns node-equality — pending nodes are witness structure).

3. **G3 — endings.** Stuck = not done ∧ no licensed *targeting* move; witnessed moves explicitly non-ending (band-preserving, exposure-only); abandonment named as the third, strategic ending (engine's visited-sets/hop-bounds are exactly this).

4. **G4 — γ fixed.** `γ = J(σ(ν_A), σ(ν_B)) · δ^D̄` — Jaccard core (symmetric, band-consistency) times δ to A-side atom-weighted mean witness depth (directional: how hard-won). Three requirements: band-consistency, granularity-invariance (kills unweighted mean-compose), depth-monotonicity (expand ↓ / contract ↑). Engine's mean_compose is a known non-conforming instance, to reconcile.

5. **G5 — ask atom removed.** User judged ASK_BPE_TOKEN a hack routing canons that would otherwise ground; engine removal planned. A = {a₀…a₃₀}; the ask is structural only — Unknown (s:[]) is its shape. See [[sources/obs-2026-09-10-ask-bpe-token-to-be-removed-ks2-drops-the-ask-atom-ask-is-st]].

6. **G6 — selection excludes empty targets.** grounded(B) excludes the Unknown shape (engine: "an unknown never grounds"); ν_B = [] unreachable via selection.

7. **G7 — fit total over empty heads.** Case 1 extended: ν = [] or s = ∅ → Unknown (nothing held / nothing left). Stuck has two reachable ask conditions: no target (at entry) or ν_A emptied by permissive removes (acquired mid-run). Also fixed §10 species error (a:[c,a] is Overfit, not Denotation).

**Process lesson:** the gap-by-gap grilling rhythm (facts gathered from engine source first, then 2–3 numbered questions with recommendations, apply-on-accept, commit per gap) worked well for a formalisation doc — each round's answer sometimes forced revision of the previous round's edit (G7 revised G6's totality-only note), which surfaced honestly rather than being papered over.

**Deferred reconciliation (ks2 leads):** engine — remove ASK_BPE_TOKEN, reconcile mean_compose with canonical γ, decide is_connotation divergence (sig⊆node S2 shapes vs case-4/S3); CONTEXT.md — Token ID/ASK/Candidates/nine-structures entries (retain row); engine _candidates selects by signifies-overlap while Def 16 selects by t ∈ ν_A.

*Category: formalisation*

---
*Captured: 2026-09-10*

## Related

_Add links to related pages._
