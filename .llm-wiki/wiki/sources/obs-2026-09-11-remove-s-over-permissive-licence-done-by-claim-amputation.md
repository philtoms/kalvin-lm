---
type: source
title: "Observation: Remove's over-permissive licence: done-by-claim-amputation"
tags:
  - ks2
  - model
  - derivations
  - replace
  - semantics
status: observation
created: 2026-09-11
updated: 2026-09-11
slug: obs-2026-09-11-remove-s-over-permissive-licence-done-by-claim-amputation
relevance: high
observed_at: 2026-09-11T08:18:28.957Z
source_context: Reviewing ks2.md derivation licensing after the emptying-ν_A discussion
---

# ⭐ Observation: Remove's over-permissive licence: done-by-claim-amputation

User identified a rule-layer defect: licensed removes (Underfit/Under+over rows) permit claim amputation. Example A = mhal:[m,h,a,l,l] vs B = [a,l,l] reaches done at J=1 while the fixed claim mhal is no longer witnessed — a false done, worse than the known emptying→Unknown case. The blindness is structural: no rule reads A's own fit; γ and band feedback read only the relationship C(A,B), which improves throughout the amputation; §10 absorb then holds the amputated kline in memory as selectable. Agent found a second instance: §7's S3 replace also amputates — A = ab:[ab] vs B = ab:[y] (disjoint y) passes Def 16 selection, replace's never-empty interior constraint holds, yet own gap grows ∅→ab. Proposed fix (user): reframe S2 as asymmetric replacement routed through the §7 composite — removes must be paired with covering candidates; adds need no pairing (excess is benign, claim stays witnessed). Key wrinkle: in Underfit, ν_B∖ν_A = ∅ so candidates must come from M (CONTEXT.md "bridging the gap through Candidates") or the move is unlicensed → underfit-vs-B becomes unfinishable, done requires σ(ν_B) ⊇ s. Open fork: is the head s a fixed commitment (enforce claim-gap-monotone interior constraint) or a revisable hypothesis (amputation = belief revision, absorb rewrites the head)? Nothing decided yet; no doc edits made.

*Relevance: high*
*Context: Reviewing ks2.md derivation licensing after the emptying-ν_A discussion*
*Tags: ks2 model derivations replace semantics*

---
*Observed: 2026-09-11T08:18:28.957Z*
