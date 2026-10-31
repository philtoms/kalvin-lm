---
type: source
title: "Observation: TRS R3 loop carries shrinking signature — consumed set removed"
tags:
  - trs
  - rationalise
  - redesign
  - residual-signature
  - termination
  - r3-loop
status: observation
created: 2026-10-16
updated: 2026-10-16
slug: obs-2026-10-16-trs-r3-loop-carries-shrinking-signature-consumed-set-removed
relevance: high
observed_at: 2026-10-16T16:10:36.542Z
source_context: "TRS rationalise redesign, step 4: residual signature replaces consumed set"
---

# ⭐ Observation: TRS R3 loop carries shrinking signature — consumed set removed

Redesign step 4 implemented in src/trs/model.py (uncommitted): consumed replaced by an increasingly smaller signature. The loop carries `grounded` (accumulated bits) and recomputes res_sig = entry.sig & ~grounded, res_adj = ungrounded nodes each round; candidates derive from overlap(res_sig) — the substituted supplier drops STRUCTURALLY (its overlap with the claim is exactly what it grounded), no bookkeeping. Grounding events: S1 candidate (grounds rc), S2 canonical fit (grounds F; the U-side becomes the residual IN-LOOP — no U-product recursion; only O-side re-enters), S3 substitution measured in NEW bits f = rc & can(n′) & ~grounded. One grounding per round, then re-derive. Exit: residual asks (connect/acquire) or claim fully grounds → admit ⟨c_e, work⟩ when an S3 sub licensed it (fit-only grounding does not re-mint — avoids overfit restatement of the fit). TERMINATION BUG the user caught (first version hung): shrank must mean grounded GREW — a witness covering only already-grounded bits must not count as shrink, else the round re-derives identically forever. Semantic gains on mhall step 2: claim now MHALL:[a,little,lamb,MarySubject,hadVerb] — ALLQuery pollution GONE (Query supplier dropped once object bits grounded via the ALL fit), Mary|had intermediate claim GONE (loop carries the U-side), zero halts, zero pending proposals — the ask grounds to completion. New ask semantics: 'R4: substitution grounds nothing' when the rewrite carries no NEW claim bits (the old contentless licensed claim ⟨C,{A,B}⟩ no longer mints — ask instead); the residual names the ask (⟨B,{B}⟩ connect). 4 tests updated, 2 added (grounds-nothing, structural drop). 171 pass.

*Relevance: high*
*Context: TRS rationalise redesign, step 4: residual signature replaces consumed set*
*Tags: trs rationalise redesign residual-signature termination r3-loop*

---
*Observed: 2026-10-16T16:10:36.542Z*
