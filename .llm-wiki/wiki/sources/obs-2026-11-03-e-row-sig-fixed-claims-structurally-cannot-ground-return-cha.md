---
type: source
title: "Observation: E=row.sig fixed — claims structurally cannot ground; return channel is the missing vertical"
tags:
  - trs
  - cogitator
  - recursion
  - grounding
  - dead-end
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-e-row-sig-fixed-claims-structurally-cannot-ground-return-cha
relevance: high
observed_at: 2026-11-03T17:04:05.164Z
source_context: Verifying E=row.sig + per-candidate frames; proving why goals cannot ground
---

# ⭐ Observation: E=row.sig fixed — claims structurally cannot ground; return channel is the missing vertical

User's fix landed: E = row.sig in the drain (each row re-pairs with its OWN signature — the seed-stuck bug that let non-opening pairings ground the claim is closed; claim now only from the opening row closing), and per-candidate frames (frame = _frame.copy() per candidate iteration; the mint joins the candidate's own frame; _frame base never mutated — candidates isolated from each other). VERIFIED: tests 162/15/16, but NO claim grounds — mhall step 2 blows the 4096 cap (1057+ exhausted halts). STRUCTURAL PROOF of the dead end: a side settles only by emptiness | candidate's own frame | joints. The opening pairing (E=MHALL × SVO goal: F=0, U=MHALL, O=SVO) needs BOTH sides framed: MHALL can never mint (F==MHALL needs an entry ⊇ MHALL; the only MHALL-sig candidates are like-for-like barred — fingerprint MHALL; MHALL:[SVO] unheld until the claim grounds — circular) and never stalls (has overlapping candidates) so no joint; SVO likewise. And recursion results are DISCARDED, so a child grounding its region never settles the parent's side — even drain re-pairing can't help: each fork's _frame contains only its own chain, and the opening's two sides ground in DIFFERENT forks (U fork, O fork) that never merge. RESOLUTION OPTIONS for ruling: (1) return channel — explore returns grounded; parent re-checks the side after the recursion (reference's 'child reentry resolved'; returns are outcomes, not caller state — isolation preserved); (2) drain-accumulating frames — close-mints join the fork's _frame so rows above see rows below as the fork pops (vertical expansion in drain form) — but ALONE cannot ground the opening (two sides, two forks); (3) row latches in the shared heap (horizontal via heap — violates orthogonality). Also noted: the mint pairing no longer emits a pair event (if F==E branch dropped its Event) — mints invisible in the trace now.

*Relevance: high*
*Context: Verifying E=row.sig + per-candidate frames; proving why goals cannot ground*
*Tags: trs cogitator recursion grounding dead-end*

---
*Observed: 2026-11-03T17:04:05.164Z*
