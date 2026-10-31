---
type: source
title: "Observation: Known resolutions trace as rows — the Object twin surfaces between pair and close"
tags:
  - trs
  - wdmh
  - trace
  - visibility
  - known-resolutions
  - quiet-branch
status: observation
created: 2026-10-31
updated: 2026-10-31
slug: obs-2026-10-31-known-resolutions-trace-as-rows-the-object-twin-surfaces-bet
relevance: high
observed_at: 2026-10-31T08:47:42.257Z
source_context: Fixing invisible already-known reentry resolutions after user review
---

# ⭐ Observation: Known resolutions trace as rows — the Object twin surfaces between pair and close

User flagged wdmh row 17 (the SV×SVO close, "BP Object") as shadowing row 16 with no visible trace of the "reenter Object" child it pushed. Diagnosis: the depth-7 Object twin resolved through the QUIET branch — every closer was a proven no-op (frame Object owned by a sibling-subtree row, frame SV minted by its own parent) — so it resolved as already-known with NO row, and the SV row's o-side settled invisibly. Same invisibility class as the HALL×MHALL slot displacement, but the invisible thing is a whole reentry resolution. Fix (commit e82fc34): known resolutions record a row — candidate None, `known=True`, step "known, BP …" — covering both the quiet branch (all closers barred/proven) and the pop-time ancestor-frame skip; single emission (born closed); rframe recorded so leans can cite it. wdmh now shows "7 Object … known, BP SV" between the SV pairing (row 16) and its close (row 18); mhall's Verb→SO and Subject→VO twins surface the same way. NOTE: user has been committing in parallel (abd3b4b step strings 'grounded'/'BP' plain; 7dec1ea model strip removing 9 legacy R2/R3 tests — 176→167 baseline; abaafdc wiki) — the known-row change is neutral on the stripped baseline (A/B verified).

*Relevance: high*
*Context: Fixing invisible already-known reentry resolutions after user review*
*Tags: trs wdmh trace visibility known-resolutions quiet-branch*

---
*Observed: 2026-10-31T08:47:42.257Z*
