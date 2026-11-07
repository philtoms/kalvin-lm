---
type: source
title: "Observation: User recode verified: drains whole stack copy, grounds both scripts at 28 pairings"
tags:
  - trs
  - cogitator
  - recursion
  - user-recoded
  - verification
status: observation
created: 2026-11-03
updated: 2026-11-03
slug: obs-2026-11-03-user-recode-verified-drains-whole-stack-copy-grounds-both-sc
relevance: high
observed_at: 2026-11-03T16:10:02.519Z
source_context: Verifying the user's recode of the Cogitator recursion
---

# ⭐ Observation: User recode verified: drains whole stack copy, grounds both scripts at 28 pairings

User recoded the recursion to their design requirements (cogitator.py, uncommitted): explore(E, stack, frame, memo, depth, psig, pcsig) pushes its row, copies stack/frame/memo, drains its WHOLE stack copy unguarded (`while stack:`) — including the inherited path; returns None (results never flow up). Key semantics: (1) close check fires BEFORE recursion — u_ok/o_ok computed from the explore's own frame at pairing time; a side that doesn't settle gets ONE recursive explore (U first, then `continue`; O likewise) and the candidate moves on without it — the ONLY way a fork's work settles the parent is the global joint list (exchanges). (2) The mint add_frame(F, frame) joins the CALLER's frame — visible to the caller's later candidates (frame scope = the explore, not the candidate); the fork inherits a copy. (3) E stays the SEED's signature through the whole drain: ancestor rows re-pair the seed at their depth/psig/pcsig (VO appears at depths 7,6,3,2,1; WDMH step 4's claim grounds from `| 1 | VO | HV:Verb |… grounded |` — a depth-1 row closing on a non-goal pairing mints Kline(entry,{goal}) — the claim need not come from the opening pairing). RESULT: both scripts ground with the same four stores as every prior working engine (SVO canon, MHALL:[SVO], MHALL canon, WDMH:[MHALL]); wdmh step 4 costs 28 pairings (vs the 4096 cap under my strict-fork drafts); rows=10/explore-step, exchanges fire at multiple depths as forks re-meet parked stalls (DH: stall d3, exchange d2 and d1). Tests 162/15/16. Open characterizations for user ruling: E-fixed drain and claim-from-any-depth-1-close are load-bearing outcomes of this shape; exhausted halts (WDH, SV, HALL ×4) recorded alongside the grounded claim.

*Relevance: high*
*Context: Verifying the user's recode of the Cogitator recursion*
*Tags: trs cogitator recursion user-recoded verification*

---
*Observed: 2026-11-03T16:10:02.519Z*
