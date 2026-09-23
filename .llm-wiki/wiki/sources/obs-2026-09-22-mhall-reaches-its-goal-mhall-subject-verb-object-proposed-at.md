---
type: source
title: "Observation: mhall reaches its goal: MHALL:[Subject, Verb, Object] proposed at S1 via connotation walks"
tags:
  - mhall
  - ask
  - walk
  - connotation
  - bridge
  - implementation
  - goal-reached
status: observation
created: 2026-09-22
updated: 2026-09-22
slug: obs-2026-09-22-mhall-reaches-its-goal-mhall-subject-verb-object-proposed-at
relevance: critical
observed_at: 2026-09-22T17:33:50.987Z
source_context: Implementing the amended walk/S4 semantics in the rationaliser
---

# 🔴 Observation: mhall reaches its goal: MHALL:[Subject, Verb, Object] proposed at S1 via connotation walks

The amended algebra implemented and verified end-to-end on mhall: the canonical run now PROPOSES again. Trace: T02 feeds MHALL:[Mary, had, a, little, lamb] S4 → the ask attends → walks its underfit slots through the connotations (M→MarySubject→Subject, H→hadVerb→Verb, ALL→Object) → proposes MHALL:[Subject, Verb, Object] S1 255 → the == goal grades it (γ=1) → re-fed S1 → grounds. wdmh.ks unchanged (WDMH:[ALL, had, Mary] S1 255); 90/90 tests green throughout. Four code seams: (1) rationaliser._fast_route — asks fed at S4 attend instead of refuse+remove (S4-optimism); (2) hop.candidate_goals — the == declared goal enters by declaration (dbg.goal label match over held klines, coverage waived) and goals read held content only (frame/ltm — STM bridges excluded; they were polluting selection and crowding SVO out of the top-8); (3) derivation — two-form walk (_walk_neighbours: descent by heading + ascent into covering heads with inert self-ascent guard) shared by descend/_meet; bridge target = the ν_B overfit slot (meeting value when a goal node, else B's departure — §9 form); _canonicalise's _exposes gate removed (doc-faithful survey, no exposure requirement); (4) hop._reentry — dbg rides the re-entry so the ask's goal declaration persists across hops (this was the last blocker: hop 3+ lost the declared goal and stalled at [Mary, had, ALL]). Honest caveats: ALL→Object bridged via the lamb:[Object] denotation (3 edges), NOT the Query 2-hop chain (4 edges) — the L=O binding shortcuts the stress test; the Query chain remains unexercised in mhall. The six S2/S3 scaffolds still attend (their evidence-kline derivation is future work). The ask-as-grounded-canon (Phil's "an ASK is grounded because it is a canon") is NOT yet implemented — only the routing change; §13 doc sentences still pending Phil's wording. Probes kept: probe_mhall_ask_hops.py, probe_mhall_bridge_provenance.py (+ earlier stall/attends probes).

*Relevance: critical*
*Context: Implementing the amended walk/S4 semantics in the rationaliser*
*Tags: mhall ask walk connotation bridge implementation goal-reached*

---
*Observed: 2026-09-22T17:33:50.987Z*
