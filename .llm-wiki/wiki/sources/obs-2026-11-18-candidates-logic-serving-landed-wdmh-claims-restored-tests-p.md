---
type: source
title: "Observation: Candidates-logic serving landed — wdmh claims restored, tests ported"
tags:
  - cogitator
  - harness
  - scaffolding
  - candidates
  - stall
status: observation
created: 2026-11-18
updated: 2026-11-18
slug: obs-2026-11-18-candidates-logic-serving-landed-wdmh-claims-restored-tests-p
relevance: high
observed_at: 2026-11-18T18:25:14.062Z
source_context: Porting tests and docs to the user's candidates-logic stall semantics
---

# ⭐ Observation: Candidates-logic serving landed — wdmh claims restored, tests ported

User reshaped the on-demand scaffolding design (their edits, unstaged on top of my staged work): _Entry→public Entry; _candidates→module-level candidates(sig, entry, d) exported from rationalisation (with common_ancestor); on_stall contract now Callable[[KLine, Entry], bool] — the sink receives the stalled kline AND its entry, applies the candidates logic to the harness's scaffolding Domain (now a Domain, not a dict) from the stalled entry's seat, adds qualifying klines to the live domain, answers served=True; the cogitator then retries via the ordinary _push(A, entry.parent, entry.comp) — the stalled component re-descends from its parent with at-map gating (cleaner than my hand-rolled Entry copy). Served stalls are NOT filed in st.stalled and emit no event (the retry's turns report instead); unserved stalls file+broadcast. Exchange (S3) thus remains only for unserved signatures. Effect on scripts: wdmh.ks claims RESTORED (MHALL:[SVO] and WDMH:[MHALL] both claim again, 44 resolved vs 49 in the old fully-primed walk); mhall.ks near-identical to the primed baseline; mhall-ag.ks byte-identical (identities barred as candidates, unknowns nodeless — nothing qualifies). I ported 7 broken tests to the new sink contract, updated harness docstrings + docs/rationalisation.md (§2.1/§2.2/§2.3: candidates-logic serving, the answering sink). Noted asymmetry (left as-is): the sink bars candidates by the stalled entry's chain (includes parent.A) while _push recomputes with the parent's chain (excludes parent.A) — a scaffold with sig == the parent ask's sig is never served, which matches the parent-ask bar's intent.

*Relevance: high*
*Context: Porting tests and docs to the user's candidates-logic stall semantics*
*Tags: cogitator harness scaffolding candidates stall*

---
*Observed: 2026-11-18T18:25:14.062Z*
