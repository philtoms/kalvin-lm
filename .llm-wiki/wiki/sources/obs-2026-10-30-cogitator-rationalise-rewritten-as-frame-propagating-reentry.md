---
type: source
title: "Observation: Cogitator.rationalise rewritten as frame-propagating reentry runner"
tags:
  - trs
  - cogitator
  - reentry
  - framing
  - exchange
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-cogitator-rationalise-rewritten-as-frame-propagating-reentry
relevance: high
observed_at: 2026-10-30T07:16:01.472Z
source_context: Rewriting Cogitator.rationalise to generate the expected ask→goal trace
---

# ⭐ Observation: Cogitator.rationalise rewritten as frame-propagating reentry runner

Rewrote src/trs/model.py Cogitator.rationalise as an agenda-driven reentry runner per the expected mhall/wdmh trace tables. Key mechanics: side(k)=sig∨can(adj) is the full content a pairing decomposes against; F==side(E) frames the fit immediately; U/O children reenter as ⟨region, witness nodes contained in region⟩ (LIFO, U-side leads); a row resolves when both sides are framed (empty/covered-by-frames/child-resolved); resolution frames the row's F and the reentry's sig (pure frame-set propagation, no parent links); goal row = root×goal pairing, its resolution admits the claim ⟨ask.sig, {goal.sig}⟩ (MHALL:[SVO], WDMH:[MHALL] both ground); S3 root exchange = a U-side and an O-side reentry stalled at S4 on the same signature → admit ⟨sig, N₁∪N₂⟩ (S1) and frame it (wdmh grounds did|have:[did,have]). Old seen-by-entry-fingerprint blocked the depth-2 MHALL reentry — new seen keys (entry,candidate) pairs. dev/dialogue/trs_harness.py passes goal into the ask step and renders the row table. Stale tests: tests/test_trs_model.py and tests/test_trs_properties.py still import a module-level rationalise removed in e93c192 (pre-existing breakage, unchanged).

*Relevance: high*
*Context: Rewriting Cogitator.rationalise to generate the expected ask→goal trace*
*Tags: trs cogitator reentry framing exchange*

---
*Observed: 2026-10-30T07:16:01.472Z*
