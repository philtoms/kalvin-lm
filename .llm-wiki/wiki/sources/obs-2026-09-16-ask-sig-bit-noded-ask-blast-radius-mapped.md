---
type: source
title: "Observation: ASK_SIG bit + noded ask blast radius mapped"
tags:
  - ask
  - compiler
  - engine
  - hop
  - harness
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-ask-sig-bit-noded-ask-blast-radius-mapped
relevance: high
observed_at: 2026-09-16T15:18:11.173Z
source_context: ASK kline ASK_SIG|nodes blast-radius investigation
---

# ⭐ Observation: ASK_SIG bit + noded ask blast radius mapped

Investigated the blast of compiling the `==` ask as `WDMH|ASK_SIG:[what,did,Mary,have]` (ASK_SIG = bit 63 = word-word bit 31, nodes = the canon's word nodes) instead of `WDMH:[]`. Probed empirically (dev/dialogue/probe_ask_sig_blast.py, post-run wdmh-underfit state). Verified motivation: empty ask → 0 Def 22 candidates; noded forms → 17 candidates, top `had:[did,have]` then MHALL canon (matches the worked example). Key readings: bit'd form is sig_level S2 (not S4), non-terminal, permanently underfit (the bit is undeliverable residual), is_canon/is_exact False, is_groundable structurally False (kills the spurious empty-ask grounding). Blast surface: (1) the bit sits inside WORD_BITS/_TYPE_MASK → pollutes every measurement: J(WDMH,MHALL) 0.143→0.125, misfit_mass 6→7 — needs a mask-vs-accept decision; (2) ask-ness leaves structure (noded+covered reads S2/S1) → KSignifier.is_ask must be reintroduced (removed in b0d9403); (3) harness graded() skips noded asks → the ask would feed at compile-time S4 → _fast_route refuses it on arrival; (4) is_ask_content/_answer key pools by signature value — the bit'd ask misses the plain-sig canon pool (answer release dead, canon leaks into scaffolding feed); (5) _drive discriminates ask-vs-proposal by empty nodes — noded asks escalate instead of signature-discovery reply; (6) candidate_goals self-exclusion compares (sig,nodes) — the bit defeats it: after canon release the ask's own canon becomes its γ=1.0 top goal (verified); exclusion must compare content; (7) ast_emitter _mts_canonicalise_seen key (sig,nodes) collides authored ask with MTS canon — authored ask swallowed; key needs op; (8) decoder ASK path raises on nodes; (9) StructuralSupervisor.grade's under(bit'd-sig) misses script entries — recovers only via the clean nodes_sig content branch; S1 branch unreachable; (10) Engine._propose stamps the bit on all ask-derived proposals and _reentry propagates it down hop chains. Central tension: b0d9403 removed the bit because "the ask is structural" (empty ⟺ S4); with nodes the bit must BE part of the structure — rule: residual == ASK_SIG exactly ∧ covered nodes ⟹ the ask (S4). No-bit noded alternative is identity-collapsed with the canon (is_canon True, groundable) — question and answer the same kline; some marker is required, the bit is minimal.

*Relevance: high*
*Context: ASK kline ASK_SIG|nodes blast-radius investigation*
*Tags: ask compiler engine hop harness*

---
*Observed: 2026-09-16T15:18:11.173Z*
