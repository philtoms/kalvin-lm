---
type: source
title: "Observation: On-demand scaffolding landed — stall sink live at S4, retry by stack push"
tags:
  - cogitator
  - harness
  - scaffolding
  - stall
  - retry
status: observation
created: 2026-11-18
updated: 2026-11-18
slug: obs-2026-11-18-on-demand-scaffolding-landed-stall-sink-live-at-s4-retry-by-
relevance: high
observed_at: 2026-11-18T17:20:53.313Z
source_context: "Implementing on-demand scaffolding: stall event, harness lookup, stack-push retry"
---

# ⭐ Observation: On-demand scaffolding landed — stall sink live at S4, retry by stack push

On-demand scaffolding landed (user-directed design). Model (src/rationalisation/cogitator.py): on_stall now fires at S4 LIVE, mid-run (was: at run end via _finish loop); after the sink returns, the stalled entry reenters — a fresh _Entry (same st, comp, chain, parent=stalled.parent) pushed onto the stack iff _candidates(A) is non-empty, i.e. only when the sink grew D. Retry is a push, never a block; RLock makes the sink-initiated D.add + stack push safe inside _turn. Harness (dev/dialogue/harness.py): non-paired entries now held in runner.scaffolding: dict[int masked sig, list[KLine]] instead of primed; D starts empty; _stall sink looks up the stalled signature exactly, Domain.add's True-when-new marks the stall served (served stalls are NOT filed as open proposals; unserved ones are). Results: mhall-ag.ks byte-identical (both MHALL and SVO stall, nothing served); mhall.ks collapses from rich primed walk to the same two root stalls (its scaffolding never exactly matches a stalled region); wdmh.ks PROVES serve end-to-end — step 1's MHALL-region stall is served (held MHALL:[had,ALL] canon, sig 0x1f00000000 == the region), lands in D, and step 2 reads it as a candidate (HALL/Mary resolved). The old wdmh claim (WDMH:[MHALL]) is gone: WDH-region stall has no exact-match scaffolding, so the framers stay held — honest state, motivates richer lookup later. Retry push itself is chain-barred in wdmh (scaffold sig == root ask sig — the parent-ask bar); unit test test_a_served_stall_reenters_with_fresh_candidates proves the retry path directly (empty domain, sink serves k(9,[1,8]) at the U-stall, retried entry aligns). docs/rationalisation.md updated (§2.1 held-not-primed, §2.2/§2.3 stall leaves at S4 live + reenter rule, §2.4, T2 tuple now includes stall answers). Suite: 163 passed, 16 xfailed.

*Relevance: high*
*Context: Implementing on-demand scaffolding: stall event, harness lookup, stack-push retry*
*Tags: cogitator harness scaffolding stall retry*

---
*Observed: 2026-11-18T17:20:53.313Z*
