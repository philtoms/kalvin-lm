---
type: source
title: "Observation: Ends gone: stall sink mirrors the claim channel"
tags:
  - rationalisation
  - cogitator
  - streaming
  - stalls
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-ends-gone-stall-sink-mirrors-the-claim-channel
relevance: critical
observed_at: 2026-11-17T14:55:41.635Z
source_context: Removing ends; stall sink follows the claim channel
---

# 🔴 Observation: Ends gone: stall sink mirrors the claim channel

Ends removed (user-directed, third streaming step): Rationalisation is now events alone. Cogitator gains on_stall (constructor sink, symmetric to on_claim); unaligned stalls broadcast at the run's end as bare KLine(sig, []) — 'the claim channel's equal'. _halt and the bound-halt End are gone; the entry cap now stops the run silently (no record — flagged to user). report.py: End class and proposals() deleted; Proposal survives as the harness's pending shape (kind always 'connect' for stall broadcasts; acquire/continue kinds died with their reasons). Harness: _stall sink appends to runner.stalls ledger AND files Pending(Proposal('connect', kline, effort(kline))) — the pending/discharge machinery is now fed live by the sink, not from the returned report. Halt display reads the stall ledger (present() takes stalls param). Tests: proposals-cover-ends test replaced by test_unaligned_stalls_broadcast_bare (both starved sides broadcast — note k(C,C) over base_domain+k(ABC,AB) actually GROUNDS via the ABC candidate; only a barren substrate stalls); honest property rewritten on the stall sink (bare S4, effort 1, domain untouched). 4 xfail debt tests still reference proposals/ends — NameError → xfail, same tolerance as their r.stores references. Suite: 23 passed, 16 xfailed, 0 failed; probe parity.

*Relevance: critical*
*Context: Removing ends; stall sink follows the claim channel*
*Tags: rationalisation cogitator streaming stalls*

---
*Observed: 2026-11-17T14:55:41.635Z*
