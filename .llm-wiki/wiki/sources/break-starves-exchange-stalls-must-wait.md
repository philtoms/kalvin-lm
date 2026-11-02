---
type: source
title: "Reentry probe: break starves the exchange; stalls must wait"
status: insight
category: architecture
created: 2026-11-02
updated: 2026-11-02
slug: break-starves-exchange-stalls-must-wait
---

# Reentry probe: break starves the exchange; stalls must wait

Emulating the reference's S3 exchange requirement in the simple reentry probe (dev/probes/probe_simple_reentry.py) surfaced a second rule the pseudocode got wrong besides the fit gate: **the pseudocode's `break` after ground starves the exchange.** A stall now waits (no self-witness); two stalls on one value from different parents mint one joint frame, nothing stored — the reference's `_exchange` rule (`r.parent is not a.parent`), emulated in 8 lines with a waiting dict keyed by value; no stall groups, no back-propagation bookkeeping. But with `break`, HLAL grounds on MHALL:[had,ALL] and the NEXT candidate in insertion order — had:[did,have], the S3 canon — never pairs, so the second DH stall never arrives and WDMH cannot ground. The reference pairs every candidate up front (its "independent batches"); the probe now does the same: the for-run never breaks on an inner ground, the entry pops at its FIRST grounding candidate, and the claim alone stops the run. After both fixes the probe agrees with the reference on every scenario of mhall.ks, wdmh.ks and the two-scenarios script (mhall 161 pairs/2 stalls/5 exchanges; wdmh step4 131 pairs/3 stalls/4 exchanges, claim WDMH:[MHALL]). Detail worth knowing: the probe's wdmh exchange pairs HLAL×had:did-have's DH stall with HAL×had:did-have's (both o-sides, ancestor-descendant rows) where the reference's canonical pair is WDH's u-side stall meeting HLAL's o-side stall — same rule (same value, different parents), different witness pair; the probe's LIFO explores the o-subtree first, the reference's u-side leads. The exchange is symmetric so both ground. A transverse-independence requirement (no common ancestry) would be stricter than the reference implements.

*Category: architecture*

---
*Captured: 2026-11-02*

## Related

_Add links to related pages._
