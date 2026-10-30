---
type: source
title: "Observation: Framing is on-path only: F==E or stalled U==O exchange"
tags:
  - trs
  - framing
  - on-path
  - resolution
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-framing-is-on-path-only-f-e-or-stalled-u-o-exchange
relevance: high
observed_at: 2026-10-30T13:31:37.753Z
source_context: On-path framing correction
---

# ⭐ Observation: Framing is on-path only: F==E or stalled U==O exchange

User's rule: \"All framing is on-path: F==E or U==O (stalled).\" Implemented as a resolution-semantics change, not just display: (1) _side_ok sides settle via a resolved child (incl. stalled+exchanged), or a frame whose producing reentry is in the row's ANCESTOR CHAIN (established on the way down), or a joint/exchange frame — the GLOBAL frame pool no longer feeds resolution at all (it remains the learning record/stores); (2) _run's already-known skip uses the same on-chain check; (3) _cover_frame restricted to chain/joint frames; (4) frame column renders the row's own F==E frame + on-path supports citing producers (prefer tabled > on-chain > any), deduped. The invented \"Mary ← MS:Subject\" off-path references are gone — that display had no rule behind it. Consequences: (MHALL × MS) now dies honestly in mhall (its HALL branch R4-halts on empty-adj children — had:[], Query:[] — that off-path frames used to cover); the ALLQ branch closes instead — mhall 10 rows, wdmh 7, all frames F==E/exchange, all citations to tabled rows except wdmh's \"HALL ← MHALL\" which cites (HALL × MHALL-canon), an F==E pairing of the TABLED reentry but a sibling row not itself tabled. The remaining semantic edges for the user: (a) that sibling-row citation, (b) wdmh WDH row's fit what still frames nowhere (honest under (c)), (c) mhall row 2 is now ALLQ not MS — their originally expected MS row 2 does not survive strict on-path framing. 134 tests pass, deterministic.

*Relevance: high*
*Context: On-path framing correction*
*Tags: trs framing on-path resolution*

---
*Observed: 2026-10-30T13:31:37.753Z*
