---
type: source
title: "Observation: Rulings: flood-then-distill, orthogonal paths, diagnostic harness"
tags:
  - trs
  - rulings
  - parallelism
  - flood
  - distill
  - orthogonality
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-rulings-flood-then-distill-orthogonal-paths-diagnostic-harne
relevance: high
observed_at: 2026-10-30T16:47:34.019Z
source_context: Implementing Cogitator.rationalise fixes after trace-table verification
---

# ⭐ Observation: Rulings: flood-then-distill, orthogonal paths, diagnostic harness

User rulings on the trace-table verification: (1) In-path rule confirmed as entry-path reading; no on/off-path distinction — "all paths are orthogonal. The table is one of possibilities" — the distilled trace is one valid orthogonal path, not a required exact reproduction; one row per reentry (orthogonality). (2) Architecture choice: whichever is more amenable to parallelism — flood-then-distill wins (candidate subtrees explore independently, interact only via frames/exchange stalls/D; ordering is only a distillation preference). (3) Harness output is diagnostic only; keep the underlying model clean, but maintain trace shape in practice. Implementation consequences: μ-least-first candidate order with reversed child pushing gives the determinism (the μ-least candidate's whole subtree completes before the next candidate's — the MS chain finishes before ALLQ's, so resolved_by slots pick the reductive chain); self-content pairings (side(c)==side(entry), like-for-like) must be excluded or the ALL-canon/SVO-canon identity pairings short-circuit the tabled chains; resolution frames its reentry's region (record-only, never settles — produces the final "MHALL, SVO"/"WDH, HALL" frame columns); vacuous exchanges (union adjacency self-witness only, e.g. Mod:[Mod]) skip the admit and the sticky slot-claim (root cause of the wdmh ALL→Q→O chain vanishing from the distilled path); slot = resolved_by (first-to-resolve), exchange feeds stay sticky per prior ruling "the S3 path is the path".

*Relevance: high*
*Context: Implementing Cogitator.rationalise fixes after trace-table verification*
*Tags: trs rulings parallelism flood distill orthogonality*

---
*Observed: 2026-10-30T16:47:34.019Z*
