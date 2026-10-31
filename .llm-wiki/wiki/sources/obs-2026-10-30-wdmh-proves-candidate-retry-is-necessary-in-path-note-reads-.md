---
type: source
title: "Observation: wdmh proves candidate retry is necessary; in-path note reads as entry-path rule"
tags:
  - trs
  - in-path
  - retry-necessity
  - candidate-order
  - wdmh
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-wdmh-proves-candidate-retry-is-necessary-in-path-note-reads-
relevance: high
observed_at: 2026-10-30T16:25:05.528Z
source_context: Verifying mhall/wdmh trace tables before rewriting Cogitator.rationalise
---

# ⭐ Observation: wdmh proves candidate retry is necessary; in-path note reads as entry-path rule

Two findings from table verification. (A) RETRY/GENERALITY IS PROVEN NECESSARY: with μ-least-first candidate order (the current code's sort), wdmh's HALL entry pairs HV:Verb first — that subtree fully resolves HALL but never mints the second DH stall, so the exchange never fires, WDH stays blocked and the run cannot ground; only HALL×H:did-have (5th in μ-order) produces the DH stall that exchanges and grounds the run. So either the flood-pair-everything architecture (current) or DFS with candidate retry on global blockage is required — a single-shot ordered descent cannot reach the wdmh goal. (B) IN-PATH RULE RESOLVED: the note "MHALL -> HALL -> MHALL is illegal" must be read as the ENTRY-path rule (a candidate whose content equals an ancestor ENTRY's content is illegal) — because the expected mhall table itself contains SVO→VO→SVO (row 3/4: the o-child VO re-meets its origin candidate SVO), the same shape under a content-flow reading. Under the entry-path reading, wdmh's untabled HALL×MHALL pairing (raw row 16, legal under the current code's first-hop origin exemption) is legal-but-off-path. The current _in_path implements content-flow (o-side candidate flow), NOT the entry-path rule — the banned u-chain cycle (ancestor entry content re-appearing as candidate) is unblocked.

*Relevance: high*
*Context: Verifying mhall/wdmh trace tables before rewriting Cogitator.rationalise*
*Tags: trs in-path retry-necessity candidate-order wdmh*

---
*Observed: 2026-10-30T16:25:05.528Z*
