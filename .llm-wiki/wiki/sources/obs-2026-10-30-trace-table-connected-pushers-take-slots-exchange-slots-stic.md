---
type: source
title: "Observation: Trace table connected: pushers take slots; exchange slots sticky"
tags:
  - trs
  - trace-table
  - connectedness
  - repair
status: observation
created: 2026-10-30
updated: 2026-10-30
slug: obs-2026-10-30-trace-table-connected-pushers-take-slots-exchange-slots-stic
relevance: high
observed_at: 2026-10-30T10:05:04.614Z
source_context: Fixing the floating Subject reentry in the orthogonal trace table
---

# ⭐ Observation: Trace table connected: pushers take slots; exchange slots sticky

Connected-table invariant enforced: every tabled row's entry (except the root and the root's own pairing) is claimed as a reenter-child by an earlier tabled row's step. Post-walk repair pass in Cogitator._trace: compute claimed reentries (children with kept rows or kept stalls); any floating row's PUSHER row takes its reentry's table slot (replacing the first-reached holder) — e.g. ⟨SV⟩'s slot moved from (SV,SVO) to (SV,HV:Verb) so the Subject reentry is claimed. Pre-seeded exchange-feed slots are STICKY (never displaced — the repair otherwise oscillated between (HALL,ALL:Object) wanting the had-chain and (HALL,had:DH) wanting the exchange); a floating row that cannot connect without displacing a sticky holder is dropped. Final wdmh path 11 rows, fully claimed: WDMH→WDH(reenter DH,Object)→DH stalled; Object→SV(HV:Verb reenter Subject)→Subject(SVO)→VO(QO)→Verb(SVO)→SO(SVO); HALL×had:did-have (reenter DH)→DH resolved (exchange). mhall 12 rows, 0 floating. Verification probe: recompute claimed set via _on_path_children over the path; both scripts report floating:0. 134 tests pass.

*Relevance: high*
*Context: Fixing the floating Subject reentry in the orthogonal trace table*
*Tags: trs trace-table connectedness repair*

---
*Observed: 2026-10-30T10:05:04.614Z*
