---
type: source
title: "Observation: Engine core ported: two-ended guard, ν_B walk, b_walks control — 43 tests green"
tags:
  - kalvin
  - derivation
  - engine
  - def17
  - def14
  - port
  - tests
status: observation
created: 2026-09-14
updated: 2026-09-14
slug: obs-2026-09-14-engine-core-ported-two-ended-guard-b-walk-b-walks-control-43
relevance: high
observed_at: 2026-09-14T15:51:12.681Z
source_context: Porting two-party Def 17 + Def 14 into the engine core
---

# ⭐ Observation: Engine core ported: two-ended guard, ν_B walk, b_walks control — 43 tests green

Ported the two-party Def 17 and Def 14 into the engine core src/kalvin/derivation.py (uncommitted): targetings() S2 guard now two-ended (forward departs gap OR arriving witness adopts excess; reverse consumes gap OR head lands in excess — closes the case where the literal A-centric guard blocked all forward moves in pure overfit); slot_walk() parameterized end_mask (excess for ν_A slots, σ(ν_A) for ν_B slots); _walk split into _walk_a/_walk_b — the B-side departs the goal's overfit nodes, arrives at the anchor, composes head-ward (head = anchor, witness = arrived nodes shared with goal + departed slot node); b_walks ctor flag as control. tests/test_derivation.py 5→8 tests, suite 40→43 passing: appendix golden master (composed M:[M,ALL] depth 2, final [M,ALL,H], done), stuck-at-entry with b_walks=False, and held-M:[M,ALL] adoption via the two-ended guard with no walk (unlicensed under old guard). Findings: the drift oracle from 76470b8's message lives on the derivation-model branch (50a9f5e), not algebra — this branch's pytest suite is the guardrail; engine still uses gap()/excess() naming vs the doc's underfit/overfit (renamed in docs at 62ed466) — pending rename, left untouched to keep the port diff reviewable. Anchor refinement toward ν_A's node resolution is not implemented (mirrors the probe: no-ops in the tested scenario); same as refine() which only expands.

*Relevance: high*
*Context: Porting two-party Def 17 + Def 14 into the engine core*
*Tags: kalvin derivation engine def17 def14 port tests*

---
*Observed: 2026-09-14T15:51:12.681Z*
