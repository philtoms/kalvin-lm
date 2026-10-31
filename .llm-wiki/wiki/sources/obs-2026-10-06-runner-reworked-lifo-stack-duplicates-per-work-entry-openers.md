---
type: source
title: "Observation: Runner reworked: LIFO stack, duplicates, per-work-entry openers"
tags:
  - trs
  - runner
  - stack
  - lifo
  - duplicates
  - per-entry-opener
  - amendment-18
  - cycles
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-runner-reworked-lifo-stack-duplicates-per-work-entry-openers
relevance: high
observed_at: 2026-10-06T10:17:26.954Z
source_context: Reworking the rationalise queue to stack semantics with per-entry openers
---

# ⭐ Observation: Runner reworked: LIFO stack, duplicates, per-work-entry openers

RUNNER REWORKED to the user's three queue rules (amendment 18): (1) the work queue is an ORDINARY STACK — duplicates allowed, seen-dedupe deleted; (2) next work entry = queue.pop() (latest queued, LIFO); (3) the OPENER is cleared on pop and set to the work entry's FIRST candidate — it does not change for that work entry. Traces are now PER WORK ENTRY: re-entering products are new work entries opening their own traces; the cross-reentry opener propagation is gone. Spec alignment: Definition 9's trace paragraph rewritten (per work entry, opened by the first candidate), R1 restated ('formed on the work entry'; the fuller proof paragraph that had been lost in the earlier non-atomic failure was rebuilt), D10 updated, §5 gains the work-stack statement (duplicates, latest-first, arrival not guaranteed — bounds are strategy), changelog (18). MAJOR BEHAVIOUR CHANGE: the illegal MHALL:[word] store burst at wdmh step 1 is GONE (no S1 among the sentence's first-round candidates → nothing stores; the vacuous cross-reentry licensing path no longer exists). REMAINING store class (the live form of the open illegal-store ruling): sibling-S1-licensed first candidates — e.g. Subject:[Mary] at step 2, ⟨AB,{A}⟩ in the model fixture: the work entry's first candidate (often misfit-shaped) stored when a LATER candidate on the same entry reaches S1. NEW CYCLE CLASS discovered: a self-witnessing fit product (candidate ⟨2,{3}⟩ → product ⟨2, witness(2)⟩ = ⟨2,{2}⟩ — the work entry itself) re-queues the entry forever under duplicates-allowed; LIFO then STARVES products queued deeper (the R3 product ⟨3,{1,2}⟩ never popped); runs bound-loop honestly (§7). Documented in test_lifo_cycles_starve_queued_products (bound end records the min-effort pending at E=0). Ends lists now flood with duplicate halts (each duplicate pop records an end) — presentation groups them; fine. Tests updated to the specified mechanics: 158 passing (model: first-candidate store, reentry fixture, D-live fixture — run 1 now bound-loops with the rescue store ⟨AB,{C}⟩ preventing the emptied halt; harness: step 1 stores []). Uncommitted.

*Relevance: high*
*Context: Reworking the rationalise queue to stack semantics with per-entry openers*
*Tags: trs runner stack lifo duplicates per-entry-opener amendment-18 cycles*

---
*Observed: 2026-10-06T10:17:26.954Z*
