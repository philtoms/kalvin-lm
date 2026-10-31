---
type: source
title: "Observation: TRS harness hooked up: .ks scripts run the significance-routed model"
tags:
  - trs
  - harness
  - scripted-runs
  - dialogue
  - kscript
  - discharge
  - bootstrap
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-trs-harness-hooked-up-ks-scripts-run-the-significance-routed
relevance: high
observed_at: 2026-10-06T06:08:58.465Z
source_context: Hooking up the dialogue harness and scripted runs for the TRS model
---

# ⭐ Observation: TRS harness hooked up: .ks scripts run the significance-routed model

TRS HARNESS HOOKED UP (dev/dialogue/trs_harness.py + --model trs on dev/dialogue/harness.py): the same .ks scripts now run through the significance-routed model. Design: TrsRunner compiles via the SAME compile_source (shared word_bits, resolver for imports), maps each KValue to a TRS Kline (int sig, frozenset of int nodes), presents entries in AUTHORED ORDER — one step per entry, D live throughout (D14). BOOTSTRAP: word identities are SEEDED into D as the I-1 substrate (tokenizer-derivable facts, same authority as the engine path's batch primes) — needed because from an empty D nothing can enter (Definition 9 needs an overlapping neighbour; R1 is sole admission for what the run LEARNS). PROPOSAL DISCHARGE (Definition 15) runs cross-step: a later presented entry discharges a pending proposal iff (σ_E ∨ can(N_E)) ∧ target ≠ 0 — the raiser never answers itself (bug fixed: initial version let the raising entry discharge its own proposals). Presenter (non-judging): per step — stores, halts grouped by reason with exemplars + counts; end — domain held (canon/claim marked), pending proposals (most significant first, D13), discharged count. CLI: `PYTHONPATH=src:. python -m dev.dialogue.harness script.ks --model trs` (or -m dev.dialogue.trs_harness directly); -p is engine-only for now (continuous operation later). FIRST READINGS from mhall/wdmh runs: (1) the R1 rewrite visible in miniature — the sentence entry's five word fit-products each ground at S1 against seeded identities, licensing five openers MHALL:[word] — 'the sentence means Mary' one licensed partial claim per word; (2) grammar entries (Subject < M etc.) teach NOTHING new to this model: the entry's signature is only a lookup key — candidates carry can(N_E), so the Mary↔Subject link never enters a candidate head (engine-shaped scripts meet the model's character — a finding, not a bug); (3) sentence-level entries fan out heavily (every overlapping neighbour × R2 misfit products → many E=1 acquire halts; step 9 hit the 64-round bound — honest §7 behaviour); (4) wdmh: 917 proposals discharged by later entries, 62 open — the cross-step discharge machinery works, though overlap-discharge is coarse (bit-rich sigs overlap easily). Tests: tests/test_trs_harness.py (3) — scripted run stores + I-1 held, discharge-by-later-only, entry mapping. 158 passing. Uncommitted.

*Relevance: high*
*Context: Hooking up the dialogue harness and scripted runs for the TRS model*
*Tags: trs harness scripted-runs dialogue kscript discharge bootstrap*

---
*Observed: 2026-10-06T06:08:58.465Z*
