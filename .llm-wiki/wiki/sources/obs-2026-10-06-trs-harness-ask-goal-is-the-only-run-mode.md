---
type: source
title: "Observation: TRS harness: ask==goal is the only run mode"
tags:
  - trs
  - harness
  - ask-goal
  - simplification
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-trs-harness-ask-goal-is-the-only-run-mode
relevance: medium
observed_at: 2026-10-06T13:36:08.273Z
source_context: Making ask==goal the only TRS harness mode
---

# 🔍 Observation: TRS harness: ask==goal is the only run mode

ask==goal is now the ONLY TRS harness mode (user ruling: 'trs is always an ask-goal run'). --ask-goal flag removed from both trs_harness.main and harness.py --model trs; TrsRunner.run(source, resolver) unconditionally: compile → last `==` ask + goal = two entries (ask first, goal answers) → all other klines prime the domain (direct Domain.add, harness fact not model admission) → rationalise the two. The old present-every-entry loop is gone entirely. Scripts without a findable ask==goal pair raise ValueError ('ask==goal: no `==` ask with a findable goal kline') — main() reports it as an error. Module docstring rewritten: every run is an ask==goal configuration. Tests: test_scripts_run_and_store now asserts 2 steps; test_compiled_entry_maps_to_a_kline switched from a bare sentence (no ask — now rejected) to SCRIPT; ask-goal tests renamed (test_the_two_entries_are_ask_then_goal, test_a_script_without_an_ask_is_rejected). 158 passing. wdmh unchanged through the default path. Uncommitted.

*Relevance: medium*
*Context: Making ask==goal the only TRS harness mode*
*Tags: trs harness ask-goal simplification*

---
*Observed: 2026-10-06T13:36:08.273Z*
