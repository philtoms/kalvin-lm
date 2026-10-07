---
type: source
title: "Observation: ask==goal configuration: two entries, rest primes the domain"
tags:
  - trs
  - harness
  - ask-goal
  - priming
  - scripts
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-ask-goal-configuration-two-entries-rest-primes-the-domain
relevance: high
observed_at: 2026-10-06T13:09:54.342Z
source_context: Implementing ask==goal configuration in the TRS harness
---

# ⭐ Observation: ask==goal configuration: two entries, rest primes the domain

ASK==GOAL CONFIGURATION implemented in the TRS harness (user request): '--ask-goal' flag (TrsRunner.run(ask_goal=True) + CLI in both trs_harness.main and harness.py --model trs). Semantics: the script's LAST `==` ask and its goal kline are held as the two ENTRIES (ask first, goal answers); ALL other compiled klines PRIME the domain directly before rationalisation (like the old seed — direct Domain.add of the content-masked klines, bypassing R1; this is a HARNESS configuration, not a model admission). Pair selection: last ask with a findable goal ('the final question is the point of the script' — first-found wrongly picked the imported MHALL==SVO in wdmh); goal = LAST label match ('the fullest authored definition' — first-match picked the compiler's intermediate MHALL:[had,ALL] instead of the full-sentence canon MHALL:[Mary,had,a,little,lamb]). wdmh run: step 1 = WDMH ask (S1 vs primed canon), stores WDMH:[Mary], ~60 acquire asks raised; step 2 = MHALL goal, answers 2 proposals via discharge. Refactor: run() loop extracted to _step(); _run_ask_goal added; is_ask imported from kalvin.kline. Tests: test_ask_goal_holds_two_entries_primes_the_rest, test_ask_goal_without_a_pair_raises (ValueError); 158 passing. NOTE the two non-atomic-write failures this round (harness.py add_argument insert; verified absent by grep, re-applied with temp-file+os.replace and in-command verification). NOTE: both steps store ⟨σ_E,{Mary}⟩ openers — the sibling-S1 first-candidate store (open illegal-store ruling) is visible in this configuration too. Uncommitted.

*Relevance: high*
*Context: Implementing ask==goal configuration in the TRS harness*
*Tags: trs harness ask-goal priming scripts*

---
*Observed: 2026-10-06T13:09:54.342Z*
