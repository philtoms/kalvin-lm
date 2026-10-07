---
type: source
title: "Observation: Verify step 1 settled: goal rationalises before the ask"
tags:
  - trs
  - redesign
  - reentry
  - verify-step-1
  - goal-first
status: observation
created: 2026-10-06
updated: 2026-10-06
slug: obs-2026-10-06-verify-step-1-settled-goal-rationalises-before-the-ask
relevance: high
observed_at: 2026-10-06T14:54:19.163Z
source_context: Step-by-step rationalisation reentry redesign; verify step 1
---

# ⭐ Observation: Verify step 1 settled: goal rationalises before the ask

VERIFY STEP 1 settled (user ruling: 'goal before ask'): TrsRunner.run now presents the GOAL kline as step 1, the ASK as step 2 (one-line swap; test renamed test_the_two_entries_are_goal_then_ask). mhall.ks ask==goal run now: step 1 = SVO:[Subject,Verb,Object] (S1 against primed domain — no stores, 6 R4 halts: the three SVO-shaped residuals and their mirrors), step 2 = MHALL ask (S1, stores MHALL:[Mary], Query:[Query], Object:[Object], Verb:[Verb], Subject:[Subject], ALL:[Det] — the sibling-S1 opener stores again). Context: the user is REDESIGNING rationalisation reentry step by step (recursion at R2/R3) using the existing build as the baseline — 'use the existing solution to see how far we get, and exactly where we diverge'. mhall.ks is the test case: prove MHALL == SVO. Verified facts for the redesign baseline: 16 compiled entries; ask pair = MHALL(op=ASK,goal='SVO') + SVO(op=CANONICALISES,nodes=3) goal kline; 14 klines prime D directly; goal rationalises first, ask second. 158 total tests passing. Uncommitted.

*Relevance: high*
*Context: Step-by-step rationalisation reentry redesign; verify step 1*
*Tags: trs redesign reentry verify-step-1 goal-first*

---
*Observed: 2026-10-06T14:54:19.163Z*
