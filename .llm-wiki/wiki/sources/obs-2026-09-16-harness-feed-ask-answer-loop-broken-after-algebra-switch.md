---
type: source
title: "Observation: Harness feed→ask→answer loop broken after algebra switch"
tags:
  - dialogue
  - harness
  - engine
  - architecture
status: observation
created: 2026-09-16
updated: 2026-09-16
slug: obs-2026-09-16-harness-feed-ask-answer-loop-broken-after-algebra-switch
relevance: high
observed_at: 2026-09-16T13:46:08.860Z
source_context: Harness fitness review after algebra-backed switch
---

# ⭐ Observation: Harness feed→ask→answer loop broken after algebra switch

Reviewed src/dialogue/harness.py after the algebra switch (hop layer, commit 23fd75d) and the new scripted ASK compilation (`==` goal-targeted, c900cf5; UNKNOWN unified into ASK, b0d9403). Verdict: the harness mechanics (compile → annotate-groups → batch scaffolding → feed → present) still work, but the core feed→ask→answer loop is dead. Three breaks: (1) the compiled `==` ask `WDMH:[]` is stamped S4 and Engine._fast_route refuses every S4-stamped feed (a path built for declined replies), so the question never enters the work list — the algebra (§13) says the S4 ask IS the queued entry that drives strategy; (2) the MTS canon `WDMH:[what,did,Mary,have]` (the algebra's A0) is S1-stamped, grounds on arrival, and never queues either — both forms of the question bypass cogitation; (3) Harness._answer answers proposals by node-equality + synthesized S3 countersigns (reciprocal-pair era) — a done proposal is value-equal to the goal, never node-equal to a script kline, so it always escalates and the default supervisor declines it. Only `-e` (StructuralSupervisor.grade) has the value-equality rule. Probes: dev/dialogue/probe_ask_fate.py, probe_ks_compile_dump.py, probe_hop_wdmh.py.

*Relevance: high*
*Context: Harness fitness review after algebra-backed switch*
*Tags: dialogue harness engine architecture*

---
*Observed: 2026-09-16T13:46:08.860Z*
