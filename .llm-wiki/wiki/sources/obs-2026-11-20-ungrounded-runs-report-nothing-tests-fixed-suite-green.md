---
type: source
title: "Observation: Ungrounded runs report nothing — tests fixed, suite green"
tags:
  - cogitator
  - tests
  - ruling
  - docs
status: observation
created: 2026-11-20
updated: 2026-11-20
slug: obs-2026-11-20-ungrounded-runs-report-nothing-tests-fixed-suite-green
relevance: high
observed_at: 2026-11-20T08:18:25.008Z
source_context: Fixing tests and docs after the ungrounded-runs-report-nothing ruling
---

# ⭐ Observation: Ungrounded runs report nothing — tests fixed, suite green

User ruled: ungrounded runs are completely normal and expected, without reporting. Applied: (1) one-shot rationalise() wrapper returns None when the run never finishes — no IndexError; (2) test_a_served_stall_reenters_with_fresh_candidates rewritten around a groundable two-serve scenario (ask ABCD canon, goal k(AD,A,D): U=BC served by k(BC,B,C), F=AD served by k(AD,A,D)) so the serve→retry→frame→claim path is observable in a report that actually leaves; (3) test_an_unserved_stall_stands → test_an_unserved_stall_stands_silent: stalls broadcast live through the sink, r == [] (no report), domain untouched; (4) the two harness tests now run scaffold=True (they were written for scaffolding-on-demand and silently broke when run() defaulted to scaffold=False — held klines went to domain, breaking domain ⊆ scaffolding assertions). Suite: 163 passed, 16 xfailed, zero failures — first fully green run since the redesign began. docs/rationalisation.md §2.1: ungrounded runs are normal, report nothing; WALK already stated the mechanism.

*Relevance: high*
*Context: Fixing tests and docs after the ungrounded-runs-report-nothing ruling*
*Tags: cogitator tests ruling docs*

---
*Observed: 2026-11-20T08:18:25.008Z*
