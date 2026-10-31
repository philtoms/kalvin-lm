---
type: source
title: "Observation: trs/model.py stripped to Cogitator-only support"
tags:
  - trs
  - refactor
  - dead-code
  - cogs
status: observation
created: 2026-10-31
updated: 2026-10-31
slug: obs-2026-10-31-trs-model-py-stripped-to-cogitator-only-support
relevance: high
observed_at: 2026-10-31T08:40:25.247Z
source_context: Refactoring src/trs/model.py to strip non-Cogitator code
---

# ⭐ Observation: trs/model.py stripped to Cogitator-only support

Refactored src/trs/model.py to contain only code supporting the Cogitator under the TRS harness. Removed: r2_products + r3_substitute (legacy §5 rewrite rules superseded by Cogitator reentry, only tests used them), Domain.exact/overlap/covered (unused lookups; covered served only r3_substitute), Domain.__len__, Kline.merge, Cogitator._pair_effort (all dead — zero callers), Rationalisation.rounds (never read). Kept the grading measures (significance/entropy/effort/BETA/decompose) — effort is the halt measure on the runtime path; significance/entropy are Definition 13, effort's base and the property tests' oracle. Also removed the 9 legacy rewrite-rule tests and their generators (s2/s3_scenarios) plus exports in src/trs/__init__.py. dev/probes/probe_ask_step2_trace.py and probe_mhall_walk.py import r2_products3 (already nonexistent) — stale, untouched. Verified: full suite 167 passed/16 xfailed (trs subset 42→33 passed, 9 removed), harness output on mhall.ks and wdmh.ks byte-identical pre/post via KALVIN_DATA_DIR + PYTHONPATH before-copy diff.

*Relevance: high*
*Context: Refactoring src/trs/model.py to strip non-Cogitator code*
*Tags: trs refactor dead-code cogs*

---
*Observed: 2026-10-31T08:40:25.247Z*
