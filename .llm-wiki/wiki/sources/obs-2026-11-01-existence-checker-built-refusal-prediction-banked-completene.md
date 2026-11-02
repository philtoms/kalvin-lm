---
type: source
title: "Observation: Existence checker built — refusal prediction banked, completeness gap measured at ~20%"
tags:
  - rationalisation
  - t4
  - completeness
  - refusal
  - prediction
  - existence-checker
status: observation
created: 2026-11-01
updated: 2026-11-01
slug: obs-2026-11-01-existence-checker-built-refusal-prediction-banked-completene
relevance: critical
observed_at: 2026-11-01T08:32:49.644Z
source_context: Building the T4 certificate existence checker and measuring completeness
---

# 🔴 Observation: Existence checker built — refusal prediction banked, completeness gap measured at ~20%

T4 instrument built and the refusal half banked. src/rationalisation/existence.py: exists_certificate(d0, entry, goal) decides certificate availability via a fixpoint over abstract reentries — explores every pairing the rules allow, settles every side the rules permit, opens exchanges when a region is reached twice (provenance-tracked: the run's exchange parties can be value-identical, differing only in parentage). Deliberate over-approximations (documented, each safe for refusal): frames global, path/seen/covered bars ignored, exchanges = two reachings not two stalls, no bounds. REFUSAL PREDICTION MACHINE-CHECKED: with T3, exists=False ⇒ no schedule can ground — property-tested (grounding ⇒ exists) over random scenarios, corpus consistent (all 3 scenarios grounds⇒exists), the exhaustion regression refused before running, disjoint-content refused. COMPLETENESS MEASURED (400-scenario sweep): ~60% ground, ~20% refused correctly, ~20% carry a certificate the deterministic run MISSES — attribution dominated by the in_path bar (runs halt 'exhausted' on pairings the checker rightly explores). The first named completeness loss is the sig-level path bar itself — the same bar that fix A tried to relax and that broke mhall: the tension is now quantified (relaxing it loses mhall's grounding; keeping it forfeits ~20% of available certificates). Probe xfailed as the permanent instrument. Checker debugging lessons: (1) the opening reentry must be tag-distinguished — mhall's opening u-child equals the entry kline; (2) _side must read the region slot not the tag slot (a bug that made every childed side settle as 'empty'); (3) settle queries must not short-circuit child exploration; (4) frames mint at pairing time regardless of the row's fate; (5) the opening pairs with the goal ALONE — candidate pairings for it exist in no run. Suite: 176 passed, 17 xfailed. docs/rationalisation.md §4 rewritten: Prediction and the measured gap.

*Relevance: critical*
*Context: Building the T4 certificate existence checker and measuring completeness*
*Tags: rationalisation t4 completeness refusal prediction existence-checker*

---
*Observed: 2026-11-01T08:32:49.644Z*
