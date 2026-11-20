---
type: source
title: "Observation: rationalisation stack migrated to kalvin KLine"
tags:
  - rationalisation
  - types
  - migration
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-rationalisation-stack-migrated-to-kalvin-kline
relevance: critical
observed_at: 2026-11-17T09:00:05.161Z
source_context: KLine type migration across the rationalisation stack
---

# 🔴 Observation: rationalisation stack migrated to kalvin KLine

User chose option A (full migration): the whole rationalisation stack now uses kalvin.kline.KLine; algebra.Kline is retired. KLine.nodes is an ordered list supporting duplicate entries — the stated motivation. algebra.py lost the Kline class and gained `fingerprint(k: KLine) -> int` as a free function (σ | can(nodes)); `can()`/`overfit()` widened to Iterable. Domain.witness() now returns list[KNode] ([] when unheld). The identity bar `c.adj != frozenset([c.sig])` became `not is_identity(c)` (kalvin.kline, nodes == [signature]) in both cogitator.py and model.py. Semantic edge: {S:[S,S]} was unrepresentable under frozenset (collapsed to identity); as a list it is a two-node canon — new input-space behaviour, not a change to old behaviour. Tests: test_kline_rejects_empty_signature removed (KLine's constructor does not enforce non-empty signatures — the rule belonged to the retired class).

*Relevance: critical*
*Context: KLine type migration across the rationalisation stack*
*Tags: rationalisation types migration*

---
*Observed: 2026-11-17T09:00:05.161Z*
