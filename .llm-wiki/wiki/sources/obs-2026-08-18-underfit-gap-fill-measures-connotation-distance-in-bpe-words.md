---
type: source
title: "Observation: Underfit gap-fill measures connotation distance in BPE words"
tags:
  - dialogue
  - engine
  - s2
  - gap-fill
  - bpe
status: observation
created: 2026-08-18
updated: 2026-08-18
slug: obs-2026-08-18-underfit-gap-fill-measures-connotation-distance-in-bpe-words
relevance: high
observed_at: 2026-08-18T13:10:15.458Z
source_context: "Lean harness S2 proposals: asymmetric connotation distance"
---

# ⭐ Observation: Underfit gap-fill measures connotation distance in BPE words

Underfit gap-fill in src/dialogue/pivot_fill.py now walks a connotation chain measured on the BPE half of node values. Seeds: gap-covering grounded klines whose head is not connoted by another gap-covering kline (upstream rule — makes W>Q(uery)>O distinguishable from W>O). Walk: word→word edges matched on BPE token ids. Grade: compose_terminal over decay(hops), sorted best-first. mhall measurement: W>O grades the ALL fill 0xfc (1 hop); W>Q(uery)>O grades ALL 0xfb (2 hops) — greater distance, less significance, as specified. Also required this session: ks parser fix (nested scope newline-consumption swallowed following annotations), ast_emitter nested-scope head inline-annotation binding (W > Q(uery) binds node to 'query' word), and an engine cogitate groundable-promotion arm (groundable entries re-fed after a cascade sweep stalled in STM). Known artifact: the gap's own word surfaces as a fill (WDMH:[Mary, DH, what] at 0xfc).

_Relevance: high_
_Context: Lean harness S2 proposals: asymmetric connotation distance_
_Tags: dialogue engine s2 gap-fill bpe_

---

_Observed: 2026-08-18T13:10:15.458Z_
