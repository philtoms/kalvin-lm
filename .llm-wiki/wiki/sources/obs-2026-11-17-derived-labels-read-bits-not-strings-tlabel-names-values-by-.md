---
type: source
title: "Observation: Derived labels read bits, not strings; _tlabel names values by their own labels"
tags:
  - algebra
  - labels
  - mts
  - harness
  - bit-accuracy
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-derived-labels-read-bits-not-strings-tlabel-names-values-by-
relevance: high
observed_at: 2026-11-17T16:41:57.473Z
source_context: Making _provenance usable — meaningful derived labels
---

# ⭐ Observation: Derived labels read bits, not strings; _tlabel names values by their own labels

Bit-accuracy is why derived labels must read the value, not the operands' label strings: in wdmh.ks, WDMH (what, did, Mary, have) vs MHALL (Mary, had, a, little, lamb) — had and have share the initial H but are different word bits, so fit = Mary alone (M), underfit = WDH, overfit = HALL. Character/initial set arithmetic on MTS strings would wrongly claim fit=MH. Consequent harness change: _tlabel precedence flipped — a value that carries a label is named by it (getattr(v,'label') before the script labels dict), and _event_line passes ev.frame as the KNode (not int()) so frames render via their own labels too. Whole trace table is uniformly MTS: reentry asks, F/U/O, frames, stalls (M:[], D:[] where they were Mod:[], Det:[]).

*Relevance: high*
*Context: Making _provenance usable — meaningful derived labels*
*Tags: algebra labels mts harness bit-accuracy*

---
*Observed: 2026-11-17T16:41:57.473Z*
