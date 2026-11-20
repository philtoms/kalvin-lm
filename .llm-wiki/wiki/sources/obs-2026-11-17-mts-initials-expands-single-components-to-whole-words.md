---
type: source
title: "Observation: mts_initials expands single components to whole words"
tags:
  - mts
  - labels
  - readability
status: observation
created: 2026-11-17
updated: 2026-11-17
slug: obs-2026-11-17-mts-initials-expands-single-components-to-whole-words
relevance: medium
observed_at: 2026-11-17T16:45:35.729Z
source_context: Expanding single-character derived tokens back to words
---

# 🔍 Observation: mts_initials expands single components to whole words

mts_initials refined: it now returns mts(parts) — the same single-part rule as every other label site. A single-component derived value stands whole (fit(WDMH, MHALL) labels "Mary", not "M"; stalls read Mod:[], Det:[] again), compounds take one character per word (WDH, HALL). One MTS rule system-wide: single part whole, compound one uppercased character per part (abbreviation parts whole). Canonical wdmh example now: A=WDMH, C=MHALL, F=Mary, U=WDH, O=HALL.

*Relevance: medium*
*Context: Expanding single-character derived tokens back to words*
*Tags: mts labels readability*

---
*Observed: 2026-11-17T16:45:35.729Z*
